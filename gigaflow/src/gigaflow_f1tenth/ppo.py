"""Recurrent continuous PPO (paper-aligned hyperparameters, direct tensors)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from gigaflow_f1tenth.buffers import CompactRolloutBatch
from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.critic import CentralValueCritic, critic_values_over_time
from gigaflow_f1tenth.model import (
    ACT_LIMIT,
    LOG_STD_MAX,
    LOG_STD_MIN,
    ConditionedActor,
    _log_prob_pre_tanh,
    orthogonal_init_,
)

RESUME_STATE_VERSION = 3


@dataclass(frozen=True)
class AdaptiveFilterState:
    """EWMA of max |A|; eta = eta_scale * ewma (paper algorithm)."""

    ewma_max_abs_adv: float
    beta: float
    eta_scale: float
    initialized: bool = False

    @property
    def eta(self) -> float:
        return self.eta_scale * self.ewma_max_abs_adv


# CUDA BF16 autocast for neural actor/critic forwards only (never FP16).
AMP_DTYPE = torch.bfloat16


@dataclass
class PPOUpdateStats:
    retention: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    learning_rate: float
    filter_eta: float
    early_stopped: bool = False
    epochs_completed: int = 0
    pre_update_approx_kl: float = 0.0
    logp_delta_max: float = 0.0
    logp_delta_mean: float = 0.0
    action_pretanh_max_abs: float = 0.0
    surrogate_loss: float = 0.0
    entropy_loss: float = 0.0
    entropy_coef: float = 0.0
    candidate_kl: float = 0.0
    full_rollout_kl: float = 0.0
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    actor_update_to_weight_norm: float = 0.0
    actor_steps: int = 0
    critic_steps: int = 0
    actor_rolled_back: bool = False
    actor_rollback_full_rollout: bool = False
    actor_rollback_count: int = 0
    consecutive_actor_rollbacks: int = 0
    actor_lr_safety_multiplier: float = 1.0
    rollback_stop_requested: bool = False
    actor_kl_rejection_reason: str = ""
    actor_kl_warmup_active: bool = False
    rollback_free_accepted_updates: int = 0
    telemetry: dict[str, float] = field(default_factory=dict)


class CollectEvaluateParityError(RuntimeError):
    """Frozen-weight collect/evaluate disagreement — stop before optimizer step."""


@dataclass
class PPOState:
    actor: ConditionedActor
    critic: CentralValueCritic
    actor_optimizer: Any
    critic_optimizer: Any
    actor_scheduler: Any
    critic_scheduler: Any
    filter_state: AdaptiveFilterState
    update_index: int
    actor_lr_safety_multiplier: float
    actor_rollback_count: int
    consecutive_actor_rollbacks: int
    rollback_updates: list[int]
    rollback_free_accepted_updates: int
    carry_hidden: Tensor | None = None
    carry_reset: Tensor | None = None
    amp_scaler: Any | None = None


@dataclass(frozen=True)
class PreparedPPOInputs:
    """Reconstructed tensors required for a PPO update (not stored in compact rollouts)."""

    sensor_obs: Tensor  # [T, S, sensor_dim]
    ego_state: Tensor  # [T, S, D_ego]
    other_agents: Tensor  # [T, S, N, D_el]
    other_mask: Tensor  # [T, S, N]
    old_values: Tensor | None = None  # [T, S]; computed if None
    last_values: Tensor | None = None  # [S]; bootstrap past the final step
    bootstrap_values: Tensor | None = None  # [T, S]; V(next_state) per step


def entropy_coefficient(cfg: ExperimentConfig, update_index: int) -> float:
    ppo = cfg.ppo
    if ppo.ent_coef_initial is None:
        assert ppo.ent_coef is not None
        return float(ppo.ent_coef)
    assert ppo.ent_coef_final is not None
    assert ppo.ent_anneal_updates is not None
    k = min(max(int(update_index), 0), int(ppo.ent_anneal_updates))
    if k >= int(ppo.ent_anneal_updates):
        return float(ppo.ent_coef_final)
    phase = math.pi * float(k) / float(ppo.ent_anneal_updates)
    return float(
        ppo.ent_coef_final
        + 0.5
        * (ppo.ent_coef_initial - ppo.ent_coef_final)
        * (1.0 + math.cos(phase))
    )


@runtime_checkable
class PPOLearner(Protocol):
    def state(self) -> PPOState:
        ...

    def compute_advantages(
        self,
        batch: CompactRolloutBatch,
        values: Any,
        last_values: Any,
        next_values: Any | None = None,
    ) -> tuple[Any, Any]:
        """Return (advantages, returns) over active/trainable masks; bootstrap timeouts."""

    def update(
        self,
        batch: CompactRolloutBatch,
        prepared: PreparedPPOInputs,
    ) -> PPOUpdateStats:
        """Recurrent minibatches preserve agent sequences; filter actor and critic."""


def initial_filter_state(cfg: ExperimentConfig) -> AdaptiveFilterState:
    return AdaptiveFilterState(
        ewma_max_abs_adv=0.0,
        beta=cfg.ppo.adaptive_filter_beta,
        eta_scale=cfg.ppo.adaptive_filter_eta_scale,
        initialized=False,
    )


def _atanh(x: Tensor) -> Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squashed_gaussian_log_prob(
    mu: Tensor,
    log_std: Tensor,
    actions: Tensor,
    *,
    act_limit: float = ACT_LIMIT,
) -> tuple[Tensor, Tensor]:
    """Log-prob and entropy for tanh-squashed Gaussian actions."""
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)
    dist = Normal(mu, std)
    a = (actions / act_limit).clamp(-0.999999, 0.999999)
    pre_tanh = _atanh(a)
    logp = dist.log_prob(pre_tanh).sum(dim=-1)
    # Match model._squashed_gaussian correction term (act_limit=1).
    logp = logp - (
        2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    ).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    return logp, entropy


def evaluate_actions_sequence(
    actor: Any,
    sensor_obs: Tensor,
    private_condition: Tensor,
    hidden: Tensor,
    actions: Tensor,
    reset_mask: Tensor | None = None,
    pre_tanh: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Score stored action sequences; prefers actor.evaluate_actions_sequence.

    Fallback walks encode_trunk + GRU + MLP head when the conditioned-model
    actor has not yet exposed a first-class sequence evaluator (interface gap).
    Returns log_prob [B,T], entropy [B,T], final_hidden [B,H].
    Prefer ``pre_tanh`` (collection sample) over atanh(action) recovery.
    """
    if hasattr(actor, "evaluate_actions_sequence"):
        try:
            return actor.evaluate_actions_sequence(
                sensor_obs,
                private_condition,
                hidden,
                actions,
                reset_mask=reset_mask,
                pre_tanh=pre_tanh,
            )
        except TypeError:
            # Tiny/test actors may not accept pre_tanh yet.
            return actor.evaluate_actions_sequence(
                sensor_obs,
                private_condition,
                hidden,
                actions,
                reset_mask=reset_mask,
            )
    required = ("encode_trunk", "gru", "net", "mu_layer", "log_std_layer")
    if not all(hasattr(actor, name) for name in required):
        raise TypeError(
            "actor must implement evaluate_actions_sequence or expose "
            "encode_trunk/gru/net/mu_layer/log_std_layer for PPO sequence scoring"
        )
    b, t, _ = sensor_obs.shape
    if private_condition.dim() == 2:
        cond_seq = private_condition.unsqueeze(1).expand(b, t, -1)
    elif private_condition.dim() == 3:
        cond_seq = private_condition
    else:
        raise ValueError(
            f"private_condition shape {tuple(private_condition.shape)} unsupported"
        )
    logps: list[Tensor] = []
    ents: list[Tensor] = []
    h = hidden
    apply_reset = getattr(actor, "_apply_reset_mask", None)
    for i in range(t):
        rm = None if reset_mask is None else reset_mask[:, i]
        if apply_reset is not None:
            h = apply_reset(h, rm)
        elif rm is not None:
            h = h * (~rm).unsqueeze(-1).to(dtype=h.dtype)
        trunk = actor.encode_trunk(sensor_obs[:, i], cond_seq[:, i])
        out, h_n = actor.gru(trunk.unsqueeze(1), h.unsqueeze(0).contiguous())
        h = h_n.squeeze(0)
        features = out.squeeze(1)
        net_out = actor.net(features)
        mu = actor.mu_layer(net_out)
        log_std = actor.log_std_layer(net_out)
        if pre_tanh is not None:
            logp, ent = _log_prob_pre_tanh(
                actor.net,
                actor.mu_layer,
                actor.log_std_layer,
                features,
                pre_tanh[:, i],
            )
        else:
            logp, ent = squashed_gaussian_log_prob(mu, log_std, actions[:, i])
        logps.append(logp)
        ents.append(ent)
    return torch.stack(logps, dim=1), torch.stack(ents, dim=1), h


def _assert_collect_evaluate_parity(
    *,
    new_logp_t: Tensor,
    old_logp: Tensor,
    keep_w: Tensor,
    actions: Tensor,
    pre_tanh: Tensor | None,
    sensor_obs: Tensor,
    condition: Tensor,
    reset_mask: Tensor,
    gru_start: Tensor,
    max_pre_update_kl: float,
    max_logp_delta: float,
) -> tuple[float, float, float, float]:
    """Hard gate: frozen-weight rescoring must match ``old_logp``.

    ``old_logp`` here is already an evaluate-path rescore by the time this
    runs, not the raw collection value; see the call site in ``update`` for
    what this gate does and does not cover.
    """
    for name, t in (
        ("sensor_obs", sensor_obs),
        ("condition", condition),
        ("actions", actions),
        ("old_logp", old_logp),
        ("gru_start", gru_start),
    ):
        if not torch.isfinite(t).all():
            raise CollectEvaluateParityError(f"non-finite {name} before PPO update")
    if not torch.isfinite(reset_mask.to(dtype=torch.float32)).all():
        raise CollectEvaluateParityError("non-finite reset_mask before PPO update")

    action_pretanh_max = 0.0
    if pre_tanh is not None:
        if not torch.isfinite(pre_tanh).all():
            raise CollectEvaluateParityError("non-finite pre_tanh before PPO update")
        recon = torch.tanh(pre_tanh)
        action_pretanh_max = float((recon - actions).abs().max().item())
        if action_pretanh_max > 1.0e-4:
            raise CollectEvaluateParityError(
                f"action/pre_tanh mismatch max_abs={action_pretanh_max:.3e}"
            )

    delta = (new_logp_t - old_logp).abs()
    denom = keep_w.sum().clamp(min=1.0)
    mean_delta = float((delta * keep_w).sum() / denom)
    # max over kept rows; if none kept, treat as 0
    if float(keep_w.sum().item()) > 0.0:
        max_delta = float(delta[keep_w.bool()].max().item())
    else:
        max_delta = 0.0
    log_ratio = new_logp_t - old_logp
    ratio = log_ratio.exp()
    pre_kl = float((((ratio - 1.0) - log_ratio) * keep_w).sum() / denom)

    if max_delta > max_logp_delta:
        raise CollectEvaluateParityError(
            f"collect/evaluate logp delta max={max_delta:.3e} "
            f"mean={mean_delta:.3e} exceeds max_logp_delta={max_logp_delta}"
        )
    if abs(pre_kl) > max_pre_update_kl:
        raise CollectEvaluateParityError(
            f"pre_update_approx_kl={pre_kl:.3e} exceeds "
            f"max_pre_update_kl={max_pre_update_kl}"
        )
    return pre_kl, max_delta, mean_delta, action_pretanh_max


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    done: Tensor,
    timeout: Tensor,
    last_values: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    valid: Tensor | None = None,
    next_values: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """GAE where every episode boundary cuts the trace.

    ``done`` marks true terminals; ``timeout`` marks horizon truncations.
    A truncation bootstraps from ``next_values[t] = V(s'_t)``, the value of the
    state the agent was actually in when the episode was cut — after async
    respawn ``values[t+1]`` belongs to a different episode. Continuing rows
    bootstrap from ``values[t+1]``, or ``last_values`` past the final step.
    """
    if rewards.shape != values.shape or rewards.shape != done.shape:
        raise ValueError("rewards, values, done must share shape [T, S]")
    if timeout.shape != rewards.shape:
        raise ValueError("timeout must match rewards shape [T, S]")
    if last_values.shape != rewards.shape[1:]:
        raise ValueError(
            f"last_values shape={tuple(last_values.shape)}; "
            f"expected {tuple(rewards.shape[1:])}"
        )
    if next_values is None:
        if bool(timeout.any()):
            raise ValueError(
                "next_values (V of the pre-respawn next state) is required to "
                "bootstrap truncated episodes"
            )
    elif next_values.shape != rewards.shape:
        raise ValueError("next_values must match rewards shape [T, S]")
    dtype = values.dtype
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    last_values = torch.nan_to_num(
        last_values, nan=0.0, posinf=0.0, neginf=0.0
    )
    if next_values is not None:
        next_values = torch.nan_to_num(
            next_values.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
    advantages = torch.zeros_like(rewards)
    next_value = last_values.to(dtype=dtype)
    next_advantage = torch.zeros_like(last_values, dtype=dtype)
    zero = torch.zeros_like(next_advantage)
    for step in range(rewards.shape[0] - 1, -1, -1):
        truncated = timeout[step]
        terminal = done[step] & ~truncated
        # Terminals bootstrap nothing; truncations bootstrap their own next
        # state; continuations chain through the following step's value.
        bootstrap = torch.where(terminal, zero, next_value)
        if next_values is not None:
            bootstrap = torch.where(truncated, next_values[step], bootstrap)
        delta = rewards[step] + gamma * bootstrap - values[step]
        carry = (~(terminal | truncated)).to(dtype=dtype)
        next_advantage = delta + gamma * gae_lambda * carry * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    if valid is not None:
        advantages = advantages * valid.to(dtype=dtype)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    returns = torch.nan_to_num(
        advantages + values, nan=0.0, posinf=0.0, neginf=0.0
    )
    return advantages, returns


def adaptive_advantage_keep_mask(
    advantages: Tensor,
    valid: Tensor,
    filter_state: AdaptiveFilterState,
    *,
    enabled: bool = True,
) -> tuple[Tensor, AdaptiveFilterState, float]:
    """Paper filter: eta = eta_scale * EWMA(max |A|); drop |A| < eta.

    Paper EWMA (beta on the *current* max):
    ``ewma <- beta * max_|A| + (1 - beta) * previous``, with the first
    observation initializing ``ewma = max_|A|``. Invalid / non-finite
    advantages never contribute to the max or the keep mask.
    When ``enabled`` is False (ablation), keep all valid samples and freeze EWMA.
    """
    valid_f = valid.to(dtype=torch.bool)
    if not bool(valid_f.any()):
        keep = torch.zeros_like(advantages, dtype=torch.bool)
        return keep, filter_state, float(filter_state.eta)
    if not enabled:
        return valid_f.clone(), filter_state, 0.0

    finite = torch.isfinite(advantages)
    usable = valid_f & finite
    if not bool(usable.any()):
        keep = torch.zeros_like(advantages, dtype=torch.bool)
        return keep, filter_state, float(filter_state.eta)
    abs_adv = advantages.abs()
    max_abs = float(abs_adv[usable].max().item())
    if not (max_abs < float("inf")):
        max_abs = 0.0
    beta = float(filter_state.beta)
    if not filter_state.initialized:
        ewma = max_abs
    else:
        prev = float(filter_state.ewma_max_abs_adv)
        if not (prev < float("inf")):
            prev = 0.0
        ewma = beta * max_abs + (1.0 - beta) * prev
    new_state = AdaptiveFilterState(
        ewma_max_abs_adv=ewma,
        beta=filter_state.beta,
        eta_scale=filter_state.eta_scale,
        initialized=True,
    )
    eta = float(new_state.eta)
    keep = usable & (abs_adv >= eta)
    return keep, new_state, eta


def agents_per_minibatch(minibatch_size: int, rollout_length: int, num_slots: int) -> int:
    """Preserve full agent sequences; minibatch_size is a transition budget."""
    per = max(1, int(minibatch_size) // max(1, int(rollout_length)))
    return max(1, min(per, int(num_slots)))


def export_resume_state(learner: "RecurrentPPO") -> dict[str, Any]:
    """Exact PPO resume payload (actor/critic/optim/sched/filter/carry/AMP/RNG)."""
    st = learner.state()
    actor = st.actor
    critic = st.critic
    if not hasattr(actor, "state_dict") or not hasattr(critic, "state_dict"):
        raise TypeError("actor and critic must expose state_dict for resume")
    payload: dict[str, Any] = {
        "resume_state_version": RESUME_STATE_VERSION,
        "update_index": int(st.update_index),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": st.actor_optimizer.state_dict(),
        "critic_optimizer": st.critic_optimizer.state_dict(),
        "actor_scheduler": (
            st.actor_scheduler.state_dict() if st.actor_scheduler is not None else None
        ),
        "critic_scheduler": (
            st.critic_scheduler.state_dict() if st.critic_scheduler is not None else None
        ),
        "actor_lr_safety_multiplier": float(st.actor_lr_safety_multiplier),
        "actor_rollback_count": int(st.actor_rollback_count),
        "consecutive_actor_rollbacks": int(st.consecutive_actor_rollbacks),
        "rollback_updates": list(st.rollback_updates),
        "rollback_free_accepted_updates": int(st.rollback_free_accepted_updates),
        "filter_state": {
            "ewma_max_abs_adv": float(st.filter_state.ewma_max_abs_adv),
            "beta": float(st.filter_state.beta),
            "eta_scale": float(st.filter_state.eta_scale),
            "initialized": bool(st.filter_state.initialized),
        },
        "carry_hidden": (
            None if st.carry_hidden is None else st.carry_hidden.detach().cpu()
        ),
        "carry_reset": (
            None if st.carry_reset is None else st.carry_reset.detach().cpu()
        ),
        "amp_scaler": (
            st.amp_scaler.state_dict()
            if st.amp_scaler is not None and hasattr(st.amp_scaler, "state_dict")
            else None
        ),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    return payload


def load_resume_state(learner: "RecurrentPPO", payload: dict[str, Any]) -> None:
    version = int(payload.get("resume_state_version", -1))
    if version != RESUME_STATE_VERSION:
        raise ValueError(
            f"unsupported resume_state_version={version}; "
            f"expected {RESUME_STATE_VERSION}; single-optimizer learner states "
            "cannot be resumed safely"
        )
    learner.actor.load_state_dict(payload["actor"])
    learner.critic.load_state_dict(payload["critic"])
    learner.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    learner.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    if payload.get("actor_scheduler") is not None:
        learner.actor_scheduler.load_state_dict(payload["actor_scheduler"])
    if payload.get("critic_scheduler") is not None:
        learner.critic_scheduler.load_state_dict(payload["critic_scheduler"])
    learner.actor_lr_safety_multiplier = float(
        payload["actor_lr_safety_multiplier"]
    )
    learner.actor_rollback_count = int(payload["actor_rollback_count"])
    learner.consecutive_actor_rollbacks = int(
        payload["consecutive_actor_rollbacks"]
    )
    learner.rollback_updates = [
        int(value) for value in payload.get("rollback_updates", ())
    ]
    learner.rollback_free_accepted_updates = int(
        payload.get("rollback_free_accepted_updates", 0)
    )
    fs = payload["filter_state"]
    learner.filter_state = AdaptiveFilterState(
        ewma_max_abs_adv=float(fs["ewma_max_abs_adv"]),
        beta=float(fs["beta"]),
        eta_scale=float(fs["eta_scale"]),
        initialized=bool(fs.get("initialized", True)),
    )
    learner.update_index = int(payload["update_index"])
    learner._apply_actor_safety_lr()
    device = learner.device
    if payload.get("carry_hidden") is not None:
        learner.carry_hidden = payload["carry_hidden"].to(device=device)
    if payload.get("carry_reset") is not None:
        learner.carry_reset = payload["carry_reset"].to(device=device)
    if payload.get("amp_scaler") is not None and learner.scaler is not None:
        learner.scaler.load_state_dict(payload["amp_scaler"])
    if payload.get("torch_rng") is not None:
        torch.set_rng_state(payload["torch_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def _weighted_kl(new_logp: Tensor, old_logp: Tensor, weight: Tensor) -> Tensor:
    log_ratio = new_logp.to(dtype=torch.float32) - old_logp
    ratio = log_ratio.exp()
    denom = weight.sum().clamp(min=1.0)
    return (((ratio - 1.0) - log_ratio) * weight).sum() / denom


def _percentile_metrics(prefix: str, values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("p0", "p1", "p50", "p99", "p99_9", "p100")}
    data = values.detach().to(dtype=torch.float32).flatten()
    quantiles = torch.tensor(
        [0.0, 0.01, 0.5, 0.99, 0.999, 1.0],
        device=data.device,
        dtype=torch.float32,
    )
    result = torch.quantile(data, quantiles).cpu().tolist()
    names = ("p0", "p1", "p50", "p99", "p99_9", "p100")
    return {f"{prefix}_{name}": float(value) for name, value in zip(names, result)}


def _rng_snapshot() -> tuple[Tensor, list[Tensor] | None]:
    cpu = torch.get_rng_state().clone()
    cuda = None
    if torch.cuda.is_available():
        cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return cpu, cuda


def _assert_rng_unchanged(before: tuple[Tensor, list[Tensor] | None]) -> None:
    cpu, cuda = before
    if not torch.equal(cpu, torch.get_rng_state()):
        raise RuntimeError("deterministic actor rescoring mutated the CPU RNG")
    if cuda is not None:
        current = torch.cuda.get_rng_state_all()
        if len(cuda) != len(current) or any(
            not torch.equal(left, right) for left, right in zip(cuda, current)
        ):
            raise RuntimeError("deterministic actor rescoring mutated a CUDA RNG")


def _rescore_actor(
    actor: Any,
    sensor_obs: Tensor,
    condition: Tensor,
    hidden: Tensor,
    actions: Tensor,
    reset_mask: Tensor,
    pre_tanh: Tensor | None,
    *,
    diagnostics: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    before = _rng_snapshot()
    log_std = None
    if diagnostics:
        try:
            result = actor.evaluate_actions_sequence(
                sensor_obs,
                condition,
                hidden,
                actions,
                reset_mask=reset_mask,
                pre_tanh=pre_tanh,
                return_diagnostics=True,
            )
            if len(result) == 4:
                logp, entropy, _, log_std = result
            else:
                logp, entropy, _ = result
        except TypeError:
            logp, entropy, _ = evaluate_actions_sequence(
                actor,
                sensor_obs,
                condition,
                hidden,
                actions,
                reset_mask=reset_mask,
                pre_tanh=pre_tanh,
            )
    else:
        logp, entropy, _ = evaluate_actions_sequence(
            actor,
            sensor_obs,
            condition,
            hidden,
            actions,
            reset_mask=reset_mask,
            pre_tanh=pre_tanh,
        )
    _assert_rng_unchanged(before)
    return logp.to(dtype=torch.float32), entropy.to(dtype=torch.float32), log_std


class RecurrentPPO:
    """Single-machine recurrent continuous PPO with adaptive advantage filtering."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        actor: ConditionedActor,
        critic: CentralValueCritic,
        *,
        total_updates: int | None = None,
        target_kl: float | None = None,
        orthogonal_init: bool = False,
        device: str | None = None,
    ) -> None:
        if not isinstance(actor, nn.Module) or not isinstance(critic, nn.Module):
            raise TypeError("actor and critic must be torch.nn.Module instances")
        self.cfg = cfg
        self.device = torch.device(device or cfg.worlds.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        if orthogonal_init:
            orthogonal_init_(self.actor)
            orthogonal_init_(self.critic)
        self.gamma = float(cfg.ppo.gamma)
        self.gae_lambda = float(cfg.ppo.gae_lambda)
        self.clip_ratio = float(cfg.ppo.clip_ratio)
        self.vf_coef = float(cfg.ppo.vf_coef)
        self.max_grad_norm = float(cfg.ppo.max_grad_norm)
        self.num_epochs = int(cfg.ppo.num_epochs)
        self.minibatch_size = int(cfg.ppo.minibatch_size)
        self.rollout_length = int(cfg.ppo.rollout_length)
        self.value_clip = bool(cfg.ppo.value_clip)
        if self.value_clip:
            raise ValueError("value_clip must be false (paper-aligned PPO)")
        self.amp = bool(cfg.ppo.amp) and self.device.type == "cuda"
        # BF16 does not need loss scaling; keep a disabled scaler for resume shape.
        self.amp_dtype = AMP_DTYPE
        self.filter_enabled = bool(
            cfg.ppo.adaptive_filter_enabled and cfg.ablations.adaptive_filter_enabled
        )
        kl = cfg.ppo.target_kl if target_kl is None else target_kl
        self.target_kl = None if kl is None or float(kl) <= 0.0 else float(kl)
        self.actor_kl_soft = (
            float(cfg.ppo.actor_kl_soft)
            if float(cfg.ppo.actor_kl_soft) > 0.0
            else self.target_kl
        )
        self.actor_kl_hard = (
            float(cfg.ppo.actor_kl_hard)
            if float(cfg.ppo.actor_kl_hard) > 0.0
            else None
        )
        self.actor_lr_backoff = float(cfg.ppo.actor_lr_backoff)
        self.actor_lr_scale_min = float(cfg.ppo.actor_lr_scale_min)
        self.actor_kl_warmup_updates = int(cfg.ppo.actor_kl_warmup_updates)
        self.actor_kl_stop_window_updates = int(cfg.ppo.actor_kl_stop_window_updates)
        self.actor_kl_stop_window_count = int(cfg.ppo.actor_kl_stop_window_count)
        self.actor_lr_recovery_interval_updates = int(
            cfg.ppo.actor_lr_recovery_interval_updates
        )
        self.actor_lr_recovery_multiplier = float(cfg.ppo.actor_lr_recovery_multiplier)
        self.actor_kl_stop_enabled = bool(cfg.ppo.actor_kl_stop_enabled)
        self.max_pre_update_kl = float(cfg.ppo.max_pre_update_kl)
        self.max_logp_delta = float(cfg.ppo.max_logp_delta)
        updates = cfg.ppo.total_updates if total_updates is None else total_updates
        self.total_updates = max(1, int(updates))

        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=float(cfg.ppo.learning_rate)
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=float(cfg.ppo.learning_rate)
        )
        self.actor_scheduler = CosineAnnealingLR(
            self.actor_optimizer, T_max=self.total_updates, eta_min=0.0
        )
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optimizer, T_max=self.total_updates, eta_min=0.0
        )
        # Compatibility aliases for callers that only inspect the actor LR.
        self.optimizer = self.actor_optimizer
        self.scheduler = self.actor_scheduler
        self.actor_lr_safety_multiplier = 1.0
        self.actor_rollback_count = 0
        self.consecutive_actor_rollbacks = 0
        self.rollback_updates: list[int] = []
        self.rollback_free_accepted_updates = 0
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.filter_state = initial_filter_state(cfg)
        self.update_index = 0
        self.carry_hidden: Tensor | None = None
        self.carry_reset: Tensor | None = None
        self._num_slots: int | None = None

    def state(self) -> PPOState:
        return PPOState(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=self.critic_scheduler,
            filter_state=self.filter_state,
            update_index=self.update_index,
            actor_lr_safety_multiplier=self.actor_lr_safety_multiplier,
            actor_rollback_count=self.actor_rollback_count,
            consecutive_actor_rollbacks=self.consecutive_actor_rollbacks,
            rollback_updates=list(self.rollback_updates),
            rollback_free_accepted_updates=self.rollback_free_accepted_updates,
            carry_hidden=self.carry_hidden,
            carry_reset=self.carry_reset,
            amp_scaler=self.scaler,
        )

    def _scheduled_actor_lr(self) -> float:
        values = self.actor_scheduler.get_last_lr()
        return float(values[0] if values else self.cfg.ppo.learning_rate)

    def _apply_actor_safety_lr(self) -> None:
        lr = self._scheduled_actor_lr() * self.actor_lr_safety_multiplier
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr

    def _step_actor_scheduler(self) -> None:
        scheduled = self._scheduled_actor_lr()
        for group in self.actor_optimizer.param_groups:
            group["lr"] = scheduled
        self.actor_scheduler.step()
        self._apply_actor_safety_lr()

    def ensure_carry(self, num_slots: int) -> tuple[Tensor, Tensor]:
        """Allocate or return live GRU carry + pending reset mask."""
        h = int(self.cfg.agents.gru_hidden_dim)
        if (
            self.carry_hidden is None
            or self.carry_reset is None
            or self.carry_hidden.shape[0] != num_slots
        ):
            if hasattr(self.actor, "initial_hidden"):
                self.carry_hidden = self.actor.initial_hidden(
                    num_slots, str(self.device)
                )
            else:
                self.carry_hidden = torch.zeros(
                    num_slots, h, device=self.device, dtype=torch.float32
                )
            self.carry_reset = torch.ones(
                num_slots, device=self.device, dtype=torch.bool
            )
            self._num_slots = num_slots
        assert self.carry_hidden is not None and self.carry_reset is not None
        return self.carry_hidden, self.carry_reset

    def begin_rollout(self, num_slots: int) -> Tensor:
        """Snapshot detached carry into rollout-start hidden (shape-stable)."""
        hidden, _ = self.ensure_carry(num_slots)
        # Clone so callers can retain h0 while advance_carry mutates live carry.
        return hidden.detach().clone()

    def advance_carry(
        self,
        next_hidden: Tensor,
        *,
        done: Tensor,
        reset_mask: Tensor | None = None,
    ) -> None:
        """Update live carry after a collection step; clear only reset/done rows."""
        hidden, pending = self.ensure_carry(int(next_hidden.shape[0]))
        hidden.copy_(next_hidden.detach())
        clear = done.to(dtype=torch.bool)
        if reset_mask is not None:
            clear = clear | reset_mask.to(dtype=torch.bool)
        hidden.masked_fill_(clear.unsqueeze(-1), 0.0)
        pending.copy_(clear)

    def detach_carry_after_update(self) -> None:
        """Carry GRU state across rollouts; detach graph, do not zero all rows."""
        if self.carry_hidden is not None:
            self.carry_hidden = self.carry_hidden.detach()
        if self.carry_reset is not None:
            self.carry_reset = self.carry_reset.detach()

    def compute_advantages(
        self,
        batch: CompactRolloutBatch,
        values: Any,
        last_values: Any,
        next_values: Any | None = None,
    ) -> tuple[Tensor, Tensor]:
        values_t = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        last_t = torch.as_tensor(last_values, device=self.device, dtype=torch.float32)
        next_t = None
        if next_values is not None:
            next_t = torch.as_tensor(
                next_values, device=self.device, dtype=torch.float32
            )
        return compute_gae(
            batch.rewards.to(device=self.device, dtype=torch.float32),
            values_t,
            batch.done.to(device=self.device),
            batch.timeout.to(device=self.device),
            last_t,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            valid=batch.valid.to(device=self.device),
            next_values=next_t,
        )

    def _critic_values(
        self,
        ego_state: Tensor,
        other_agents: Tensor,
        other_mask: Tensor,
        condition: Tensor,
    ) -> Tensor:
        return critic_values_over_time(
            self.critic, ego_state, other_agents, other_mask, condition
        )

    def update(
        self,
        batch: CompactRolloutBatch,
        prepared: PreparedPPOInputs,
    ) -> PPOUpdateStats:
        t_steps, num_slots = batch.rewards.shape
        if t_steps != self.rollout_length:
            raise ValueError(
                f"rollout has {t_steps} steps; expected {self.rollout_length}"
            )
        device = self.device
        valid = batch.valid.to(device=device)
        rewards = batch.rewards.to(device=device, dtype=torch.float32)
        done = batch.done.to(device=device)
        timeout = batch.timeout.to(device=device)
        actions = batch.actions.to(device=device, dtype=torch.float32)
        # Do not sanitize old_logp/sensor_obs to 0: that silently creates huge
        # PPO ratios on kept rows. Fail the parity gate instead.
        old_logp = batch.old_logp.to(device=device, dtype=torch.float32)
        reset_mask = batch.reset_mask.to(device=device)
        gru_start = batch.gru_start.to(device=device, dtype=torch.float32)
        condition = batch.condition.to(device=device, dtype=torch.float32)
        pre_tanh = getattr(batch, "pre_tanh", None)
        if pre_tanh is not None:
            pre_tanh = pre_tanh.to(device=device, dtype=torch.float32)
        sensor_obs = prepared.sensor_obs.to(device=device, dtype=torch.float32)
        ego_state = torch.nan_to_num(
            prepared.ego_state.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        other_agents = torch.nan_to_num(
            prepared.other_agents.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        other_mask = prepared.other_mask.to(device=device)

        with torch.no_grad():
            if prepared.old_values is None:
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp,
                ):
                    old_values = self._critic_values(
                        ego_state, other_agents, other_mask, condition
                    )
                old_values = old_values.to(dtype=torch.float32)
            else:
                old_values = prepared.old_values.to(
                    device=device, dtype=torch.float32
                )
            bootstrap_values = None
            if prepared.bootstrap_values is not None:
                bootstrap_values = prepared.bootstrap_values.to(
                    device=device, dtype=torch.float32
                )
            if prepared.last_values is not None:
                last_values = prepared.last_values.to(
                    device=device, dtype=torch.float32
                )
            elif bootstrap_values is not None:
                # Past the rollout edge the only true next state is s'_{T-1}.
                last_values = bootstrap_values[-1].clone()
            else:
                last_values = old_values[-1].clone()
            advantages, returns = compute_gae(
                rewards,
                old_values,
                done,
                timeout,
                last_values,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                valid=valid,
                next_values=bootstrap_values,
            )
            keep, self.filter_state, eta = adaptive_advantage_keep_mask(
                advantages,
                valid,
                self.filter_state,
                enabled=self.filter_enabled,
            )
            retained = int(keep.sum().item())
            total_valid = int(valid.sum().item())
            retention = (
                float(retained / total_valid) if total_valid > 0 else 0.0
            )
            ent_coef = entropy_coefficient(self.cfg, self.update_index)
            lr = float(self.actor_optimizer.param_groups[0]["lr"])
            if retained == 0:
                # Do not step the LR schedule without an optimizer step.
                self.update_index += 1
                self.detach_carry_after_update()
                return PPOUpdateStats(
                    retention=0.0,
                    policy_loss=0.0,
                    value_loss=0.0,
                    entropy=0.0,
                    approx_kl=0.0,
                    clip_fraction=0.0,
                    grad_norm=0.0,
                    learning_rate=lr,
                    filter_eta=eta,
                    early_stopped=False,
                    epochs_completed=0,
                    entropy_coef=ent_coef,
                    actor_rollback_count=self.actor_rollback_count,
                    consecutive_actor_rollbacks=self.consecutive_actor_rollbacks,
                    actor_lr_safety_multiplier=self.actor_lr_safety_multiplier,
                )
            kept_adv = advantages[keep]
            raw_advantages = kept_adv.clone()
            adv_mean = kept_adv.mean()
            adv_std = kept_adv.std(unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            keep_w = keep.to(dtype=torch.float32)
            if not torch.isfinite(old_logp[keep]).all():
                raise CollectEvaluateParityError(
                    "non-finite old_logp on advantage-filtered transitions"
                )
            if not torch.isfinite(sensor_obs[keep]).all():
                raise CollectEvaluateParityError(
                    "non-finite sensor_obs on advantage-filtered transitions"
                )

        # Hard collect/evaluate parity gate on every slot (frozen weights).
        # Scoring a filtered subset instead would change the batch shape, and
        # cuDNN/cuBLAS pick different kernels per shape: the resulting float32
        # drift compounds through the recurrence and reaches ~5e-2 in log-prob,
        # which is noise, not disagreement. Rows the filter dropped carry zero
        # weight, so the full slot set reports the same statistics.
        #
        # Despite its name, this gate cannot see genuine collect-vs-evaluate
        # divergence: trainer.reconstruct_prepared already overwrote `old_logp`
        # (the collect-path value passed in via `batch`) with its own
        # evaluate-path rescore, using the same frozen weights and the same
        # whole-sequence batch shape as `new_logp_t` below. So both sides of
        # this comparison are evaluate-path outputs, and it only bounds
        # within-evaluate-path kernel nondeterminism, not the true collect vs.
        # evaluate gap (measured up to 6.1e-3 in log-prob at production
        # dimensions, and invisible here by construction). The bit-exact
        # observation digest check in reconstruct_prepared is the real
        # reconstruction guard. Restoring the collect-time old_logp here would
        # reintroduce that ~6e-3 systematic ratio bias with no gate to catch
        # it — do not do that without also changing this gate's inputs.
        pre_update_approx_kl = 0.0
        logp_delta_max = 0.0
        logp_delta_mean = 0.0
        action_pretanh_max_abs = 0.0
        with torch.no_grad():
            obs_d = sensor_obs.transpose(0, 1).contiguous()
            act_d = actions.transpose(0, 1).contiguous()
            reset_d = reset_mask.transpose(0, 1).contiguous()
            pre_d = None
            if pre_tanh is not None:
                pre_d = pre_tanh.transpose(0, 1).contiguous()
            if condition.ndim == 3:
                cond_d = condition.transpose(0, 1).contiguous()
            else:
                cond_d = condition
            # Actor rescoring stays FP32 to match collection logp exactly.
            new_logp_d, _, _ = evaluate_actions_sequence(
                self.actor,
                obs_d,
                cond_d,
                gru_start,
                act_d,
                reset_mask=reset_d,
                pre_tanh=pre_d,
            )
            new_logp_t = new_logp_d.transpose(0, 1).to(dtype=torch.float32)
            (
                pre_update_approx_kl,
                logp_delta_max,
                logp_delta_mean,
                action_pretanh_max_abs,
            ) = _assert_collect_evaluate_parity(
                new_logp_t=new_logp_t,
                old_logp=old_logp,
                keep_w=keep_w,
                actions=actions,
                pre_tanh=pre_tanh,
                sensor_obs=sensor_obs,
                condition=condition,
                reset_mask=reset_mask,
                gru_start=gru_start,
                max_pre_update_kl=self.max_pre_update_kl,
                max_logp_delta=self.max_logp_delta,
            )

        mb_agents = agents_per_minibatch(
            self.minibatch_size, self.rollout_length, num_slots
        )
        metric_sums = {
            "policy_loss": 0.0,
            "surrogate_loss": 0.0,
            "entropy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
        }
        actor_steps = 0
        critic_steps = 0
        epochs_completed = 0
        actor_stopped = False
        actor_rolled_back = False
        actor_rollback_full_rollout = False
        candidate_kl_max = 0.0
        actor_kl_rejection_reason = ""

        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        actor_snapshot = [p.detach().clone() for p in actor_params]
        actor_optimizer_snapshot = copy.deepcopy(self.actor_optimizer.state_dict())
        actor_scheduler_snapshot = copy.deepcopy(self.actor_scheduler.state_dict())
        safety_snapshot = float(self.actor_lr_safety_multiplier)
        actor_weight_norm = math.sqrt(
            sum(float(param.double().square().sum().item()) for param in actor_snapshot)
        )

        for epoch in range(self.num_epochs):
            perm = torch.randperm(num_slots, device=device)
            for start in range(0, num_slots, mb_agents):
                idx = perm[start : start + mb_agents]
                if idx.numel() == 0:
                    continue
                # Sequences: [B, T, ...]
                obs_seq = sensor_obs[:, idx].transpose(0, 1).contiguous()
                act_seq = actions[:, idx].transpose(0, 1).contiguous()
                reset_seq = reset_mask[:, idx].transpose(0, 1).contiguous()
                pre_seq = None
                if pre_tanh is not None:
                    pre_seq = pre_tanh[:, idx].transpose(0, 1).contiguous()
                hidden = gru_start[idx]
                if condition.ndim == 3:
                    cond = condition[:, idx].transpose(0, 1).contiguous()
                else:
                    cond = condition[idx]
                mb_keep = keep_w[:, idx]
                keep_sum = float(mb_keep.sum().item())
                if keep_sum <= 0.0:
                    continue
                keep_denom = mb_keep.sum().clamp(min=1.0)
                mb_adv = advantages[:, idx]
                mb_old_logp = old_logp[:, idx]
                mb_returns = returns[:, idx]
                ego_mb = ego_state[:, idx]
                others_mb = other_agents[:, idx]
                omask_mb = other_mask[:, idx]

                self.critic_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp,
                ):
                    new_values = self._critic_values(
                        ego_mb, others_mb, omask_mb, cond
                    )
                new_values = new_values.to(dtype=torch.float32)
                value_loss_per = 0.5 * (new_values - mb_returns).square()
                value_loss = (value_loss_per * mb_keep).sum() / keep_denom
                critic_loss = self.vf_coef * value_loss
                if torch.isfinite(critic_loss.detach()):
                    critic_loss.backward()
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    if torch.isfinite(critic_grad_norm):
                        self.critic_optimizer.step()
                        metric_sums["value_loss"] += float(value_loss.detach())
                        metric_sums["critic_grad_norm"] += float(critic_grad_norm)
                        critic_steps += 1

                if actor_stopped:
                    continue

                self.actor_optimizer.zero_grad(set_to_none=True)
                new_logp, new_entropy, _ = evaluate_actions_sequence(
                    self.actor,
                    obs_seq,
                    cond,
                    hidden,
                    act_seq,
                    reset_mask=reset_seq,
                    pre_tanh=pre_seq,
                )
                new_logp_t = new_logp.transpose(0, 1).to(dtype=torch.float32)
                new_ent_t = new_entropy.transpose(0, 1).to(dtype=torch.float32)
                log_ratio = new_logp_t - mb_old_logp
                ratio = log_ratio.exp()
                unclipped = ratio * mb_adv
                clipped = (
                    ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    * mb_adv
                )
                surrogate_per = -torch.minimum(unclipped, clipped)
                surrogate_loss = (surrogate_per * mb_keep).sum() / keep_denom
                entropy = (new_ent_t * mb_keep).sum() / keep_denom
                entropy_loss = -ent_coef * entropy
                policy_loss = surrogate_loss + entropy_loss
                if not torch.isfinite(policy_loss.detach()):
                    continue
                policy_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )
                if not torch.isfinite(actor_grad_norm):
                    continue
                self.actor_optimizer.step()
                clip_frac = (
                    ((ratio - 1.0).abs() > self.clip_ratio).to(dtype=torch.float32)
                    * mb_keep
                ).sum() / keep_denom
                metric_sums["policy_loss"] += float(policy_loss.detach())
                metric_sums["surrogate_loss"] += float(surrogate_loss.detach())
                metric_sums["entropy_loss"] += float(entropy_loss.detach())
                metric_sums["entropy"] += float(entropy.detach())
                metric_sums["clip_fraction"] += float(clip_frac.detach())
                metric_sums["actor_grad_norm"] += float(actor_grad_norm)
                actor_steps += 1

                with torch.no_grad():
                    rescored, _, _ = _rescore_actor(
                        self.actor,
                        obs_seq,
                        cond,
                        hidden,
                        act_seq,
                        reset_seq,
                        pre_seq,
                    )
                    candidate_kl = float(
                        _weighted_kl(
                            rescored.transpose(0, 1), mb_old_logp, mb_keep
                        ).item()
                    )
                candidate_kl_max = max(candidate_kl_max, candidate_kl)
                if (
                    self.actor_kl_hard is not None
                    and candidate_kl > self.actor_kl_hard
                ):
                    actor_rolled_back = True
                    actor_stopped = True
                    actor_kl_rejection_reason = "candidate_kl_hard"
                elif (
                    self.actor_kl_soft is not None
                    and candidate_kl > self.actor_kl_soft
                ):
                    actor_stopped = True
            epochs_completed = epoch + 1

        full_rollout_kl = 0.0
        full_entropy = torch.zeros_like(old_logp)
        full_log_std = None
        if not actor_rolled_back:
            with torch.no_grad():
                full_logp_d, full_entropy_d, full_log_std = _rescore_actor(
                    self.actor,
                    obs_d,
                    cond_d,
                    gru_start,
                    act_d,
                    reset_d,
                    pre_d,
                    diagnostics=True,
                )
                full_entropy = full_entropy_d.transpose(0, 1)
                full_rollout_kl = float(
                    _weighted_kl(
                        full_logp_d.transpose(0, 1), old_logp, keep_w
                    ).item()
                )
            if (
                self.actor_kl_hard is not None
                and full_rollout_kl > self.actor_kl_hard
            ):
                actor_rolled_back = True
                actor_rollback_full_rollout = True
                actor_kl_rejection_reason = "full_rollout_kl_hard"

        if actor_rolled_back:
            with torch.no_grad():
                for param, saved in zip(actor_params, actor_snapshot):
                    param.copy_(saved)
            self.actor_optimizer.load_state_dict(actor_optimizer_snapshot)
            self.actor_scheduler.load_state_dict(actor_scheduler_snapshot)
            self.actor_lr_safety_multiplier = safety_snapshot
            self.actor_lr_safety_multiplier = max(
                self.actor_lr_scale_min,
                self.actor_lr_safety_multiplier * self.actor_lr_backoff,
            )
            self._apply_actor_safety_lr()
            self.actor_rollback_count += 1
            self.consecutive_actor_rollbacks += 1
            self.rollback_updates.append(self.update_index + 1)
            with torch.no_grad():
                _, restored_entropy, full_log_std = _rescore_actor(
                    self.actor,
                    obs_d,
                    cond_d,
                    gru_start,
                    act_d,
                    reset_d,
                    pre_d,
                    diagnostics=True,
                )
                full_entropy = restored_entropy.transpose(0, 1)
            full_rollout_kl = 0.0
        else:
            self.consecutive_actor_rollbacks = 0
            self.rollback_free_accepted_updates += 1
            if (
                self.rollback_free_accepted_updates
                >= self.actor_lr_recovery_interval_updates
                and self.actor_lr_safety_multiplier < 1.0
            ):
                self.actor_lr_safety_multiplier = min(
                    1.0,
                    self.actor_lr_safety_multiplier
                    * self.actor_lr_recovery_multiplier,
                )
                self.rollback_free_accepted_updates = 0
                self._apply_actor_safety_lr()

        current_update = self.update_index + 1
        actor_kl_warmup_active = current_update <= self.actor_kl_warmup_updates
        cutoff = self.update_index - (self.actor_kl_stop_window_updates - 1)
        self.rollback_updates = [
            update for update in self.rollback_updates if update >= cutoff
        ]
        if not self.actor_kl_stop_enabled:
            rollback_stop_requested = False
        elif actor_kl_warmup_active:
            rollback_stop_requested = False
        else:
            rollback_stop_requested = (
                self.consecutive_actor_rollbacks >= 2
                or len(self.rollback_updates) >= self.actor_kl_stop_window_count
            )

        if critic_steps == 0:
            epochs_completed = 0
        if actor_steps > 0 and not actor_rolled_back:
            self._step_actor_scheduler()
        if critic_steps > 0:
            self.critic_scheduler.step()

        actor_delta_norm = math.sqrt(
            sum(
                float((param.detach() - saved).double().square().sum().item())
                for param, saved in zip(actor_params, actor_snapshot)
            )
        )
        actor_update_ratio = actor_delta_norm / max(actor_weight_norm, 1e-30)

        telemetry = {}
        normalized_kept = advantages[keep]
        residuals = returns[keep] - old_values[keep]
        telemetry.update(_percentile_metrics("advantage_raw", raw_advantages))
        telemetry.update(_percentile_metrics("advantage_normalized", normalized_kept))
        telemetry.update(_percentile_metrics("returns", returns[keep]))
        telemetry.update(_percentile_metrics("values", old_values[keep]))
        telemetry.update(_percentile_metrics("residual", residuals))
        returns_var = float(returns[keep].var(unbiased=False).item())
        residual_var = float(residuals.var(unbiased=False).item())
        telemetry["explained_variance"] = (
            1.0 - residual_var / returns_var if returns_var > 1e-12 else 0.0
        )
        for dim in range(actions.shape[-1]):
            telemetry[f"action_saturation_dim{dim}"] = float(
                (actions[..., dim][keep].abs() > 0.95).float().mean().item()
            )
        entropy_denom = keep_w.sum().clamp(min=1.0)
        telemetry["latent_entropy_total"] = float(
            (full_entropy * keep_w).sum().item() / entropy_denom.item()
        )
        if full_log_std is None and hasattr(self.actor, "log_std"):
            candidate = getattr(self.actor, "log_std")
            if isinstance(candidate, Tensor):
                full_log_std = candidate.view(1, 1, -1).expand(
                    num_slots, t_steps, -1
                )
        if full_log_std is not None:
            log_std_t = full_log_std.transpose(0, 1)
            kept_log_std = log_std_t[keep].detach()
            flat_log_std = kept_log_std.flatten()
            telemetry["log_std_p50"] = float(torch.quantile(flat_log_std, 0.5))
            telemetry["log_std_p95"] = float(torch.quantile(flat_log_std, 0.95))
            telemetry["log_std_max"] = float(flat_log_std.max())
            telemetry["log_std_cap_fraction"] = float(
                (flat_log_std >= LOG_STD_MAX - 1e-6).float().mean()
            )
            entropy_dims = (
                0.5 * math.log(2.0 * math.pi * math.e) + kept_log_std
            ).mean(dim=0)
            for dim, value in enumerate(entropy_dims):
                telemetry[f"latent_entropy_dim{dim}"] = float(value)

        self.update_index += 1
        self.detach_carry_after_update()
        # Keep live carry aligned with post-update detached weights.
        if self.carry_hidden is None and self._num_slots is not None:
            self.ensure_carry(self._num_slots)

        actor_denom = max(1, actor_steps)
        critic_denom = max(1, critic_steps)
        lr = float(self.actor_optimizer.param_groups[0]["lr"])
        return PPOUpdateStats(
            retention=retention,
            policy_loss=metric_sums["policy_loss"] / actor_denom,
            value_loss=metric_sums["value_loss"] / critic_denom,
            entropy=metric_sums["entropy"] / actor_denom,
            approx_kl=full_rollout_kl,
            clip_fraction=metric_sums["clip_fraction"] / actor_denom,
            grad_norm=max(
                metric_sums["actor_grad_norm"] / actor_denom,
                metric_sums["critic_grad_norm"] / critic_denom,
            ),
            learning_rate=lr,
            filter_eta=eta,
            early_stopped=actor_stopped,
            epochs_completed=epochs_completed,
            pre_update_approx_kl=pre_update_approx_kl,
            logp_delta_max=logp_delta_max,
            logp_delta_mean=logp_delta_mean,
            action_pretanh_max_abs=action_pretanh_max_abs,
            surrogate_loss=metric_sums["surrogate_loss"] / actor_denom,
            entropy_loss=metric_sums["entropy_loss"] / actor_denom,
            entropy_coef=ent_coef,
            candidate_kl=candidate_kl_max,
            full_rollout_kl=full_rollout_kl,
            actor_grad_norm=metric_sums["actor_grad_norm"] / actor_denom,
            critic_grad_norm=metric_sums["critic_grad_norm"] / critic_denom,
            actor_update_to_weight_norm=actor_update_ratio,
            actor_steps=actor_steps,
            critic_steps=critic_steps,
            actor_rolled_back=actor_rolled_back,
            actor_rollback_full_rollout=actor_rollback_full_rollout,
            actor_rollback_count=self.actor_rollback_count,
            consecutive_actor_rollbacks=self.consecutive_actor_rollbacks,
            actor_lr_safety_multiplier=self.actor_lr_safety_multiplier,
            rollback_stop_requested=rollback_stop_requested,
            actor_kl_rejection_reason=actor_kl_rejection_reason,
            actor_kl_warmup_active=actor_kl_warmup_active,
            rollback_free_accepted_updates=self.rollback_free_accepted_updates,
            telemetry=telemetry,
        )


def build_ppo(
    cfg: ExperimentConfig,
    actor: ConditionedActor,
    critic: CentralValueCritic,
    *,
    total_updates: int | None = None,
    target_kl: float | None = None,
    orthogonal_init: bool = False,
    device: str | None = None,
) -> RecurrentPPO:
    return RecurrentPPO(
        cfg,
        actor,
        critic,
        total_updates=total_updates,
        target_kl=target_kl,
        orthogonal_init=orthogonal_init,
        device=device,
    )

"""KL-regularized offline fine-tune of a baked Gigaflow actor on teleop windows.

PPO resume from actor-only weights is rejected: those artifacts have no critic
or optimizer state. Hybrid sim+bag PPO below trains a fresh critic.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gigaflow_f1tenth.artifacts import (
    FileArtifactStore,
    bake_actor_artifact,
    export_baked_actor,
)
from gigaflow_f1tenth.irl_identity import attach_baked_condition, baked_normalized_condition
from gigaflow_f1tenth.model import (
    ActorShapes,
    ConditionedLidarGRUActor,
    build_actor,
)
from gigaflow_f1tenth.teleop_dataset import (
    split_holdout,
    synthetic_trajectories,
    windows_from_trajectories,
)

SCALAR_NAME = "composite_keep_score"
PPO_METHODS = {"ppo", "ppo-resume", "ppo_resume"}
UNIQUE_ARMS = (
    "kl_clipped_bc",
    "kl_anchored_iql",
    "awac_kl_clip",
    "cal_ql_cql_kl",
    "trunk_frozen_head_kl",
    "residual_policy_kl",
    "hybrid_sim_ppo_bag_kl",
)
DEFAULT_KL_DELTA = 0.02
DEFAULT_KL_BETA = 1.0


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class IQLHeads(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int = 2) -> None:
        super().__init__()
        self.q = _mlp([hidden_dim + action_dim, 256, 256, 1])
        self.v = _mlp([hidden_dim, 256, 256, 1])

    def q_value(self, hidden: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([hidden, actions], dim=-1)).squeeze(-1)

    def v_value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden).squeeze(-1)


class ResidualHead(nn.Module):
    def __init__(self, hidden_dim: int, bound: float = 0.15) -> None:
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(self.net(hidden))


def actor_from_payload(payload: Mapping[str, Any], device: str | torch.device):
    arch = payload.get("actor_architecture")
    if not isinstance(arch, Mapping):
        raise KeyError("artifact missing actor_architecture")
    actor = ConditionedLidarGRUActor(
        ActorShapes(
            sensor_obs_dim=int(arch["sensor_obs_dim"]),
            lidar_dim=int(arch["lidar_dim"]),
            proprio_dim=int(arch["proprio_dim"]),
            condition_dim=int(arch["condition_dim"]),
            action_dim=int(arch["action_dim"]),
            gru_hidden_dim=int(arch["gru_hidden_dim"]),
            cnn_projection_dim=int(arch["cnn_projection_dim"]),
            cnn_pool_bins=int(arch.get("pool_bins", 64)),
            cnn_pool_mode=str(arch.get("pool_mode", "max")),
            mlp_sizes=tuple(int(v) for v in arch["mlp_sizes"]),
        ),
        condition_embed_dim=int(arch.get("condition_embed_dim", 64)),
    )
    state = payload["actor_state_dict"]
    actor.load_state_dict(state, strict=False)
    vec = payload.get("deployment_condition_normalized")
    if not isinstance(vec, (list, tuple)):
        vec = baked_normalized_condition()
    attach_baked_condition(actor, vec)
    freeze_normalizer(actor)
    return actor.to(device)


def freeze_normalizer(actor: nn.Module) -> None:
    normalizer = getattr(actor, "sensor_normalizer", None)
    if normalizer is None:
        return
    normalizer.eval()
    for buf in normalizer.buffers():
        buf.requires_grad_(False)

    def _frozen_update(*_args, **_kwargs):
        return None

    normalizer.update = _frozen_update  # type: ignore[method-assign]


def load_offline_actor(checkpoint: str | Path, device: str = "cpu"):
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if "actor_state_dict" in payload:
        baked = bake_actor_artifact(payload)
        actor = actor_from_payload(baked, device)
        return actor, baked
    if "ppo" in payload and isinstance(payload["ppo"], Mapping):
        from gigaflow_f1tenth.config import config_from_dict

        cfg = config_from_dict(payload["config"], check_training_budget=False)
        actor = build_actor(cfg)
        actor.load_state_dict(payload["ppo"]["actor"], strict=False)
        baked = export_baked_actor(cfg, actor)
        actor = actor_from_payload(baked, device)
        return actor, baked
    raise KeyError("checkpoint missing actor weights")


def _condition_tensor(payload: Mapping[str, Any], batch: int, device) -> torch.Tensor:
    vec = payload.get("deployment_condition_normalized")
    if not isinstance(vec, (list, tuple)):
        vec = baked_normalized_condition()
    t = torch.as_tensor(vec, dtype=torch.float32, device=device)
    return t.unsqueeze(0).expand(int(batch), -1).contiguous()


def _batch_to_torch(windows: Mapping[str, np.ndarray], device, indices: np.ndarray):
    idx = np.asarray(indices)
    obs = torch.as_tensor(windows["obs"][idx], dtype=torch.float32, device=device)
    actions = torch.as_tensor(
        windows["actions"][idx], dtype=torch.float32, device=device
    )
    rewards = torch.as_tensor(
        windows["rewards"][idx], dtype=torch.float32, device=device
    )
    dones = torch.as_tensor(windows["dones"][idx], dtype=torch.float32, device=device)
    resets = torch.as_tensor(
        windows["resets"][idx], dtype=torch.float32, device=device
    )
    return obs, actions, rewards, dones, resets


def gaussian_kl(mu_p, std_p, mu_q, std_q) -> torch.Tensor:
    var_p = std_p.square()
    var_q = std_q.square().clamp_min(1e-8)
    return 0.5 * (
        (var_p + (mu_p - mu_q).square()) / var_q
        + 2.0 * (std_q.clamp_min(1e-8).log() - std_p.clamp_min(1e-8).log())
        - 1.0
    ).sum(dim=-1)


def hinged_kl(mu_p, std_p, mu_q, std_q, delta: float) -> torch.Tensor:
    kl = gaussian_kl(mu_p, std_p, mu_q, std_q)
    return torch.relu(kl - float(delta))


def freeze_modules(*modules: nn.Module) -> None:
    for mod in modules:
        for param in mod.parameters():
            param.requires_grad_(False)


@torch.no_grad()
def action_mse(
    actor: ConditionedLidarGRUActor,
    windows: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    device: str,
) -> float:
    actor.eval()
    n = int(windows["obs"].shape[0])
    if n == 0:
        return float("inf")
    idx = np.arange(n)
    obs, actions, _, _, resets = _batch_to_torch(windows, device, idx)
    cond = _condition_tensor(payload, obs.shape[0], device)
    hidden = actor.initial_hidden(obs.shape[0], device)
    out = actor.evaluate_actions_sequence(
        obs, cond, hidden, actions, reset_mask=resets, return_mean=True
    )
    mean = out[-1]
    return float(((mean - actions) ** 2).mean().item())


@torch.no_grad()
def holdout_stats(
    actor: ConditionedLidarGRUActor,
    windows: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    device: str,
) -> dict[str, float]:
    actor.eval()
    n = int(windows["obs"].shape[0])
    if n == 0:
        return {"mse": float("inf"), "logp": float("-inf"), "steer_jerk": 1.0}
    obs, actions, _, _, resets = _batch_to_torch(windows, device, np.arange(n))
    cond = _condition_tensor(payload, obs.shape[0], device)
    hidden = actor.initial_hidden(obs.shape[0], device)
    logp, _, _, mean = actor.evaluate_actions_sequence(
        obs, cond, hidden, actions, reset_mask=resets, return_mean=True
    )
    mse = float(((mean - actions) ** 2).mean().item())
    jerk = float((mean[:, 1:, 1] - mean[:, :-1, 1]).abs().mean().item())
    return {"mse": mse, "logp": float(logp.mean().item()), "steer_jerk": jerk}


def composite_score(
    stats: Mapping[str, float],
    closed_loop: Mapping[str, float] | None = None,
) -> float:
    score = (
        -2.0 * float(stats["mse"])
        - 1.5 * float(stats["steer_jerk"])
        + 0.05 * float(stats["logp"])
    )
    if closed_loop:
        score += 0.02 * float(closed_loop.get("progress_m", 0.0))
        score += 4.0 * float(closed_loop.get("wall_clearance_m", 0.0))
        score -= 6.0 * float(closed_loop.get("crash", 0.0))
        score -= 2.0 * float(closed_loop.get("steer_reversal", 0.0))
    return float(score)


def _expectile(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, tau, 1.0 - tau)
    return (weight * diff.square()).mean()


def _kl_to_base(actor, base, obs, cond, hidden, resets, delta: float) -> torch.Tensor:
    mu, std, _, _ = actor.distribution_params_sequence(obs, cond, hidden, resets)
    with torch.no_grad():
        mu_b, std_b, _, _ = base.distribution_params_sequence(obs, cond, hidden, resets)
    return hinged_kl(mu, std, mu_b, std_b, delta).mean()


def _trainable_actor_params(actor: nn.Module):
    return [p for p in actor.parameters() if p.requires_grad]


def finetune_kl_clipped_bc(
    actor, base, windows, payload, *, steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, kl_beta=DEFAULT_KL_BETA, smooth_w=0.0,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    opt = torch.optim.Adam(_trainable_actor_params(actor), lr=float(lr))
    actor.train()
    base.eval()
    n = int(windows["obs"].shape[0])
    last = 0.0
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, _, _, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden = actor.initial_hidden(obs.shape[0], device)
        logp, _, _ = actor.evaluate_actions_sequence(
            obs, cond, hidden, actions, reset_mask=resets
        )
        kl = _kl_to_base(actor, base, obs, cond, hidden, resets, kl_delta)
        mu, _, _, _ = actor.distribution_params_sequence(obs, cond, hidden, resets)
        smooth = (mu[:, 1:, 1] - mu[:, :-1, 1]).abs().mean()
        loss = -logp.mean() + float(kl_beta) * kl + float(smooth_w) * smooth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(_trainable_actor_params(actor), 1.0)
        opt.step()
        last = float(loss.detach().item())
    return {"train_loss": last}


def finetune_kl_anchored_iql(
    actor, base, windows, payload, *, steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, kl_beta=DEFAULT_KL_BETA, tau=0.7, gamma=0.99,
    adv_beta=1.0, **_kwargs,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    hidden_dim = int(actor.shapes().gru_hidden_dim)
    heads = IQLHeads(hidden_dim).to(device)
    pi_opt = torch.optim.Adam(_trainable_actor_params(actor), lr=float(lr))
    q_opt = torch.optim.Adam(heads.q.parameters(), lr=float(lr))
    v_opt = torch.optim.Adam(heads.v.parameters(), lr=float(lr))
    actor.train()
    heads.train()
    base.eval()
    n = int(windows["obs"].shape[0])
    last = {"q_loss": 0.0, "v_loss": 0.0, "pi_loss": 0.0}
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, rewards, dones, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden0 = actor.initial_hidden(obs.shape[0], device)
        batch, length, _ = obs.shape
        mu, _, gru_out, _ = actor.distribution_params_sequence(
            obs, cond, hidden0, resets
        )
        h = gru_out.detach()
        q_sa = heads.q_value(h.reshape(-1, hidden_dim), actions.reshape(-1, 2)).view(
            batch, length
        )
        v_s = heads.v_value(h.reshape(-1, hidden_dim)).view(batch, length)
        q_loss = F.mse_loss(
            q_sa[:, :-1],
            rewards[:, :-1] + float(gamma) * (1.0 - dones[:, :-1]) * v_s[:, 1:].detach(),
        )
        v_loss = _expectile(q_sa.detach() - v_s, float(tau))
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()
        v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        v_opt.step()
        logp, _, _ = actor.evaluate_actions_sequence(
            obs, cond, hidden0, actions, reset_mask=resets
        )
        adv = (q_sa[:, :-1].detach() - v_s[:, :-1].detach()) / max(float(adv_beta), 1e-6)
        weight = torch.exp(adv.clamp(max=10.0))
        kl = _kl_to_base(actor, base, obs, cond, hidden0, resets, kl_delta)
        pi_loss = -(weight * logp[:, :-1]).mean() + float(kl_beta) * kl
        pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(_trainable_actor_params(actor), 1.0)
        pi_opt.step()
        last = {
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "pi_loss": float(pi_loss.detach().item()),
        }
        _ = mu
    return last


def finetune_awac_kl_clip(
    actor, base, windows, payload, *, steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, kl_beta=DEFAULT_KL_BETA, gamma=0.99, lam=0.95,
    clip_eps=0.2, **_kwargs,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    hidden_dim = int(actor.shapes().gru_hidden_dim)
    heads = IQLHeads(hidden_dim).to(device)
    pi_opt = torch.optim.Adam(_trainable_actor_params(actor), lr=float(lr))
    q_opt = torch.optim.Adam(
        list(heads.q.parameters()) + list(heads.v.parameters()), lr=float(lr)
    )
    actor.train()
    n = int(windows["obs"].shape[0])
    last = 0.0
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, rewards, dones, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden0 = actor.initial_hidden(obs.shape[0], device)
        batch, length, _ = obs.shape
        _, _, gru_out, _ = actor.distribution_params_sequence(
            obs, cond, hidden0, resets
        )
        h = gru_out.detach()
        q_sa = heads.q_value(h.reshape(-1, hidden_dim), actions.reshape(-1, 2)).view(
            batch, length
        )
        v_s = heads.v_value(h.reshape(-1, hidden_dim)).view(batch, length)
        target = rewards[:, :-1] + float(gamma) * (1.0 - dones[:, :-1]) * (
            (1.0 - float(lam)) * v_s[:, 1:].detach() + float(lam) * q_sa[:, 1:].detach()
        )
        q_loss = F.mse_loss(q_sa[:, :-1], target) + F.mse_loss(
            v_s[:, :-1], q_sa[:, :-1].detach()
        )
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()
        logp, _, _ = actor.evaluate_actions_sequence(
            obs, cond, hidden0, actions, reset_mask=resets
        )
        with torch.no_grad():
            base_logp, _, _ = base.evaluate_actions_sequence(
                obs, cond, hidden0, actions, reset_mask=resets
            )
        ratio = torch.exp(logp - base_logp).clamp(1.0 - float(clip_eps), 1.0 + float(clip_eps))
        adv = (q_sa[:, :-1].detach() - v_s[:, :-1].detach())
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
        kl = _kl_to_base(actor, base, obs, cond, hidden0, resets, kl_delta)
        loss = -(ratio[:, :-1] * adv).mean() + float(kl_beta) * kl
        pi_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(_trainable_actor_params(actor), 1.0)
        pi_opt.step()
        last = float(loss.detach().item())
    return {"train_loss": last}


def finetune_cal_ql_cql_kl(
    actor, base, windows, payload, *, steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, kl_beta=DEFAULT_KL_BETA, gamma=0.99, cql_alpha=1.0,
    **_kwargs,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    hidden_dim = int(actor.shapes().gru_hidden_dim)
    heads = IQLHeads(hidden_dim).to(device)
    pi_opt = torch.optim.Adam(_trainable_actor_params(actor), lr=float(lr))
    q_opt = torch.optim.Adam(
        list(heads.q.parameters()) + list(heads.v.parameters()), lr=float(lr)
    )
    actor.train()
    n = int(windows["obs"].shape[0])
    last = 0.0
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, rewards, dones, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden0 = actor.initial_hidden(obs.shape[0], device)
        batch, length, _ = obs.shape
        mu, _, gru_out, _ = actor.distribution_params_sequence(
            obs, cond, hidden0, resets
        )
        h = gru_out.detach()
        flat_h = h.reshape(-1, hidden_dim)
        q_sa = heads.q_value(flat_h, actions.reshape(-1, 2)).view(batch, length)
        v_s = heads.v_value(flat_h).view(batch, length)
        with torch.no_grad():
            mean_act = torch.tanh(mu).clamp(-0.999, 0.999)
        q_pi = heads.q_value(flat_h, mean_act.reshape(-1, 2)).view(batch, length)
        v_ref = torch.maximum(q_sa.detach(), v_s.detach())
        td = rewards[:, :-1] + float(gamma) * (1.0 - dones[:, :-1]) * v_ref[:, 1:]
        q_loss = F.mse_loss(q_sa[:, :-1], td) + float(cql_alpha) * (
            q_pi[:, :-1] - q_sa[:, :-1]
        ).mean()
        v_loss = F.mse_loss(v_s[:, :-1], q_sa[:, :-1].detach())
        q_opt.zero_grad(set_to_none=True)
        (q_loss + v_loss).backward()
        q_opt.step()
        kl = _kl_to_base(actor, base, obs, cond, hidden0, resets, kl_delta)
        q_actor = heads.q_value(
            gru_out.reshape(-1, hidden_dim).detach(),
            torch.tanh(mu).clamp(-0.999, 0.999).reshape(-1, 2),
        )
        loss = -q_actor.mean() + float(kl_beta) * kl
        pi_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(_trainable_actor_params(actor), 1.0)
        pi_opt.step()
        last = float(loss.detach().item())
    return {"train_loss": last}


def finetune_trunk_frozen_head_kl(
    actor, base, windows, payload, **kwargs
) -> dict[str, float]:
    freeze_modules(
        actor.encoder, actor.condition_encoder, actor.gru, actor.sensor_normalizer
    )
    kwargs = dict(kwargs)
    kwargs["smooth_w"] = float(kwargs.pop("smooth_w", 0.05))
    return finetune_kl_clipped_bc(actor, base, windows, payload, **kwargs)


def finetune_residual_policy_kl(
    actor, base, windows, payload, *, steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, kl_beta=DEFAULT_KL_BETA, **_kwargs,
) -> dict[str, float]:
    freeze_modules(actor)
    head = ResidualHead(int(actor.shapes().gru_hidden_dim)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=float(lr))
    n = int(windows["obs"].shape[0])
    rng = np.random.default_rng(int(seed))
    last = 0.0
    actor.eval()
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, _, _, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden = actor.initial_hidden(obs.shape[0], device)
        with torch.no_grad():
            mu, std, gru_out, _ = actor.distribution_params_sequence(
                obs, cond, hidden, resets
            )
            base_act = torch.tanh(mu)
        residual = head(gru_out)
        pred = (base_act + residual).clamp(-1.0, 1.0)
        nll = ((pred - actions) ** 2).mean()
        mu_res = mu + residual
        kl = hinged_kl(mu_res, std, mu, std, kl_delta).mean()
        loss = nll + float(kl_beta) * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
    actor._residual_head = head
    for param in actor.parameters():
        param.requires_grad_(True)
    freeze_normalizer(actor)
    _bake_residual_into_mean(actor, head, windows, payload, device)
    return {"train_loss": last}


def _bake_residual_into_mean(actor, head: ResidualHead, windows, payload, device) -> None:
    """Fold a small residual into mu_layer bias using one holdout batch mean."""
    actor.eval()
    n = min(8, int(windows["obs"].shape[0]))
    obs, actions, _, _, resets = _batch_to_torch(windows, device, np.arange(n))
    cond = _condition_tensor(payload, obs.shape[0], device)
    hidden = actor.initial_hidden(obs.shape[0], device)
    with torch.no_grad():
        _, _, gru_out, _ = actor.distribution_params_sequence(obs, cond, hidden, resets)
        residual = head(gru_out).mean(dim=(0, 1))
        actor.mu_layer.bias.add_(residual.clamp(-0.15, 0.15))
    _ = actions


def finetune_hybrid_sim_ppo_bag_kl(
    actor, base, windows, payload, *, steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, kl_beta=DEFAULT_KL_BETA, clip_eps=0.2, **_kwargs,
) -> dict[str, float]:
    last_bc = finetune_kl_clipped_bc(
        actor, base, windows, payload,
        steps=max(1, int(steps) // 2),
        batch_size=batch_size, lr=lr, seed=seed, device=device,
        kl_delta=kl_delta, kl_beta=kl_beta,
    )
    sim_loss = _hybrid_sim_steps(
        actor, base, payload, steps=max(1, int(steps) - int(steps) // 2),
        batch_size=batch_size, lr=lr, seed=seed, device=device,
        kl_delta=kl_delta, kl_beta=kl_beta, clip_eps=clip_eps,
    )
    last_bc.update(sim_loss)
    return last_bc


def _hybrid_sim_steps(
    actor, base, payload, *, steps, batch_size, lr, seed, device,
    kl_delta, kl_beta, clip_eps,
) -> dict[str, float]:
    try:
        from gigaflow_f1tenth.config import load_config
        from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
        from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
    except Exception as exc:  # noqa: BLE001
        return {"sim_loss": 0.0, "sim_skip": str(exc)}
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
    cfg = load_config(cfg_path)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    wanted = str(device)
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        wanted = "cpu"
    try:
        sim = build_simulator(cfg, atlas, wanted)
    except Exception as exc:  # noqa: BLE001
        return {"sim_loss": 0.0, "sim_skip": str(exc)}
    n = world_slot_layout(cfg).num_slots
    opt = torch.optim.Adam(_trainable_actor_params(actor), lr=float(lr))
    last = 0.0
    horizon = 8
    for step in range(int(steps)):
        hidden = actor.initial_hidden(n, wanted)
        cond = actor.resolve_condition(None, n, wanted)
        pending = torch.zeros(n, dtype=torch.bool, device=wanted)
        logps = []
        rewards = []
        kls = []
        for _t in range(horizon):
            obs = sim.rebuild_sensors().to(device=wanted)
            out = actor.forward(obs, cond, hidden, pending, deterministic=False)
            with torch.no_grad():
                base_out = base.forward(obs, cond, hidden, pending, deterministic=False)
            mu, std, _, _ = actor.distribution_params_sequence(
                obs.unsqueeze(1), cond, hidden, pending.unsqueeze(1).float()
            )
            with torch.no_grad():
                mu_b, std_b, _, _ = base.distribution_params_sequence(
                    obs.unsqueeze(1), cond, hidden, pending.unsqueeze(1).float()
                )
            kls.append(hinged_kl(mu, std, mu_b, std_b, kl_delta).mean())
            stepped = sim.step(out.actions.detach())
            logps.append(out.log_prob)
            rewards.append(stepped["rewards"].to(device=wanted))
            hidden = out.hidden
            pending = stepped["reset_mask"].to(device=wanted, dtype=torch.bool)
            _ = base_out
        reward = torch.stack(rewards, dim=1)
        logp = torch.stack(logps, dim=1)
        adv = reward - reward.mean()
        ratio = torch.exp(logp - logp.detach()).clamp(1.0 - float(clip_eps), 1.0 + float(clip_eps))
        loss = -(ratio * adv).mean() + float(kl_beta) * torch.stack(kls).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(_trainable_actor_params(actor), 1.0)
        opt.step()
        last = float(loss.detach().item())
        _ = step
        _ = batch_size
        _ = seed
        _ = payload
    return {"sim_loss": last}


ARM_TRAINERS: dict[str, Callable[..., dict[str, float]]] = {
    "kl_clipped_bc": finetune_kl_clipped_bc,
    "kl_anchored_iql": finetune_kl_anchored_iql,
    "awac_kl_clip": finetune_awac_kl_clip,
    "cal_ql_cql_kl": finetune_cal_ql_cql_kl,
    "trunk_frozen_head_kl": finetune_trunk_frozen_head_kl,
    "residual_policy_kl": finetune_residual_policy_kl,
    "hybrid_sim_ppo_bag_kl": finetune_hybrid_sim_ppo_bag_kl,
}


def refuse_ppo(method: str) -> None:
    if str(method).strip().lower() in PPO_METHODS:
        raise ValueError("never PPO-resume from actor-only weights; use bc or iql")


def _score_actor(actor, hold_w, payload, device, closed_loop=None) -> dict[str, Any]:
    stats = holdout_stats(actor, hold_w, payload, device)
    score = composite_score(stats, closed_loop)
    return {"score": score, "stats": stats, "closed_loop": closed_loop or {}}


def run_one_arm(
    name: str,
    actor0,
    payload,
    train_w,
    hold_w,
    *,
    steps, batch_size, lr, seed, device,
    kl_delta=DEFAULT_KL_DELTA, window=16, smooth_w=0.0,
    output: Path | None = None,
) -> dict[str, Any]:
    actor = copy.deepcopy(actor0)
    base = copy.deepcopy(actor0)
    freeze_modules(base)
    freeze_normalizer(actor)
    trainer = ARM_TRAINERS[name]
    train = trainer(
        actor, base, train_w, payload,
        steps=steps, batch_size=batch_size, lr=lr, seed=seed, device=device,
        kl_delta=kl_delta, smooth_w=smooth_w,
    )
    scored = _score_actor(actor, hold_w, payload, device)
    artifact = None
    if output is not None:
        artifact = str(output / f"{name}_actor.pt")
        FileArtifactStore().write_actor_artifact(
            artifact, bake_actor_artifact(payload, actor=actor)
        )
    return {
        "name": name,
        "score": scored["score"],
        "stats": scored["stats"],
        "train": train,
        "artifact": artifact,
        "kl_delta": float(kl_delta),
        "window": int(window),
        "smooth_w": float(smooth_w),
        "actor": actor,
    }


def run_bakeoff(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    trajectories: Sequence[Mapping[str, np.ndarray]] | None = None,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 3.0e-5,
    seed: int = 0,
    device: str = "cpu",
    window: int = 16,
    method: str = "bakeoff",
    wandb_session: Any | None = None,
    phase2: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actor0, payload = load_offline_actor(checkpoint, device=device)
    trajs = list(trajectories) if trajectories is not None else synthetic_trajectories(
        seed=seed
    )
    train_trajs, hold_trajs = split_holdout(trajs, seed=seed)
    train_w = windows_from_trajectories(
        train_trajs, window=window, stride=max(1, window // 2)
    )
    hold_w = (
        windows_from_trajectories(hold_trajs, window=min(window, 8), stride=4)
        if hold_trajs
        else train_w
    )
    baseline = copy.deepcopy(actor0)
    base_scored = _score_actor(baseline, hold_w, payload, device)
    arms: dict[str, dict[str, Any]] = {
        "baseline": {
            "score": base_scored["score"],
            "stats": base_scored["stats"],
            "train": {},
            "artifact": str(output / "baseline_actor.pt"),
            "kl_delta": 0.0,
            "window": int(window),
        }
    }
    FileArtifactStore().write_actor_artifact(
        arms["baseline"]["artifact"], bake_actor_artifact(payload, actor=baseline)
    )

    names = UNIQUE_ARMS
    wanted = str(method).strip().lower()
    if wanted not in {"bakeoff", "all", ""} and wanted in ARM_TRAINERS:
        names = (wanted,)
    trained: dict[str, Any] = {}
    for i, name in enumerate(names):
        row = run_one_arm(
            name, actor0, payload, train_w, hold_w,
            steps=steps, batch_size=batch_size, lr=lr, seed=seed + i,
            device=device, window=window, output=output,
        )
        trained[name] = row.pop("actor")
        arms[name] = row

    if phase2:
        ranked = sorted(
            (k for k in names if k in arms),
            key=lambda n: float(arms[n]["score"]),
            reverse=True,
        )[:2]
        sweeps = (
            {"kl_delta": 0.005, "window": window, "smooth_w": 0.0},
            {"kl_delta": 0.02, "window": 32, "smooth_w": 0.02},
            {"kl_delta": 0.05, "window": 64, "smooth_w": 0.05},
        )
        for parent in ranked:
            for j, sweep in enumerate(sweeps):
                tag = f"{parent}_d{sweep['kl_delta']}_w{sweep['window']}"
                tw = windows_from_trajectories(
                    train_trajs,
                    window=int(sweep["window"]),
                    stride=max(1, int(sweep["window"]) // 2),
                )
                row = run_one_arm(
                    parent, actor0, payload, tw, hold_w,
                    steps=steps, batch_size=batch_size, lr=lr,
                    seed=seed + 20 + j, device=device,
                    kl_delta=float(sweep["kl_delta"]),
                    window=int(sweep["window"]),
                    smooth_w=float(sweep["smooth_w"]),
                    output=output,
                )
                row["name"] = tag
                trained[tag] = row.pop("actor")
                arms[tag] = row

    winner = max(arms, key=lambda name: float(arms[name]["score"]))
    keep = winner != "baseline"
    winner_path = output / "winner_actor.pt"
    winner_actor = baseline if winner == "baseline" else trained[winner]
    FileArtifactStore().write_actor_artifact(
        str(winner_path), bake_actor_artifact(payload, actor=winner_actor)
    )
    result = {
        "scalar": SCALAR_NAME,
        "lower_is_better": False,
        "winner": winner,
        "keep": keep,
        "decision": "keep" if keep else "revert",
        "winner_artifact": str(winner_path),
        "irl_identity": dict(payload.get("irl_identity") or {}),
        "arms": {
            name: {
                "score": float(row["score"]),
                "stats": row.get("stats", {}),
                "train": row.get("train", {}),
                "artifact": row.get("artifact"),
                "kl_delta": row.get("kl_delta"),
                "window": row.get("window"),
            }
            for name, row in arms.items()
        },
        "n_train_windows": int(train_w["obs"].shape[0]),
        "n_hold_windows": int(hold_w["obs"].shape[0]),
        "n_trajectories": int(len(trajs)),
        "device": str(device),
        "steps": int(steps),
        "checkpoint": str(checkpoint),
    }
    (output / "bakeoff.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if wandb_session is not None:
        wandb_session.start()
        metrics = {
            f"bakeoff/{SCALAR_NAME}_{name}": float(row["score"])
            for name, row in arms.items()
        }
        metrics["bakeoff/keep"] = float(keep)
        metrics["bakeoff/n_train_windows"] = float(result["n_train_windows"])
        metrics["bakeoff/n_hold_windows"] = float(result["n_hold_windows"])
        wandb_session.log_metrics(metrics, step=0)
        wandb_session.finish()
        result["wandb_run_id"] = wandb_session.run_id
    return result

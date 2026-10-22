"""Off-policy fine-tune of a baked Gigaflow actor on teleop windows.

One trainer, arm hyperparameters as data. PPO resume is rejected: actor-only
weights have no critic / optimizer state. Every fine-tune arm applies a KL
clip to a frozen copy of the base policy.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gigaflow_f1tenth.artifacts import (
    FileArtifactStore,
    bake_actor_artifact,
    export_baked_actor,
)
from gigaflow_f1tenth.irl_identity import baked_normalized_condition
from gigaflow_f1tenth.model import (
    ActorShapes,
    ConditionedLidarGRUActor,
    build_actor,
)
from gigaflow_f1tenth.sim.layout_local import VESC_SPEED
from gigaflow_f1tenth.teleop_dataset import (
    split_holdout,
    synthetic_trajectories,
    windows_from_trajectories,
)

SCALAR_NAME = "holdout_action_mse"
PPO_METHODS = {"ppo", "ppo-resume", "ppo_resume"}
DEFAULT_KL_COEF = 0.1
DEFAULT_KL_MAX = 0.05

# Seven unique distillations. Variations of the winner are built at runtime.
UNIQUE_ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "bc-kl",
        "method": "bc",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
    },
    {
        "name": "iql-kl",
        "method": "iql",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
    },
    {
        "name": "awac-kl",
        "method": "awac",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
        "beta": 1.0,
    },
    {
        "name": "crr-kl",
        "method": "crr",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
    },
    {
        "name": "bc-steer-kl",
        "method": "bc",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
        "action_weights": (0.4, 1.6),
    },
    {
        "name": "bc-force-kl",
        "method": "bc",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
        "action_weights": (1.6, 0.4),
    },
    {
        "name": "bc-fast-kl",
        "method": "bc",
        "kl_coef": DEFAULT_KL_COEF,
        "kl_max": DEFAULT_KL_MAX,
        "speed_quantile": 0.5,
    },
)


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
            mlp_sizes=tuple(int(v) for v in arch["mlp_sizes"]),
        ),
        condition_embed_dim=int(arch.get("condition_embed_dim", 64)),
    )
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return actor.to(device)


def load_offline_actor(checkpoint: str | Path, device: str = "cpu"):
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if "actor_state_dict" in payload:
        baked = bake_actor_artifact(payload)
        return actor_from_payload(baked, device), baked
    if "ppo" in payload and isinstance(payload["ppo"], Mapping):
        from gigaflow_f1tenth.config import config_from_dict

        cfg = config_from_dict(payload["config"], check_training_budget=False)
        actor = build_actor(cfg)
        actor.load_state_dict(payload["ppo"]["actor"], strict=True)
        baked = export_baked_actor(cfg, actor)
        return actor.to(device), baked
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


def gaussian_kl(
    mean_p: torch.Tensor,
    log_std_p: torch.Tensor,
    mean_q: torch.Tensor,
    log_std_q: torch.Tensor,
) -> torch.Tensor:
    var_p = torch.exp(log_std_p * 2.0)
    var_q = torch.exp(log_std_q * 2.0).clamp(min=1e-6)
    term = (var_p + (mean_p - mean_q).square()) / var_q
    return 0.5 * (log_std_q - log_std_p + term - 1.0).sum(dim=-1).mean()


def _dist_stats(actor, obs, cond, hidden, actions, resets):
    out = actor.evaluate_actions_sequence(
        obs,
        cond,
        hidden,
        actions,
        reset_mask=resets,
        return_diagnostics=True,
        return_mean=True,
    )
    logp, _ent, _h, log_std, mean = out
    return logp, mean, log_std


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
    _logp, mean, _std = _dist_stats(actor, obs, cond, hidden, actions, resets)
    return float(((mean - actions) ** 2).mean().item())


@torch.no_grad()
def holdout_kl(
    actor: ConditionedLidarGRUActor,
    base: ConditionedLidarGRUActor,
    windows: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    device: str,
) -> float:
    actor.eval()
    base.eval()
    n = int(windows["obs"].shape[0])
    if n == 0:
        return float("inf")
    obs, actions, _, _, resets = _batch_to_torch(windows, device, np.arange(n))
    cond = _condition_tensor(payload, obs.shape[0], device)
    hidden = actor.initial_hidden(obs.shape[0], device)
    _lp, mean_p, log_std_p = _dist_stats(actor, obs, cond, hidden, actions, resets)
    _lq, mean_q, log_std_q = _dist_stats(base, obs, cond, hidden, actions, resets)
    return float(gaussian_kl(mean_p, log_std_p, mean_q, log_std_q).item())


def _expectile(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, tau, 1.0 - tau)
    return (weight * diff.square()).mean()


def _filter_speed(
    windows: Mapping[str, np.ndarray], quantile: float
) -> Mapping[str, np.ndarray]:
    vx = np.abs(windows["obs"][..., VESC_SPEED]).mean(axis=1)
    thresh = float(np.quantile(vx, float(quantile)))
    keep = vx >= thresh
    if int(keep.sum()) == 0:
        return windows
    return {key: value[keep] for key, value in windows.items()}


def finetune(
    actor: ConditionedLidarGRUActor,
    base: ConditionedLidarGRUActor,
    windows: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    *,
    method: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    kl_coef: float = DEFAULT_KL_COEF,
    kl_max: float = DEFAULT_KL_MAX,
    action_weights: Sequence[float] = (1.0, 1.0),
    tau: float = 0.7,
    gamma: float = 0.99,
    beta: float = 1.0,
    speed_quantile: float | None = None,
) -> dict[str, float]:
    refuse_ppo(method)
    train_w = windows
    if speed_quantile is not None:
        train_w = _filter_speed(windows, float(speed_quantile))
    rng = np.random.default_rng(int(seed))
    opt = torch.optim.Adam(actor.parameters(), lr=float(lr))
    heads = None
    q_opt = v_opt = None
    needs_q = str(method) in {"iql", "awac", "crr"}
    if needs_q:
        heads = IQLHeads(int(actor.shapes().gru_hidden_dim)).to(device)
        q_opt = torch.optim.Adam(heads.q.parameters(), lr=float(lr))
        v_opt = torch.optim.Adam(heads.v.parameters(), lr=float(lr))
        heads.train()
    actor.train()
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)
    n = int(train_w["obs"].shape[0])
    weights = torch.as_tensor(list(action_weights), dtype=torch.float32, device=device)
    last = {
        "nll": 0.0,
        "kl": 0.0,
        "skipped": 0.0,
        "pi_loss": 0.0,
        "q_loss": 0.0,
        "v_loss": 0.0,
    }
    skipped = 0
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, rewards, dones, resets = _batch_to_torch(train_w, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden0 = actor.initial_hidden(obs.shape[0], device)
        logp, mean, log_std = _dist_stats(actor, obs, cond, hidden0, actions, resets)
        with torch.no_grad():
            _b_logp, b_mean, b_log_std = _dist_stats(
                base, obs, cond, hidden0, actions, resets
            )
        kl = gaussian_kl(mean, log_std, b_mean, b_log_std)
        if float(kl.detach().item()) > float(kl_max):
            skipped += 1
            last["kl"] = float(kl.detach().item())
            last["skipped"] = float(skipped)
            continue
        nll = -logp.mean()
        weighted = bool(
            abs(float(weights[0]) - 1.0) > 1e-6 or abs(float(weights[1]) - 1.0) > 1e-6
        )
        extra = (
            (weights * (mean - actions).square()).mean()
            if weighted
            else mean.new_zeros(())
        )
        pi_term = nll + extra
        q_loss = v_loss = mean.new_zeros(())
        if needs_q and heads is not None and q_opt is not None and v_opt is not None:
            batch, length, _ = obs.shape
            hidden_dim = int(actor.shapes().gru_hidden_dim)
            flat_obs = obs.reshape(batch * length, obs.shape[-1])
            flat_cond = (
                cond.unsqueeze(1)
                .expand(batch, length, -1)
                .reshape(batch * length, cond.shape[-1])
            )
            trunk = actor.encode_trunk(flat_obs, flat_cond).view(batch, length, -1)
            gru_out, _ = actor._gru_sequence(trunk, hidden0, resets)
            h = gru_out.detach()
            q_sa = heads.q_value(h.reshape(-1, hidden_dim), actions.reshape(-1, 2))
            q_sa = q_sa.view(batch, length)
            v_s = heads.v_value(h.reshape(-1, hidden_dim)).view(batch, length)
            q_loss = F.mse_loss(
                q_sa[:, :-1],
                (
                    rewards[:, :-1]
                    + float(gamma) * (1.0 - dones[:, :-1]) * v_s[:, 1:].detach()
                ),
            )
            if str(method) == "iql":
                v_loss = _expectile(q_sa.detach() - v_s, float(tau))
            else:
                v_loss = F.mse_loss(v_s, q_sa.detach())
            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            q_opt.step()
            v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            v_opt.step()
            adv = q_sa[:, :-1].detach() - v_s[:, :-1].detach()
            if str(method) == "crr":
                weight = (adv > 0.0).to(dtype=logp.dtype)
            else:
                weight = torch.exp((adv / max(float(beta), 1e-6)).clamp(max=10.0))
            pi_term = -(weight * logp[:, :-1]).mean() + extra
        loss = pi_term + float(kl_coef) * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        last = {
            "nll": float(nll.detach().item()),
            "kl": float(kl.detach().item()),
            "skipped": float(skipped),
            "pi_loss": float(pi_term.detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
        }
    last["skipped"] = float(skipped)
    return last


def refuse_ppo(method: str) -> None:
    if str(method).strip().lower() in PPO_METHODS:
        raise ValueError("never PPO-resume from actor-only weights; use bc or iql")


def winner_variations(winner_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dict(winner_spec)
    name = str(base.get("name", "winner"))
    kl = float(base.get("kl_coef", DEFAULT_KL_COEF))
    lr_scale = float(base.get("lr_scale", 1.0))
    return [
        {
            **base,
            "name": f"{name}-kl2x",
            "kl_coef": kl * 2.0,
            "variation_of": name,
        },
        {
            **base,
            "name": f"{name}-lr-half",
            "lr_scale": lr_scale * 0.5,
            "variation_of": name,
        },
    ]


def _write_actor(path: Path, payload: Mapping[str, Any], actor) -> str:
    FileArtifactStore().write_actor_artifact(
        str(path), bake_actor_artifact(payload, actor=actor)
    )
    return str(path)


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
    wandb_session: Any | None = None,
    arms: Sequence[Mapping[str, Any]] | None = None,
    include_variations: bool = True,
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

    specs = [dict(row) for row in (arms if arms is not None else UNIQUE_ARMS)]
    results: dict[str, dict[str, Any]] = {}
    actors: dict[str, ConditionedLidarGRUActor] = {}

    baseline_actor = copy.deepcopy(actor0)
    results["baseline"] = {
        "score": action_mse(baseline_actor, hold_w, payload, device),
        "holdout_kl": 0.0,
        "train": {},
        "artifact": _write_actor(output / "baseline_actor.pt", payload, baseline_actor),
        "spec": {"name": "baseline", "method": "none"},
    }
    actors["baseline"] = baseline_actor

    def _run_spec(spec: Mapping[str, Any]) -> None:
        name = str(spec["name"])
        refuse_ppo(str(spec.get("method", "bc")))
        student = copy.deepcopy(actor0)
        teacher = copy.deepcopy(actor0)
        teacher.eval()
        arm_lr = float(lr) * float(spec.get("lr_scale", 1.0))
        train = finetune(
            student,
            teacher,
            train_w,
            payload,
            method=str(spec["method"]),
            steps=int(spec.get("steps", steps)),
            batch_size=int(spec.get("batch_size", batch_size)),
            lr=arm_lr,
            seed=int(seed),
            device=device,
            kl_coef=float(spec.get("kl_coef", DEFAULT_KL_COEF)),
            kl_max=float(spec.get("kl_max", DEFAULT_KL_MAX)),
            action_weights=tuple(spec.get("action_weights", (1.0, 1.0))),
            tau=float(spec.get("tau", 0.7)),
            beta=float(spec.get("beta", 1.0)),
            speed_quantile=spec.get("speed_quantile"),
        )
        results[name] = {
            "score": action_mse(student, hold_w, payload, device),
            "holdout_kl": holdout_kl(student, teacher, hold_w, payload, device),
            "train": train,
            "artifact": _write_actor(output / f"{name}_actor.pt", payload, student),
            "spec": dict(spec),
        }
        actors[name] = student
        if wandb_session is not None and wandb_session.active:
            wandb_session.log_metrics(
                {
                    f"bakeoff/{SCALAR_NAME}_{name}": float(results[name]["score"]),
                    f"bakeoff/holdout_kl_{name}": float(results[name]["holdout_kl"]),
                },
                step=len(results),
            )

    if wandb_session is not None:
        wandb_session.start()

    for spec in specs:
        _run_spec(spec)

    unique_names = ["baseline", *[str(s["name"]) for s in specs]]
    unique_winner = min(unique_names, key=lambda n: float(results[n]["score"]))
    variation_specs: list[dict[str, Any]] = []
    if include_variations and unique_winner != "baseline":
        variation_specs = winner_variations(results[unique_winner]["spec"])
        for spec in variation_specs:
            _run_spec(spec)

    winner = min(results, key=lambda n: float(results[n]["score"]))
    keep = winner != "baseline"
    winner_path = output / "winner_actor.pt"
    _write_actor(winner_path, payload, actors[winner])
    result = {
        "scalar": SCALAR_NAME,
        "lower_is_better": True,
        "winner": winner,
        "unique_winner": unique_winner,
        "keep": keep,
        "decision": "keep" if keep else "revert",
        "winner_artifact": str(winner_path),
        "irl_identity": dict(payload.get("irl_identity") or {}),
        "arms": {
            name: {
                "score": float(row["score"]),
                "holdout_kl": float(row["holdout_kl"]),
                "train": row["train"],
                "artifact": row["artifact"],
                "spec": row["spec"],
            }
            for name, row in results.items()
        },
        "n_train_windows": int(train_w["obs"].shape[0]),
        "n_hold_windows": int(hold_w["obs"].shape[0]),
        "n_trajectories": int(len(trajs)),
        "device": str(device),
        "steps": int(steps),
        "checkpoint": str(checkpoint),
        "unique_arms": [str(s["name"]) for s in specs],
        "variation_arms": [str(s["name"]) for s in variation_specs],
    }
    (output / "bakeoff.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if wandb_session is not None:
        metrics = {
            f"bakeoff/{SCALAR_NAME}_{name}": float(row["score"])
            for name, row in results.items()
        }
        metrics["bakeoff/keep"] = float(keep)
        metrics["bakeoff/n_train_windows"] = float(result["n_train_windows"])
        metrics["bakeoff/n_hold_windows"] = float(result["n_hold_windows"])
        wandb_session.log_metrics(metrics, step=0)
        wandb_session.finish()
        result["wandb_run_id"] = wandb_session.run_id
    return result

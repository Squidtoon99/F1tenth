"""Off-policy fine-tune of a baked Gigaflow actor on teleop windows.

PPO resume is rejected: actor-only weights have no critic / optimizer state.
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
from gigaflow_f1tenth.teleop_dataset import (
    split_holdout,
    synthetic_trajectories,
    windows_from_trajectories,
)

SCALAR_NAME = "holdout_action_mse"
PPO_METHODS = {"ppo", "ppo-resume", "ppo_resume"}


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


def _expectile(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, tau, 1.0 - tau)
    return (weight * diff.square()).mean()


def finetune_bc(
    actor: ConditionedLidarGRUActor,
    windows: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    opt = torch.optim.Adam(actor.parameters(), lr=float(lr))
    actor.train()
    n = int(windows["obs"].shape[0])
    last = 0.0
    for step in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, _, _, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden = actor.initial_hidden(obs.shape[0], device)
        logp, _, _ = actor.evaluate_actions_sequence(
            obs, cond, hidden, actions, reset_mask=resets
        )
        loss = -logp.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        last = float(loss.detach().item())
        _ = step
    return {"train_nll": last}


def finetune_iql(
    actor: ConditionedLidarGRUActor,
    windows: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    tau: float = 0.7,
    gamma: float = 0.99,
    beta: float = 1.0,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    hidden_dim = int(actor.shapes().gru_hidden_dim)
    heads = IQLHeads(hidden_dim).to(device)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=float(lr))
    q_opt = torch.optim.Adam(heads.q.parameters(), lr=float(lr))
    v_opt = torch.optim.Adam(heads.v.parameters(), lr=float(lr))
    actor.train()
    heads.train()
    n = int(windows["obs"].shape[0])
    last = {"q_loss": 0.0, "v_loss": 0.0, "pi_loss": 0.0}
    for _ in range(int(steps)):
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, actions, rewards, dones, resets = _batch_to_torch(windows, device, idx)
        cond = _condition_tensor(payload, obs.shape[0], device)
        hidden0 = actor.initial_hidden(obs.shape[0], device)
        batch, length, _ = obs.shape
        flat_obs = obs.reshape(batch * length, obs.shape[-1])
        flat_cond = cond.unsqueeze(1).expand(batch, length, -1).reshape(
            batch * length, cond.shape[-1]
        )
        trunk = actor.encode_trunk(flat_obs, flat_cond).view(batch, length, -1)
        gru_out, _ = actor._gru_sequence(trunk, hidden0, resets)
        h = gru_out.detach()
        a = actions
        q_sa = heads.q_value(h.reshape(-1, hidden_dim), a.reshape(-1, 2)).view(
            batch, length
        )
        v_s = heads.v_value(h.reshape(-1, hidden_dim)).view(batch, length)
        q_loss = F.mse_loss(
            q_sa[:, :-1],
            (
                rewards[:, :-1]
                + float(gamma) * (1.0 - dones[:, :-1]) * v_s[:, 1:].detach()
            ),
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
        adv = (q_sa[:, :-1].detach() - v_s[:, :-1].detach()) / max(float(beta), 1e-6)
        weight = torch.exp(adv.clamp(max=10.0))
        pi_loss = -(weight * logp[:, :-1]).mean()
        pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        pi_opt.step()
        last = {
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "pi_loss": float(pi_loss.detach().item()),
        }
    return last


def refuse_ppo(method: str) -> None:
    if str(method).strip().lower() in PPO_METHODS:
        raise ValueError("never PPO-resume from actor-only weights; use bc or iql")


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
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actor0, payload = load_offline_actor(checkpoint, device=device)
    trajs = list(trajectories) if trajectories is not None else synthetic_trajectories(
        seed=seed
    )
    train_trajs, hold_trajs = split_holdout(trajs, seed=seed)
    train_w = windows_from_trajectories(train_trajs, window=window, stride=max(1, window // 2))
    hold_w = (
        windows_from_trajectories(hold_trajs, window=min(window, 8), stride=4)
        if hold_trajs
        else train_w
    )

    arms: dict[str, dict[str, Any]] = {}
    baseline_actor = copy.deepcopy(actor0)
    baseline_score = action_mse(baseline_actor, hold_w, payload, device)
    arms["baseline"] = {
        "score": baseline_score,
        "train": {},
        "artifact": str(output / "baseline_actor.pt"),
    }
    FileArtifactStore().write_actor_artifact(
        arms["baseline"]["artifact"], bake_actor_artifact(payload, actor=baseline_actor)
    )

    bc_actor = copy.deepcopy(actor0)
    bc_train = finetune_bc(
        bc_actor,
        train_w,
        payload,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
    )
    bc_score = action_mse(bc_actor, hold_w, payload, device)
    arms["bc"] = {
        "score": bc_score,
        "train": bc_train,
        "artifact": str(output / "bc_actor.pt"),
    }
    FileArtifactStore().write_actor_artifact(
        arms["bc"]["artifact"], bake_actor_artifact(payload, actor=bc_actor)
    )

    iql_actor = copy.deepcopy(actor0)
    iql_train = finetune_iql(
        iql_actor,
        train_w,
        payload,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed + 1,
        device=device,
    )
    iql_score = action_mse(iql_actor, hold_w, payload, device)
    arms["iql"] = {
        "score": iql_score,
        "train": iql_train,
        "artifact": str(output / "iql_actor.pt"),
    }
    FileArtifactStore().write_actor_artifact(
        arms["iql"]["artifact"], bake_actor_artifact(payload, actor=iql_actor)
    )

    winner = min(arms, key=lambda name: float(arms[name]["score"]))
    keep = winner != "baseline"
    winner_path = output / "winner_actor.pt"
    FileArtifactStore().write_actor_artifact(
        str(winner_path),
        bake_actor_artifact(
            payload,
            actor=(
                baseline_actor
                if winner == "baseline"
                else bc_actor
                if winner == "bc"
                else iql_actor
            ),
        ),
    )
    result = {
        "scalar": SCALAR_NAME,
        "lower_is_better": True,
        "winner": winner,
        "keep": keep,
        "decision": "keep" if keep else "revert",
        "winner_artifact": str(winner_path),
        "irl_identity": dict(payload.get("irl_identity") or {}),
        "arms": {
            name: {
                "score": float(row["score"]),
                "train": row["train"],
                "artifact": row["artifact"],
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

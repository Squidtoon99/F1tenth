"""Sophy-sim adapter: baked IRL condition, no sampling, Sophy PPO seam."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from gigaflow_f1tenth.artifacts import bake_actor_artifact
from gigaflow_f1tenth.irl_identity import baked_normalized_condition
from gigaflow_f1tenth.model import (
    ACT_LIMIT,
    ConditionedLidarGRUActor,
    actor_from_architecture,
    build_actor,
)
from gigaflow_f1tenth.rewards import CONDITION_DIM

PPO_RESUME_KEYS = ("optim", "optimizer", "actor_opt", "actor_optimizer")


def refuse_ppo_resume(payload: Mapping[str, Any] | str) -> None:
    if isinstance(payload, str):
        if payload.strip().lower() in {"ppo", "ppo-resume", "ppo_resume"}:
            raise ValueError(
                "never PPO-resume from actor-only weights; "
                "init a new critic and KL-clip to the frozen actor"
            )
        return
    ppo = payload.get("ppo")
    if not isinstance(ppo, Mapping):
        return
    if any(key in ppo for key in PPO_RESUME_KEYS):
        raise ValueError(
            "never PPO-resume from actor-only weights; "
            "init a new critic and KL-clip to the frozen actor"
        )


def gaussian_kl(
    mean_p: torch.Tensor,
    log_std_p: torch.Tensor,
    mean_q: torch.Tensor,
    log_std_q: torch.Tensor,
) -> torch.Tensor:
    var_p = torch.exp(log_std_p * 2.0)
    var_q = torch.exp(log_std_q * 2.0).clamp(min=1e-6)
    log_std_ratio = log_std_q - log_std_p
    quad = (var_p + (mean_p - mean_q).square()) / var_q
    return (log_std_ratio + 0.5 * (quad - 1.0)).sum(dim=-1).mean()


def _actor_state_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "actor_state_dict" in payload:
        return dict(payload["actor_state_dict"])
    ppo = payload.get("ppo")
    if isinstance(ppo, Mapping) and "actor" in ppo:
        return dict(ppo["actor"])
    raise KeyError("checkpoint missing actor weights")


def load_baked_actor(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
):
    """Load actor-only weights, bake IRL defaults, never restore PPO state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    actor_sd = _actor_state_dict(payload)
    stripped = {
        key: value
        for key, value in payload.items()
        if key not in {"critic", "central_critic", "ppo"}
    }
    stripped["actor_state_dict"] = actor_sd
    baked = bake_actor_artifact(stripped)
    arch = baked.get("actor_architecture")
    if isinstance(arch, Mapping):
        actor = actor_from_architecture(arch)
    else:
        from gigaflow_f1tenth.config import load_config

        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
        actor = build_actor(load_config(cfg_path))
    actor.load_state_dict(baked["actor_state_dict"], strict=True)
    return actor.to(device), baked


class BakedConditionActor(nn.Module):
    """Gigaflow CNN-GRU with a frozen IRL condition; Sophy PPO step seam."""

    def __init__(
        self,
        actor: ConditionedLidarGRUActor,
        condition: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        self.actor = actor
        shapes = actor.shapes()
        self.obs_dim = int(shapes.sensor_obs_dim)
        self.act_dim = int(shapes.action_dim)
        self.gru_hidden_dim = int(shapes.gru_hidden_dim)
        self.act_limit = float(ACT_LIMIT)
        vec = (
            baked_normalized_condition()
            if condition is None
            else condition
        )
        buffer = torch.as_tensor(vec, dtype=torch.float32)
        if tuple(buffer.shape) != (CONDITION_DIM,):
            raise ValueError(
                f"baked condition shape {tuple(buffer.shape)}; "
                f"expected ({CONDITION_DIM},)"
            )
        self.register_buffer("condition", buffer)
        for param in self.actor.condition_encoder.parameters():
            param.requires_grad_(False)

    def _cond(self, batch: int, device, dtype) -> torch.Tensor:
        return self.condition.to(device=device, dtype=dtype).unsqueeze(0).expand(
            int(batch), -1
        ).contiguous()

    def initial_hidden(self, batch_size, device="cpu", dtype=None, **_kwargs):
        hidden = self.actor.initial_hidden(int(batch_size), device=device)
        if dtype is not None:
            hidden = hidden.to(dtype=dtype)
        return hidden

    def step(
        self,
        obs,
        hidden,
        reset_mask=None,
        deterministic=False,
        with_logprob=True,
        return_pre_tanh=False,
    ):
        del with_logprob
        cond = self._cond(obs.shape[0], obs.device, obs.dtype)
        out = self.actor.forward(
            obs, cond, hidden, reset_mask=reset_mask, deterministic=deterministic
        )
        if return_pre_tanh:
            return out.actions, out.log_prob, out.hidden, out.pre_tanh
        return out.actions, out.log_prob, out.hidden

    def evaluate_actions_sequence(
        self,
        obs,
        hidden,
        actions,
        reset_mask=None,
        pre_tanh_actions=None,
        return_diagnostics=False,
        return_mean=False,
    ):
        cond = self._cond(obs.shape[0], obs.device, obs.dtype)
        result = self.actor.evaluate_actions_sequence(
            obs,
            cond,
            hidden,
            actions,
            reset_mask=reset_mask,
            pre_tanh=pre_tanh_actions,
            return_diagnostics=return_diagnostics,
            return_mean=return_mean,
        )
        logp, entropy, next_hidden, *extras = result
        if extras:
            return (logp, next_hidden, entropy, *extras)
        return logp, next_hidden, entropy

    def dist_stats(self, obs, hidden, actions, reset_mask=None, pre_tanh=None):
        cond = self._cond(obs.shape[0], obs.device, obs.dtype)
        logp, entropy, next_hidden, log_std, mean = self.actor.evaluate_actions_sequence(
            obs,
            cond,
            hidden,
            actions,
            reset_mask=reset_mask,
            pre_tanh=pre_tanh,
            return_diagnostics=True,
            return_mean=True,
        )
        del entropy, next_hidden
        return logp, mean, log_std

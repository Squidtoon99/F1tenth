"""On-car KL-clipped actor PPO from consecutive SOURCE_RL ticks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam


def actor_supports_online_ppo(actor: Any) -> bool:
    return all(
        hasattr(actor, name)
        for name in ("encode_trunk", "gru", "net", "mu_layer", "log_std_layer")
    )


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    out = torch.zeros_like(rewards)
    acc = rewards.new_zeros(())
    for t in range(int(rewards.shape[0]) - 1, -1, -1):
        acc = rewards[t] + float(gamma) * acc
        out[t] = acc
    return out


def _clone_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    clone = copy.deepcopy(module)
    return clone.to(torch.device("cpu"))


def _copy_state(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    with torch.no_grad():
        dst.load_state_dict(src.state_dict())


class OnlinePPOUpdater:
    def __init__(
        self,
        actor: torch.nn.Module,
        condition: torch.Tensor,
        *,
        rollout_steps: int = 128,
        lr: float = 3.0e-5,
        kl_coef: float = 0.1,
        kl_clip: float = 0.05,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        device: torch.device,
        output_dir: str,
        artifact_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if not actor_supports_online_ppo(actor):
            raise TypeError("online PPO requires a conditioned Gigaflow actor")
        if int(rollout_steps) < 2:
            raise ValueError("online_ppo_rollout_steps must be >= 2")
        self.actor = actor
        self.device = device
        self.condition = condition.to(device=device, dtype=torch.float32)
        if self.condition.dim() == 1:
            self.condition = self.condition.view(1, -1)
        self._train_actor = _clone_to_cpu(actor)
        self._base_actor = _clone_to_cpu(actor)
        self._train_actor.train()
        self._base_actor.eval()
        for parameter in self._base_actor.parameters():
            parameter.requires_grad_(False)
        self._train_condition = self.condition.detach().to("cpu")
        if self._train_condition.dim() == 1:
            self._train_condition = self._train_condition.view(1, -1)
        self.rollout_steps = int(rollout_steps)
        self.kl_coef = float(kl_coef)
        self.kl_clip = float(kl_clip)
        self.clip_ratio = float(clip_ratio)
        self.gamma = float(gamma)
        self.output_dir = Path(output_dir)
        self.artifact_payload = (
            dict(artifact_payload) if artifact_payload is not None else None
        )
        self.optimizer = Adam(self._train_actor.parameters(), lr=float(lr))
        self._obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._logp: list[float] = []
        self._rewards: list[float] = []

    def discard(self) -> None:
        self._obs.clear()
        self._actions.clear()
        self._logp.clear()
        self._rewards.clear()

    @property
    def buffered_steps(self) -> int:
        return len(self._obs)

    def record(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logp: float,
        reward: float,
    ) -> dict[str, float] | None:
        if not np.isfinite(logp) or not np.isfinite(reward):
            self.discard()
            return None
        if not np.isfinite(np.asarray(obs)).all():
            self.discard()
            return None
        if not np.isfinite(np.asarray(action)).all():
            self.discard()
            return None
        self._obs.append(np.asarray(obs, dtype=np.float32).reshape(-1).copy())
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        self._logp.append(float(logp))
        self._rewards.append(float(reward))
        if len(self._obs) < self.rollout_steps:
            return None
        stats = self._update()
        self.discard()
        return stats

    def _step_logp(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from gigaflow_f1tenth.model import _log_prob_actions

        trunk = self._train_actor.encode_trunk(obs, self._train_condition)
        out, h_n = self._train_actor.gru(
            trunk.unsqueeze(1), hidden.unsqueeze(0).contiguous()
        )
        logp, _entropy = _log_prob_actions(
            self._train_actor.net,
            self._train_actor.mu_layer,
            self._train_actor.log_std_layer,
            out.squeeze(1),
            action,
        )
        return logp, h_n.squeeze(0)

    def _sequence_logp(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        logps = []
        h = hidden
        for t in range(int(obs.shape[0])):
            logp, h = self._step_logp(obs[t : t + 1], actions[t : t + 1], h)
            logps.append(logp)
        return torch.cat(logps, dim=0)

    def _sync_inference_actor(self) -> None:
        with torch.no_grad():
            _copy_state(self._train_actor, self.actor)
            _copy_state(self._train_actor, self._base_actor)
        self.actor.eval()
        self._train_actor.eval()
        self._base_actor.eval()

    def _update(self) -> dict[str, float]:
        cpu = torch.device("cpu")
        obs = torch.as_tensor(np.stack(self._obs, axis=0), device=cpu, dtype=torch.float32)
        actions = torch.as_tensor(
            np.stack(self._actions, axis=0), device=cpu, dtype=torch.float32
        )
        old_logp = torch.as_tensor(self._logp, device=cpu, dtype=torch.float32)
        rewards = torch.as_tensor(self._rewards, device=cpu, dtype=torch.float32)
        returns = _discounted_returns(rewards, self.gamma)
        adv = returns - returns.mean()
        std = adv.std(unbiased=False)
        if float(std) > 1.0e-6:
            adv = adv / std
        hidden = self._train_actor.initial_hidden(1, device="cpu")
        self._train_actor.train()
        new_logp = self._sequence_logp(obs, actions, hidden)
        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        approx_kl = (old_logp - new_logp).mean()
        stats = {
            "policy_loss": float(policy_loss.detach()),
            "approx_kl": float(approx_kl.detach()),
            "steps": float(self.rollout_steps),
            "skipped": 0.0,
        }
        if float(approx_kl) > self.kl_clip:
            stats["skipped"] = 1.0
            self._train_actor.eval()
            _copy_state(self._base_actor, self._train_actor)
            return stats
        loss = policy_loss + self.kl_coef * approx_kl.clamp(min=0.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._train_actor.parameters(), 0.5)
        self.optimizer.step()
        stats["loss"] = float(loss.detach())
        self._sync_inference_actor()
        self.save()
        return stats

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "actor_latest.pt"
        payload = dict(self.artifact_payload or {})
        payload["actor_state_dict"] = {
            key: value.detach().cpu()
            for key, value in self._train_actor.state_dict().items()
        }
        vec = self._train_condition.detach().cpu().reshape(-1).tolist()
        payload["deployment_condition_normalized"] = vec
        payload["baked_condition"] = vec
        torch.save(payload, path)
        return path

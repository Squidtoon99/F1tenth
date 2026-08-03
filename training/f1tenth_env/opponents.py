from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from . import runtime as rt
from .geom import quat_to_xyz


@dataclass
class OpponentContext:

    step_state: dict[str, Any]
    opp_pos: torch.Tensor
    opp_vel: torch.Tensor
    opp_quat: torch.Tensor
    opp_last_actions: torch.Tensor
    env_cfg: dict[str, Any]
    device: torch.device
    ego_pos: torch.Tensor | None = None
    ego_vel: torch.Tensor | None = None
    ego_quat: torch.Tensor | None = None
    opp_obs: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class OpponentController:

    requires_observation: bool = False

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, mask: torch.Tensor) -> None:
        return None


class ScriptedCenterlineOpponent(OpponentController):

    requires_observation = False

    def __init__(self, env_cfg: dict[str, Any]):
        self.kp_ey = float(env_cfg.get("opponent_kp_ey", 1.0))
        self.kh_heading = float(env_cfg.get("opponent_kh_heading", 1.0))
        self.kp_speed = float(env_cfg.get("opponent_kp_speed", 1.0))
        self.target_speed = float(env_cfg.get("opponent_target_speed", 3.0))
        speed_range = env_cfg.get("opponent_target_speed_range")
        self.speed_range = tuple(speed_range) if speed_range else None
        self.lateral_offset_m = float(env_cfg.get("opponent_lateral_offset_m", 0.0))
        self.delta_max = float(
            env_cfg.get("delta_max", env_cfg.get("max_steer", 0.33))
        )
        self._speed_buf: torch.Tensor | None = None
        self._offset_buf: torch.Tensor | None = None

    def _ensure_bufs(self, num_envs: int, device: torch.device) -> None:
        if self._speed_buf is None or self._speed_buf.numel() != num_envs:
            self._speed_buf = torch.full(
                (num_envs,), self.target_speed, dtype=rt.tc_float, device=device
            )
            self._offset_buf = torch.zeros(
                (num_envs,), dtype=rt.tc_float, device=device
            )

    def reset(self, mask: torch.Tensor) -> None:
        if mask is None:
            return
        self._ensure_bufs(mask.shape[0], mask.device)
        assert self._speed_buf is not None and self._offset_buf is not None
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return
        if self.speed_range is not None:
            lo, hi = self.speed_range
            draws = torch.rand(n_reset, dtype=rt.tc_float, device=mask.device)
            self._speed_buf[mask] = draws * (hi - lo) + lo
        if self.lateral_offset_m > 0.0:
            draws = torch.rand(n_reset, dtype=rt.tc_float, device=mask.device)
            self._offset_buf[mask] = (draws * 2.0 - 1.0) * self.lateral_offset_m

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        frenet = ctx.step_state["frenet"]
        boundary = ctx.step_state["boundary"]
        ey = boundary["ey"].reshape(-1)

        seg_dir = frenet["seg_dir"]
        seg_dir = seg_dir / torch.linalg.norm(seg_dir, dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        speed = (ctx.opp_vel[:, :2] * seg_dir).sum(dim=-1)
        self._ensure_bufs(ey.shape[0], ey.device)
        assert self._speed_buf is not None and self._offset_buf is not None
        offset = self._offset_buf.to(ey.device, ey.dtype)
        target = self._speed_buf.to(speed.device, speed.dtype)

        track_angle = torch.atan2(seg_dir[:, 1], seg_dir[:, 0])
        yaw = quat_to_xyz(ctx.opp_quat, rpy=True, degrees=False)[:, 2]
        heading_err = yaw - track_angle
        heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

        speed_denom = speed.abs().add(0.5).clamp_min(0.5)
        cross_track = torch.atan(self.kp_ey * (ey - offset) / speed_denom)
        steer_target = -(self.kh_heading * heading_err + cross_track)
        max_steer = max(self.delta_max, 1e-6)
        steer = steer_target / max_steer
        steer = torch.clamp(steer, -1.0, 1.0)

        # Speed-tracking P-controller maps into force/brake effort: positive
        # error → drive current, negative → brake current (ADR 0006).
        throttle = torch.clamp(self.kp_speed * (target - speed), -1.0, 1.0)
        return torch.stack([throttle, steer], dim=-1)


@dataclass
class _PoolMember:
    actor: nn.Module
    obs_mean: torch.Tensor
    inv_std: torch.Tensor
    actor_compiled: bool = False


def _actor_is_recurrent(actor: torch.nn.Module) -> bool:
    raw = getattr(actor, "_orig_mod", actor)
    return (
        hasattr(raw, "step")
        and hasattr(raw, "initial_hidden")
        and hasattr(raw, "gru_hidden_dim")
    )


class PolicyOpponent(OpponentController):

    requires_observation = True

    def __init__(
        self,
        actor: torch.nn.Module,
        device: torch.device,
        obs_mean: torch.Tensor | None = None,
        obs_var: torch.Tensor | None = None,
        norm_eps: float = 1e-8,
        norm_clip: float = 10.0,
        act_clip: float = 1.0,
    ):
        self.actor = actor
        self.device = device
        self.norm_eps = float(norm_eps)
        self.norm_clip = float(norm_clip)
        self.act_clip = float(act_clip)
        self.obs_dim = int(getattr(actor, "obs_dim", actor.net[0].weight.shape[1]))
        self.act_dim = int(getattr(actor, "act_dim", actor.mu_layer.out_features))
        self.actor_architecture = dict(getattr(actor, "actor_architecture", {}))
        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_var = obs_var.to(device) if obs_var is not None else None
        self._inv_std: torch.Tensor | None = None
        self._actor_compiled = False
        self._hidden: torch.Tensor | None = None
        self._recurrent = _actor_is_recurrent(actor)
        self._pool: list[_PoolMember] | None = None
        self._env_policy_idx: torch.Tensor | None = None
        self._set_norm_stats(self.obs_mean, self.obs_var)
        self.actor.eval()

    def _raw_actor(self) -> torch.nn.Module:
        return getattr(self.actor, "_orig_mod", self.actor)

    def _ensure_hidden(self, num_envs: int) -> torch.Tensor:
        raw = self._raw_actor()
        if (
            self._hidden is None
            or self._hidden.shape[0] != num_envs
            or self._hidden.device != self.device
        ):
            self._hidden = raw.initial_hidden(
                num_envs, device=self.device, dtype=torch.float32
            )
        return self._hidden

    def reset(self, mask: torch.Tensor) -> None:
        if not self._recurrent or mask is None:
            return
        num_envs = int(mask.shape[0])
        hidden = self._ensure_hidden(num_envs)
        mask_b = mask.to(device=hidden.device, dtype=torch.bool)
        if mask_b.any():
            hidden[mask_b] = 0

    def _set_norm_stats(
        self, mean: torch.Tensor | None, var: torch.Tensor | None
    ) -> None:
        self.obs_mean = mean
        self.obs_var = var
        if mean is None or var is None:
            self._inv_std = None
            return
        self._inv_std = torch.rsqrt(var + self.norm_eps)

    def _maybe_compile_actor(self) -> None:
        if self._actor_compiled or self.device.type != "cuda":
            return
        self.actor = torch.compile(self._raw_actor(), mode="reduce-overhead")
        self._actor_compiled = True

    def _maybe_compile_pool_member(self, member: _PoolMember) -> None:
        if member.actor_compiled or self.device.type != "cuda":
            return
        raw = getattr(member.actor, "_orig_mod", member.actor)
        member.actor = torch.compile(raw, mode="reduce-overhead")
        member.actor_compiled = True

    @property
    def pool_size(self) -> int:
        return len(self._pool) if self._pool is not None else 1

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_mean is None or self._inv_std is None:
            return obs
        return self._normalize_with(self.obs_mean, self._inv_std, obs)

    def _normalize_with(
        self,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(
            (obs - mean) * inv_std, -self.norm_clip, self.norm_clip
        )

    def _member_inv_std(self, var: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(var.to(self.device, dtype=torch.float32) + self.norm_eps)

    def _reject_incompatible_snapshot(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        actor_architecture: dict | None,
    ) -> None:
        if actor_architecture is not None:
            if dict(actor_architecture) != self.actor_architecture:
                raise ValueError(
                    "Snapshot actor_architecture mismatch: "
                    f"got {dict(actor_architecture)!r}, "
                    f"expected {self.actor_architecture!r}"
                )
        raw = self._raw_actor()
        expected = raw.state_dict()
        expected_keys = set(expected)
        actual_keys = set(actor_state_dict)
        missing = sorted(expected_keys - actual_keys)
        if missing:
            raise ValueError(
                f"Snapshot missing actor key {missing[0]!r}; "
                f"expected architecture {self.actor_architecture!r}."
            )
        unexpected = sorted(actual_keys - expected_keys)
        if unexpected:
            raise ValueError(
                f"Snapshot unexpected actor key {unexpected[0]!r}; "
                f"expected architecture {self.actor_architecture!r}."
            )
        for key in sorted(expected_keys):
            want = tuple(int(d) for d in expected[key].shape)
            got = tuple(int(d) for d in actor_state_dict[key].shape)
            if got != want:
                raise ValueError(
                    f"Snapshot actor[{key!r}] shape={got}; expected {want}. "
                    "Privileged/symmetric or cross-architecture schemas are rejected."
                )

    def load_snapshot(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
        actor_architecture: dict | None = None,
    ) -> None:
        self._reject_incompatible_snapshot(actor_state_dict, actor_architecture)
        mean = obs_mean.to(self.device, dtype=torch.float32).reshape(-1)
        var = obs_var.to(self.device, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.obs_dim or var.numel() != self.obs_dim:
            raise ValueError(
                f"Snapshot normalizer dim mean={mean.numel()} var={var.numel()}; "
                f"expected actor obs dim {self.obs_dim}."
            )
        raw = self._raw_actor()
        raw.load_state_dict(actor_state_dict, strict=True)
        raw.to(device=self.device, dtype=torch.float32)
        raw.eval()
        self._recurrent = _actor_is_recurrent(raw)
        self._maybe_compile_actor()
        self._set_norm_stats(mean, var)
        # Snapshot weights changed; drop any carried GRU state.
        if self._hidden is not None:
            self._hidden.zero_()

    def load_opponent_pool(
        self,
        entries: list[Any],
    ) -> None:
        from f1tenth_policy import actor_from_architecture

        if not entries:
            raise ValueError("Opponent pool must contain at least one entry")
        first_arch = dict(entries[0].actor_architecture)
        for entry in entries[1:]:
            if dict(entry.actor_architecture) != first_arch:
                raise ValueError(
                    "Opponent pool actor_architecture mismatch across entries"
                )
        if dict(first_arch) != self.actor_architecture:
            raise ValueError(
                "Opponent pool architecture does not match env actor template"
            )
        members: list[_PoolMember] = []
        for entry in entries:
            actor = actor_from_architecture(first_arch).to(
                device=self.device, dtype=torch.float32
            )
            actor.load_state_dict(entry.actor, strict=True)
            actor.eval()
            mean = entry.mean.to(self.device, dtype=torch.float32).reshape(-1)
            var = entry.var.to(self.device, dtype=torch.float32).reshape(-1)
            if mean.numel() != self.obs_dim or var.numel() != self.obs_dim:
                raise ValueError(
                    f"Pool entry normalizer dim mean={mean.numel()} var={var.numel()}; "
                    f"expected actor obs dim {self.obs_dim}."
                )
            members.append(
                _PoolMember(
                    actor=actor,
                    obs_mean=mean,
                    inv_std=self._member_inv_std(var),
                )
            )
        self._pool = members
        self._env_policy_idx = None
        self._recurrent = _actor_is_recurrent(members[0].actor)
        for member in members:
            self._maybe_compile_pool_member(member)
        if self._hidden is not None:
            self._hidden.zero_()

    def assign_policies(
        self,
        mask: torch.Tensor,
        policy_indices: torch.Tensor,
    ) -> None:
        if self._pool is None:
            raise RuntimeError("assign_policies requires a loaded opponent pool")
        if self._env_policy_idx is None:
            self._env_policy_idx = torch.zeros(
                int(mask.shape[0]), dtype=torch.long, device=self.device
            )
        mask_b = mask.to(device=self.device, dtype=torch.bool)
        idx = policy_indices.to(device=self.device, dtype=torch.long).reshape(-1)
        if idx.numel() != int(mask_b.sum().item()):
            raise ValueError(
                f"assign_policies got {idx.numel()} indices for "
                f"{int(mask_b.sum().item())} reset envs"
            )
        if idx.min().item() < 0 or idx.max().item() >= len(self._pool):
            raise ValueError(
                f"assign_policies index out of range [0, {len(self._pool)})"
            )
        self._env_policy_idx[mask_b] = idx

    def policy_assignment_counts(self) -> torch.Tensor | None:
        if self._pool is None or self._env_policy_idx is None:
            return None
        return torch.bincount(
            self._env_policy_idx.cpu(),
            minlength=len(self._pool),
        )

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        if ctx.opp_obs is None:
            raise ValueError(
                "PolicyOpponent.requires_observation is True but ctx.opp_obs is None; "
                "the env must build the opponent's egocentric observation."
            )
        return self.act_observation(ctx.opp_obs)

    def act_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"PolicyOpponent observation dim={observation.shape[-1]}; "
                f"expected sensor actor dim {self.obs_dim}."
            )
        if not self._recurrent:
            raise ValueError(
                "PolicyOpponent requires a recurrent lidar_cnn_gru actor"
            )
        obs = observation.to(device=self.device, dtype=torch.float32)
        if self._pool is None:
            return self._act_single_policy(obs)
        return self._act_pooled_policies(obs)

    def _act_single_policy(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            model_obs = self._normalize(observation)
            hidden = self._ensure_hidden(observation.shape[0])
            action, _, next_hidden = self.actor.step(
                model_obs,
                hidden,
                reset_mask=None,
                deterministic=True,
                with_logprob=False,
            )
            self._hidden = next_hidden
            return torch.clamp(action, -self.act_clip, self.act_clip)

    def _act_pooled_policies(self, observation: torch.Tensor) -> torch.Tensor:
        assert self._pool is not None
        if self._env_policy_idx is None:
            raise RuntimeError("Opponent pool is loaded but env assignments are unset")
        num_envs = observation.shape[0]
        hidden = self._ensure_hidden(num_envs)
        actions = torch.empty(
            num_envs, self.act_dim, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            for policy_idx, member in enumerate(self._pool):
                env_mask = self._env_policy_idx == policy_idx
                if not bool(env_mask.any()):
                    continue
                env_ids = env_mask.nonzero(as_tuple=True)[0]
                model_obs = self._normalize_with(
                    member.obs_mean,
                    member.inv_std,
                    observation[env_ids],
                )
                action, _, next_hidden = member.actor.step(
                    model_obs,
                    hidden[env_ids],
                    reset_mask=None,
                    deterministic=True,
                    with_logprob=False,
                )
                actions[env_ids] = action
                hidden[env_ids] = next_hidden
        self._hidden = hidden
        return torch.clamp(actions, -self.act_clip, self.act_clip)


class MixedOpponentController(OpponentController):

    requires_observation = True

    def __init__(self, policy: PolicyOpponent):
        self.policy = policy
        self.mode_buf: torch.Tensor | None = None
        self.cap_buf: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor) -> None:
        self.policy.reset(mask)

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        raise RuntimeError(
            "MixedOpponentController.act is unused; Warp samples the mix in-kernel"
        )

    def load_snapshot(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
        actor_architecture: dict | None = None,
    ) -> None:
        self.policy.load_snapshot(
            actor_state_dict,
            obs_mean,
            obs_var,
            actor_architecture=actor_architecture,
        )


def make_opponent(
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> OpponentController | None:
    strategy = env_cfg.get("opponent_strategy")
    if strategy is None:
        return None
    if strategy == "scripted":
        return ScriptedCenterlineOpponent(env_cfg)
    if strategy == "policy":
        return _make_policy_opponent(env_cfg, obs_cfg, device)
    if strategy == "mixed":
        return MixedOpponentController(
            _make_policy_opponent(env_cfg, obs_cfg, device)
        )
    raise ValueError(f"Unknown opponent_strategy: {strategy!r}")


def _make_policy_opponent(
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> PolicyOpponent:
    # Imported lazily so the scripted path has no dependency on the RL stack.
    from qrsac import make_actor

    obs_dim = int(obs_cfg.get("num_actor_obs", obs_cfg["num_obs"]))
    act_dim = int(env_cfg.get("num_actions", 2))
    actor_type = env_cfg.get("actor_type", "lidar_cnn_gru")
    hidden = list(
        env_cfg.get(
            "actor_hidden_layers",
            env_cfg.get("opponent_hidden_layers", [1024, 1024, 1024]),
        )
    )
    lidar_pool_bins = int(env_cfg.get("lidar_pool_bins", 32))
    lidar_projection_dim = int(env_cfg.get("lidar_projection_dim", 256))
    layout_version = int(obs_cfg.get("actor_layout_version", 2))

    actor = make_actor(
        actor_type=actor_type,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden,
        activation=torch.nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=lidar_pool_bins,
        lidar_projection_dim=lidar_projection_dim,
    ).to(device=device, dtype=torch.float32)

    obs_mean = obs_var = None
    ckpt_path = env_cfg.get("opponent_ckpt")
    if ckpt_path:
        from f1tenth_policy import (
            actor_architecture_from_module,
            validate_sensor_policy_artifact,
        )

        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Opponent checkpoint payload must be a dict")
        expected_architecture = actor_architecture_from_module(actor)
        validate_sensor_policy_artifact(
            payload,
            expected_actor_obs_dim=obs_dim,
            expected_action_dim=act_dim,
            expected_layout_version=layout_version,
            expected_architecture=expected_architecture,
        )
        actor.load_state_dict(payload["actor"], strict=True)
        if "obs_norm" in payload:
            obs_mean = payload["obs_norm"]["mean"].to(dtype=torch.float32)
            obs_var = payload["obs_norm"]["var"].to(dtype=torch.float32)

    actor.to(device=device, dtype=torch.float32).eval()

    return PolicyOpponent(
        actor=actor,
        device=device,
        obs_mean=obs_mean,
        obs_var=obs_var,
        norm_eps=float(obs_cfg.get("norm_eps", 1e-8)),
        norm_clip=float(obs_cfg.get("norm_clip", 10.0)),
    )

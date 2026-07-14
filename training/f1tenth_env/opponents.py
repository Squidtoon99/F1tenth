"""Generic, pluggable opponent framework for 1v1 racing.

There is always exactly one opponent (hard 1v1; no multi-agent generality). The
environment talks only to an :class:`OpponentController`; whether the opponent is
a scripted controller or a neural policy is invisible to the env.

Concrete controllers:
- :class:`ScriptedCenterlineOpponent` - a centerline-following P-controller with
  closed-loop longitudinal speed control (shipped, the default 1v1 opponent).
- :class:`PolicyOpponent` - a frozen-policy opponent for self-play. The delayed
  snapshot/refresh training loop lives in ``standalone_trainer.py`` (``SelfPlayManager``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from . import runtime as rt
from .geom import quat_to_xyz


@dataclass
class OpponentContext:
    """Everything a controller needs to choose the opponent's actions this step.

    ``opp_obs`` is only populated when the controller declares
    ``requires_observation = True`` (e.g. the policy opponent), so the scripted
    opponent never pays for building a full egocentric observation.
    """

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
    """Base class for the single 1v1 opponent.

    Subclasses return ``(num_envs, num_actions)`` actions in ``[-1, 1]`` from
    :meth:`act`. ``requires_observation`` tells the env whether to build the
    opponent's full egocentric observation before calling :meth:`act`.
    """

    requires_observation: bool = False

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, mask: torch.Tensor) -> None:
        """Optional hook for stateful controllers. No-op by default."""
        return None


class ScriptedCenterlineOpponent(OpponentController):
    """Track-following opponent: P-control on lateral error + heading error.

    Each env row holds a target cruise speed (kept below the ego's pace so an
    overtake is feasible) and a lateral line offset, both sampled per reset. When
    ``opponent_target_speed_range`` is unset the speed is the fixed
    ``opponent_target_speed``; when ``opponent_lateral_offset_m`` is 0 the line is
    the centerline -- so the defaults reproduce a fixed-speed centerline follower.
    """

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
            env_cfg.get("delta_max", env_cfg.get("max_steer", 0.44))
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

        delta_max = max(self.delta_max, 1e-6)
        steer = -(self.kp_ey * (ey - offset) + self.kh_heading * heading_err) / delta_max
        steer = torch.clamp(steer, -1.0, 1.0)

        # Speed-tracking P-controller maps into force/brake effort: positive
        # error → drive current, negative → brake current (ADR 0006).
        throttle = torch.clamp(self.kp_speed * (target - speed), -1.0, 1.0)
        return torch.stack([throttle, steer], dim=-1)


class PolicyOpponent(OpponentController):
    """Frozen-policy opponent for future self-play.

    Wraps a ``SquashedGaussianMLPActor`` and (optional) observation-normalization
    statistics; acts deterministically on the opponent's egocentric observation.
    The self-play *training loop* that periodically refreshes ``actor`` from the
    learner is deferred - the env only needs this controller to exist.
    """

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
        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_var = obs_var.to(device) if obs_var is not None else None
        self.actor.eval()

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_mean is None or self.obs_var is None:
            return obs
        normed = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.norm_eps)
        return torch.clamp(normed, -self.norm_clip, self.norm_clip)

    def load_snapshot(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
    ) -> None:
        """Hot-swap frozen actor weights and observation-normalization stats."""
        self.actor.load_state_dict(actor_state_dict)
        self.actor.to(device=self.device, dtype=torch.float32)
        self.actor.eval()
        self.obs_mean = obs_mean.to(self.device, dtype=torch.float32)
        self.obs_var = obs_var.to(self.device, dtype=torch.float32)

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        if ctx.opp_obs is None:
            raise ValueError(
                "PolicyOpponent.requires_observation is True but ctx.opp_obs is None; "
                "the env must build the opponent's egocentric observation."
            )
        with torch.no_grad():
            model_obs = self._normalize(ctx.opp_obs.to(self.device, dtype=torch.float32))
            action, _ = self.actor(model_obs, deterministic=True, with_logprob=False)
        return torch.clamp(action, -self.act_clip, self.act_clip).to(ctx.opp_obs.device)


class MixedOpponentController(OpponentController):
    """Per-env mixed opponent population (GT Sophy-style).

    Each parallel env row is independently assigned, on reset, to either the
    scripted centerline follower or the frozen self-play policy according to the
    configured mix weights. This is the simple hard-1v1 analogue of GT Sophy's
    mixed opponent population (built-in slower AI + curated policy snapshots),
    which the paper found important so the agent does not overfit to pure
    self-play opponents.

    ``mode_buf`` is a per-row bool: ``True`` -> policy, ``False`` -> scripted.
    """

    requires_observation = True

    def __init__(
        self,
        scripted: ScriptedCenterlineOpponent,
        policy: PolicyOpponent,
        scripted_weight: float = 0.3,
        policy_weight: float = 0.7,
        policy_speed_cap_prob: float = 0.0,
        policy_speed_cap_range: tuple[float, float] = (2.5, 5.0),
    ):
        self.scripted = scripted
        self.policy = policy
        total = float(scripted_weight) + float(policy_weight)
        if total <= 0.0:
            raise ValueError(
                "opponent_mix weights must sum to a positive value; got "
                f"scripted={scripted_weight}, policy={policy_weight}"
            )
        self.policy_prob = float(policy_weight) / total
        self.cap_prob = float(policy_speed_cap_prob)
        self.cap_range = tuple(policy_speed_cap_range)
        self.mode_buf: torch.Tensor | None = None
        # Per-env speed cap for policy-mode rows; +inf means uncapped.
        self.cap_buf: torch.Tensor | None = None

    def _ensure_buf(self, num_envs: int, device: torch.device) -> None:
        if self.mode_buf is None or self.mode_buf.numel() != num_envs:
            # Default everyone to policy until the first reset assigns a mix.
            self.mode_buf = torch.ones(num_envs, dtype=torch.bool, device=device)
        if self.cap_buf is None or self.cap_buf.numel() != num_envs:
            self.cap_buf = torch.full(
                (num_envs,), float("inf"), dtype=rt.tc_float, device=device
            )

    def reset(self, mask: torch.Tensor) -> None:
        if mask is None:
            return
        self._ensure_buf(mask.shape[0], mask.device)
        assert self.mode_buf is not None and self.cap_buf is not None
        n_reset = int(mask.sum().item())
        if n_reset > 0:
            draws = torch.rand(n_reset, device=mask.device) < self.policy_prob
            self.mode_buf[mask] = draws
            # Reset caps for these rows, then cap a fraction of policy-mode rows.
            self.cap_buf[mask] = float("inf")
            if self.cap_prob > 0.0:
                lo, hi = self.cap_range
                cap_draw = torch.rand(mask.shape[0], device=mask.device) < self.cap_prob
                capped = mask & self.mode_buf & cap_draw
                # Skew caps toward the high end (u**2 concentrates near 0 -> caps
                # near hi), so most capped opponents are fast-but-passable rather
                # than trivially slow, with a thinner tail down to lo.
                u = torch.rand(mask.shape[0], dtype=rt.tc_float, device=mask.device)
                caps = hi - (hi - lo) * u * u
                self.cap_buf = torch.where(capped, caps, self.cap_buf)
        self.scripted.reset(mask)
        self.policy.reset(mask)

    def _apply_speed_cap(
        self, policy_act: torch.Tensor, ctx: OpponentContext
    ) -> torch.Tensor:
        """Coast (zero throttle) on policy-mode rows whose forward speed exceeds
        their per-env cap, so a capped opponent settles near a slower cruise."""
        assert self.cap_buf is not None
        if not bool(torch.isfinite(self.cap_buf).any()):
            return policy_act
        seg_dir = ctx.step_state["frenet"]["seg_dir"]
        seg_dir = seg_dir / torch.linalg.norm(
            seg_dir, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        speed = (ctx.opp_vel[:, :2] * seg_dir).sum(dim=-1)
        over = speed > self.cap_buf.to(speed.device, speed.dtype)
        throttle = torch.where(over, torch.zeros_like(policy_act[:, 0]), policy_act[:, 0])
        return torch.stack([throttle, policy_act[:, 1]], dim=-1)

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        num_envs = ctx.opp_pos.shape[0]
        self._ensure_buf(num_envs, ctx.opp_pos.device)
        assert self.mode_buf is not None
        scripted_act = self.scripted.act(ctx)
        policy_act = self._apply_speed_cap(self.policy.act(ctx), ctx)
        mode = self.mode_buf.to(scripted_act.device).unsqueeze(-1)
        return torch.where(mode, policy_act, scripted_act)

    def load_snapshot(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
    ) -> None:
        """Forward a self-play snapshot to the inner policy opponent."""
        self.policy.load_snapshot(actor_state_dict, obs_mean, obs_var)


def make_opponent(
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> OpponentController | None:
    """Factory keyed on ``env_cfg['opponent_strategy']``.

    Returns ``None`` when no opponent is configured (solo / 1v0). Hard 1v1: this
    only ever returns a single controller (the mixed controller still drives one
    opponent, just sampling its behavior per env row).
    """
    strategy = env_cfg.get("opponent_strategy")
    if strategy is None:
        return None
    if strategy == "scripted":
        return ScriptedCenterlineOpponent(env_cfg)
    if strategy == "policy":
        return _make_policy_opponent(env_cfg, obs_cfg, device)
    if strategy == "mixed":
        mix = env_cfg.get("opponent_mix", {})
        cap_range = mix.get("policy_speed_cap_range", (2.5, 5.0))
        return MixedOpponentController(
            scripted=ScriptedCenterlineOpponent(env_cfg),
            policy=_make_policy_opponent(env_cfg, obs_cfg, device),
            scripted_weight=float(mix.get("scripted_weight", 0.3)),
            policy_weight=float(mix.get("policy_weight", 0.7)),
            policy_speed_cap_prob=float(mix.get("policy_speed_cap_prob", 0.0)),
            policy_speed_cap_range=tuple(cap_range),
        )
    raise ValueError(f"Unknown opponent_strategy: {strategy!r}")


def _make_policy_opponent(
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> PolicyOpponent:
    # Imported lazily so the scripted path has no dependency on the RL stack.
    from qrsac import SquashedGaussianMLPActor

    obs_dim = int(obs_cfg["num_obs"])
    act_dim = int(env_cfg.get("num_actions", 2))
    hidden = list(env_cfg.get("opponent_hidden_layers", [512, 512, 512]))

    actor = SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden,
        activation=torch.nn.ReLU,
        act_limit=1.0,
    ).to(device=device, dtype=torch.float32)

    obs_mean = obs_var = None
    ckpt_path = env_cfg.get("opponent_ckpt")
    if ckpt_path:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(payload["actor"])
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

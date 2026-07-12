import math
import os
from typing import Any

import numpy as np
import torch

from . import geom as gu
from . import runtime as rt

from .domain_randomization import (
    apply_obs_dr,
    dr_metrics,
    init_dr_state,
    latency_actions,
    sample_dr_on_reset,
)
from f1tenth_sim.params import VehicleParams
from f1tenth_sim.suspension import quasi_static_loads
from .backends import TorchSimBackend
from .car import compute_tyre_slip
from .observations import build_observation, obs_opponent
from .opponents import (
    MixedOpponentController,
    OpponentContext,
    PolicyOpponent,
    make_opponent,
)
from .rewards import (
    compute_rewards,
    init_reward_state,
    sync_progress_state_for_resets,
)
from .terminations import (
    collision_mask,
    compute_terminations,
    init_termination_params,
    init_termination_state,
    reset_termination_state,
)
from .utils import (
    build_step_state,
    compute_oob_from_boundary_state,
    load_track_state,
)


class F1tenthEnv:

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        show_viewer=False,
        enable_recording=False,
    ):
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]

        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.show_viewer = show_viewer

        self.device = rt.device if rt.device is not None else torch.device("cpu")
        self.simulate_action_latency = self.env_cfg.get(
            "simulate_action_latency", False
        )
        self.dt = self.env_cfg.get("sim_dt", 0.01)
        self.control_interval = int(self.env_cfg.get("control_interval", 10))
        self.control_dt = self.dt * self.control_interval
        self.max_episode_steps = math.ceil(
            self.env_cfg["episode_length"] / self.control_dt
        )

        self.spawn_strategy = self.env_cfg.get("launch_strategy", "uniform_jittered")
        self.spawn_data = self.env_cfg.get("launch_strategy_data", {"num_cars": 20})

        self.obs_scales = obs_cfg["obs_scales"]
        self.track_cache_id = self.reward_cfg.get("track_cache_id", "track")

        self.track_state = load_track_state(
            track=self.env_cfg["track"],
            workspace_dir=os.path.dirname(os.path.dirname(__file__)),
            device=self.device,
        )

        self.centerline = self.track_state["centerline"]
        self.w_tr_left = self.track_state["w_tr_left"]
        self.w_tr_right = self.track_state["w_tr_right"]

        # On-device copies for sync-free reset sampling.
        self.centerline_t = torch.as_tensor(
            self.centerline, device=self.device, dtype=rt.tc_float
        )
        self.num_pts = int(self.centerline_t.shape[0])
        self.w_tr_left_torch = self.track_state["w_tr_left_torch"]
        self.w_tr_right_torch = self.track_state["w_tr_right_torch"]
        self.spawn_z = float(self.env_cfg.get("car_spawn_pos", (0.0, 0.0, 0.01))[2])

        self.backend = TorchSimBackend(
            num_envs=num_envs,
            env_cfg=self.env_cfg,
            obs_cfg=self.obs_cfg,
            reward_cfg=self.reward_cfg,
            track_state=self.track_state,
            device=self.device,
            show_viewer=show_viewer,
            enable_recording=enable_recording,
        )
        self.backend.build()
        self.has_opponent = self.backend.has_opponent
        self.n_wheels = 4

        if self.backend.has_opponent:
            self.opponent_ctrl = make_opponent(self.env_cfg, self.obs_cfg, self.device)
        else:
            self.opponent_ctrl = None

        # Mean centerline segment length, used to convert the opponent spawn gap
        # (meters) into a centerline index offset.
        seg = self.centerline_t[1:] - self.centerline_t[:-1]
        self._mean_seg_len = float(
            torch.linalg.norm(seg, dim=-1).mean().clamp_min(1e-6).item()
        )

        self.vehicle_mass = 3.74
        # Backend-agnostic vehicle geometry/mass, used to turn body-frame
        # accelerations into per-wheel normal-load ratios for the observation.
        self._susp_params = VehicleParams.from_config(self.env_cfg)
        base_action_latency = 1 if self.simulate_action_latency else 0
        self._dr = init_dr_state(
            num_envs=self.num_envs,
            env_cfg=self.env_cfg,
            device=self.device,
            num_actions=self.num_actions,
            base_vehicle_mass=self.vehicle_mass,
            base_tire_friction=float(self.env_cfg.get("tire_friction", 0.65)),
            base_action_latency=base_action_latency,
        )
        self._dr["num_obs"] = self.num_obs
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
        )

        self.base_lin_acc = torch.zeros(
            (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
        )

        self.base_pos = torch.empty(
            (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
        )
        self.base_quat = torch.empty(
            (self.num_envs, 4), dtype=rt.tc_float, device=rt.device
        )
        # World-frame ego velocity, retained for the opponent-relative observation
        # block (which works in world deltas rotated into each car's body frame).
        self.base_vel_world = torch.zeros(
            (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
        )

        # Opponent state buffers (only used when an opponent entity exists).
        if self.has_opponent:
            self.opp_base_pos = torch.zeros(
                (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
            )
            self.opp_base_quat = torch.zeros(
                (self.num_envs, 4), dtype=rt.tc_float, device=rt.device
            )
            self.opp_vel_world = torch.zeros(
                (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
            )
            self.opp_ang_world = torch.zeros(
                (self.num_envs, 3), dtype=rt.tc_float, device=rt.device
            )
            self.opp_last_actions = torch.zeros(
                (self.num_envs, self.num_actions), dtype=rt.tc_float, device=rt.device
            )

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=rt.tc_float, device=rt.device
        )
        self.reward_buf = torch.zeros(
            (self.num_envs,), dtype=rt.tc_float, device=rt.device
        )
        self.reset_buf = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=rt.device
        )

        self.episode_steps_buf = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=rt.device
        )
        self.lap_count_buf = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=rt.device
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=rt.tc_float, device=rt.device
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.reward_state = init_reward_state(
            reward_scales=self.reward_cfg["reward_scales"],
            num_envs=self.num_envs,
            device=self.device,
        )
        self.term_params = init_termination_params(self.env_cfg, self.control_dt)
        self.term_state = init_termination_state(self.num_envs, self.device)

        self.oob_consecutive_buf = self.term_state["oob_consecutive_buf"]
        self.not_moving_steps_buf = self.term_state["not_moving_steps_buf"]
        self.episode_sums = self.reward_state["episode_sums"]

        self.extras: dict[str, Any] = {
            "observations": {},
            "termination": {},
            "rewards": {},
            "metrics": {},
        }

        self._step_state: dict[str, Any] = {}
        self._step_state_valid = False
        self._opp_step_state: dict[str, Any] = {}
        self._opp_step_state_valid = False
        self._collision_state: dict[str, torch.Tensor] = {}
        self._collision_state_valid = False
        self._eval_launch_initialized = False
        self._build_observation = build_observation

        self.reset()

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """World-frame quaternion (B, 4) from yaw (B,), fully on-device."""
        yaw = yaw + float(self.env_cfg.get("reset_yaw_offset_rad", 0.0))
        half = 0.5 * yaw
        c = torch.cos(half)
        s = torch.sin(half)

        quat = torch.zeros((yaw.shape[0], 4), dtype=rt.tc_float, device=self.device)
        if str(self.env_cfg.get("reset_quat_order", "wxyz")).lower() == "wxyz":
            quat[:, 0] = c
            quat[:, 3] = s
        else:
            quat[:, 2] = s
            quat[:, 3] = c

        return quat

    def _reset_speed_range(self) -> tuple[float, float]:
        speed_range = self.spawn_data.get("mps_range")
        if speed_range is not None and len(speed_range) == 2:
            v_min, v_max = float(speed_range[0]), float(speed_range[1])
        else:
            v_min = float(self.env_cfg.get("reset_speed_min_mps", 0.0))
            v_max = float(self.env_cfg.get("reset_speed_max_mps", 0.0))
        if v_max < v_min:
            v_min, v_max = v_max, v_min
        return v_min, v_max

    def _sample_reset_speed(self) -> torch.Tensor:
        v_min, v_max = self._reset_speed_range()
        if abs(v_min) < 1e-8 and abs(v_max) < 1e-8:
            return torch.zeros((self.num_envs,), dtype=rt.tc_float, device=self.device)
        return (
            torch.rand((self.num_envs,), device=self.device, dtype=rt.tc_float)
            * (v_max - v_min)
            + v_min
        )

    def _opp_reset_speed_range(self) -> tuple[float, float]:
        opp_min = self.env_cfg.get("opponent_reset_speed_min_mps")
        opp_max = self.env_cfg.get("opponent_reset_speed_max_mps")
        if opp_min is not None and opp_max is not None:
            v_min, v_max = float(opp_min), float(opp_max)
        else:
            v_min, v_max = self._reset_speed_range()
        if v_max < v_min:
            v_min, v_max = v_max, v_min
        return v_min, v_max

    def _sample_opp_reset_speed(self) -> torch.Tensor:
        v_min, v_max = self._opp_reset_speed_range()
        if abs(v_min) < 1e-8 and abs(v_max) < 1e-8:
            return torch.zeros((self.num_envs,), dtype=rt.tc_float, device=self.device)
        return (
            torch.rand((self.num_envs,), device=self.device, dtype=rt.tc_float)
            * (v_max - v_min)
            + v_min
        )

    def _sample_lateral_offset(self, idx: torch.Tensor) -> torch.Tensor:
        w_left = self.w_tr_left_torch[idx]
        w_right = self.w_tr_right_torch[idx]
        spawn_margin = float(self.env_cfg.get("reset_spawn_margin_m", 0.2))
        max_left = (w_left - spawn_margin).clamp_min(0.05)
        max_right = (w_right - spawn_margin).clamp_min(0.05)
        B = idx.shape[0]
        return (
            torch.rand((B,), device=self.device, dtype=rt.tc_float)
            * (max_left + max_right)
            - max_right
        )

    def _sample_track_spawn_batch(
        self, centerline_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-batch (B,) on-device track spawn -> (pos (B, 3), quat (B, 4))."""
        if self.num_pts < 3:
            raise ValueError(
                "centerline must contain at least 3 points for reset sampling"
            )
        B = self.num_envs
        if centerline_idx is None:
            idx = torch.randint(0, self.num_pts, (B,), device=self.device)
        else:
            idx = centerline_idx.to(device=self.device, dtype=torch.long)

        prev_idx = (idx - 1) % self.num_pts
        next_idx = (idx + 1) % self.num_pts
        p_prev = self.centerline_t[prev_idx]
        p_curr = self.centerline_t[idx]
        p_next = self.centerline_t[next_idx]

        tangent = p_next - p_prev
        tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(
            1e-8
        )
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)

        lateral = self._sample_lateral_offset(idx)

        along_jitter = float(self.env_cfg.get("reset_along_track_jitter_m", 0.1))
        along = (
            torch.rand((B,), device=self.device, dtype=rt.tc_float) * 2.0 - 1.0
        ) * along_jitter

        spawn_xy = p_curr + normal * lateral.unsqueeze(1) + tangent * along.unsqueeze(1)

        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        yaw_jitter = float(self.env_cfg.get("reset_yaw_jitter_rad", 0.2))
        yaw = (
            yaw
            + (torch.rand((B,), device=self.device, dtype=rt.tc_float) * 2.0 - 1.0)
            * yaw_jitter
        )

        z = torch.full((B, 1), self.spawn_z, dtype=rt.tc_float, device=self.device)
        pos = torch.cat([spawn_xy, z], dim=1)
        quat = self._yaw_to_quat(yaw)
        return pos, quat

    def _centerline_frame(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tangent (B,2), normal (B,2), and point (B,2) for centerline indices."""
        prev_idx = (idx - 1) % self.num_pts
        next_idx = (idx + 1) % self.num_pts
        p_curr = self.centerline_t[idx]
        tangent = self.centerline_t[next_idx] - self.centerline_t[prev_idx]
        tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(
            1e-8
        )
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        return tangent, normal, p_curr

    def _spawn_opponent_distributed(
        self, ego_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Floor the along-track gap at the body length (+ collision margin) so two
        # cars can never spawn already overlapping under the oriented-box predicate.
        car_len = float(self.env_cfg.get("car_length", 0.568))
        min_gap = car_len + float(self.env_cfg.get("collision_margin_m", 0.0))
        gap_min = max(float(self.env_cfg.get("opponent_spawn_gap_min_m", 3.0)), min_gap)
        gap_max = max(float(self.env_cfg.get("opponent_spawn_gap_max_m", 20.0)), gap_min)
        behind_prob = float(self.env_cfg.get("opponent_spawn_behind_prob", 0.3))
        lateral_independent = bool(
            self.env_cfg.get("opponent_spawn_lateral_independent", True)
        )

        B = ego_pos.shape[0]
        ego_idx = self._closest_centerline_indices(ego_pos[:, :2])

        mag = (
            torch.rand((B,), device=self.device, dtype=rt.tc_float)
            * (gap_max - gap_min)
            + gap_min
        )
        behind = torch.rand((B,), device=self.device, dtype=rt.tc_float) < behind_prob
        sign = torch.where(behind, -1, 1)
        gap_pts = torch.clamp(
            (mag / self._mean_seg_len).round().to(dtype=torch.long), min=1
        )
        opp_idx = (ego_idx + sign * gap_pts) % self.num_pts

        if lateral_independent:
            lateral = self._sample_lateral_offset(opp_idx)
        else:
            _, normal_ego, p_ego = self._centerline_frame(ego_idx)
            lateral = ((ego_pos[:, :2] - p_ego) * normal_ego).sum(dim=-1)

        _, normal_opp, p_opp = self._centerline_frame(opp_idx)
        spawn_xy = p_opp + normal_opp * lateral.unsqueeze(1)

        tangent, _, _ = self._centerline_frame(opp_idx)
        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        z = torch.full((B, 1), self.spawn_z, dtype=rt.tc_float, device=self.device)
        pos = torch.cat([spawn_xy, z], dim=1)
        quat = self._yaw_to_quat(yaw)
        opp_speed = self._sample_opp_reset_speed()
        return pos, quat, opp_speed

    def _sample_spawn_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return full-batch (pos, quat, speed, preserve_buffers)."""
        B = self.num_envs
        preserve_buffers = False
        match self.spawn_strategy:
            case "fixed":
                pos = (
                    torch.tensor(
                        self.env_cfg["car_spawn_pos"],
                        dtype=rt.tc_float,
                        device=self.device,
                    )
                    .reshape(1, 3)
                    .expand(B, 3)
                    .contiguous()
                )
                euler = torch.tensor(
                    self.env_cfg["car_spawn_rot"],
                    dtype=rt.tc_float,
                    device=self.device,
                )
                q = gu.xyz_to_quat(euler, rpy=True, degrees=False)
                quat = q.reshape(1, 4).expand(B, 4).contiguous()
            case "eval_launch":
                if not self._eval_launch_initialized:
                    centerline_idx = torch.zeros(
                        (B,), dtype=torch.long, device=self.device
                    )
                    self._eval_launch_initialized = True
                else:
                    centerline_idx = self._closest_centerline_indices(
                        self.base_pos[:, :2]
                    )
                    preserve_buffers = True
                pos, quat = self._sample_track_spawn_batch(centerline_idx=centerline_idx)
            case _:
                pos, quat = self._sample_track_spawn_batch()
        speed = self._sample_reset_speed()
        return pos, quat, speed, preserve_buffers

    def _closest_centerline_indices(self, pos_xy: torch.Tensor) -> torch.Tensor:
        deltas = pos_xy[:, None, :] - self.centerline_t[None, :, :2]
        dist_sq = (deltas * deltas).sum(dim=-1)
        return torch.argmin(dist_sq, dim=1).to(dtype=torch.long)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        """Sync-free, mask-based reset of car state and per-env buffers.

        ``mask`` is a boolean (num_envs,) tensor. Reset sampling and buffer writes
        operate on the full batch and select rows via ``mask``/torch.where; the
        backend performs the (backend-specific) physics teleport/launch so this
        runs with no host-device synchronization regardless of how many envs reset.
        """
        pos, quat, speed, preserve_buffers = self._sample_spawn_batch()
        m = mask.unsqueeze(1)

        if self.has_opponent:
            opp_pos, opp_quat, opp_speed = self._spawn_opponent_distributed(pos)
        else:
            opp_pos, opp_quat, opp_speed = None, None, None

        sample_dr_on_reset(self._dr, mask, self.device)
        self.backend.reset(
            mask, pos, quat, speed, opp_pos, opp_quat, opp_speed, self._dr
        )

        if self.has_opponent:
            m_opp = mask.unsqueeze(1)
            self.opp_last_actions = torch.where(
                m_opp, torch.zeros_like(self.opp_last_actions), self.opp_last_actions
            )
            if self.opponent_ctrl is not None:
                self.opponent_ctrl.reset(mask)

        v_min, v_max = self._reset_speed_range()
        launch = not (abs(v_min) < 1e-8 and abs(v_max) < 1e-8)

        # Per-env buffers (masked writes, no sync).
        self.reset_buf = self.reset_buf & ~mask
        reset_termination_state(self.term_state, mask)

        if not preserve_buffers:
            self.episode_steps_buf = torch.where(
                mask, torch.zeros_like(self.episode_steps_buf), self.episode_steps_buf
            )
            self.lap_count_buf = torch.where(
                mask, torch.zeros_like(self.lap_count_buf), self.lap_count_buf
            )
            zero_act = torch.zeros_like(self.actions)
            self.actions = torch.where(m, zero_act, self.actions)
            self.last_actions = torch.where(m, zero_act, self.last_actions)

            if launch:
                max_speed = max(float(self.env_cfg.get("max_speed", 5.0)), 1e-6)
                clip_actions = float(self.env_cfg.get("clip_actions", 1.0))
                reset_throttle = torch.clamp(
                    speed / max_speed, min=-clip_actions, max=clip_actions
                )
                self.actions[:, 0] = torch.where(
                    mask, reset_throttle, self.actions[:, 0]
                )
                self.last_actions[:, 0] = torch.where(
                    mask, reset_throttle, self.last_actions[:, 0]
                )

            for value in self.reward_state["episode_sums"].values():
                value.masked_fill_(mask, 0.0)

    def _get_step_state(self) -> dict[str, Any]:
        if not self._step_state_valid:
            self._step_state = build_step_state(
                base_pos=self.base_pos,
                episode_steps_buf=self.episode_steps_buf,
                track_state=self.track_state,
                device=self.device,
                cache_id=self.track_cache_id,
            )

            ws = self.backend.read_wheel_state("ego")
            self._step_state["wheel_state"] = ws
            active = self.actions[:, 0].abs() > 1e-3
            self._step_state["tyre_slip"] = self._tyre_slip_from(ws, active)
            ego_mass = self._dr["vehicle_mass"] * self._dr["mass_scale"]
            self._step_state["tyre_load"] = self._tyre_load_from(
                ws, self.base_lin_acc, ego_mass
            )
            self._step_state["boundary"]["oob_half_extent_m"] = (
                self._oob_half_extent()
            )
            self._step_state_valid = True
        return self._step_state

    def _oob_half_extent(self) -> torch.Tensor:
        """Car half-extent along the track normal, for footprint-aware OOB.

        A rectangle of ``car_length`` x ``car_width`` yawed by ``heading_err``
        relative to the track tangent reaches this far past its centre toward the
        boundaries, so the whole body (not just the centre) is bounded.
        """
        seg_dir = self._step_state["frenet"]["seg_dir"]
        track_angle = torch.atan2(seg_dir[:, 1], seg_dir[:, 0])
        yaw = gu.quat_to_xyz(self.base_quat, rpy=True, degrees=False)[:, 2]
        heading_err = torch.atan2(
            torch.sin(yaw - track_angle), torch.cos(yaw - track_angle)
        )
        car_length = float(self.env_cfg.get("car_length", 0.0))
        car_width = float(self.env_cfg.get("car_width", 0.0))
        return 0.5 * (
            car_length * heading_err.sin().abs()
            + car_width * heading_err.cos().abs()
        )

    def _tyre_slip_from(
        self, wheel_state: dict[str, Any], active: torch.Tensor
    ) -> torch.Tensor:
        """Per-wheel slip (N,8): simulator-native if provided, else computed."""
        native = wheel_state.get("tyre_slip")
        if native is not None:
            return native
        wheel_radius = float(self.env_cfg.get("wheel_radius", 0.05))
        return compute_tyre_slip(
            wheel_state,
            wheel_radius=wheel_radius,
            active=active,
            min_lat=self._susp_params.slip_min_lat,
            min_active_long=self._susp_params.slip_min_active_long,
            min_passive_long=self._susp_params.slip_min_passive_long,
        )

    def _tyre_load_from(
        self,
        wheel_state: dict[str, Any],
        base_lin_acc: torch.Tensor,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-wheel normal-load ratio (N,4): backend-native if provided, else
        the quasi-static load transfer from body accel (matches the on-car IMU
        derivation)."""
        native = wheel_state.get("tyre_load")
        if native is not None:
            return native
        return self._compute_tyre_load(base_lin_acc, mass)

    def _compute_tyre_load(
        self, base_lin_acc: torch.Tensor, mass: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-wheel normal-load ratio Fz / Fz_static from body accel (N,4).

        Backend-agnostic (uses the quasi-static load transfer), so it matches what
        the on-car observation builder derives from IMU. ``mass`` is the per-env
        (optionally DR-sampled) mass; ``None`` uses the nominal mass.
        """
        fz = quasi_static_loads(
            self._susp_params, base_lin_acc[:, 0], base_lin_acc[:, 1], mass
        )
        return fz / max(self._susp_params.static_wheel_load(), 1e-6)

    def _update_state_buffers(self):
        st = self.backend.read_state()
        self.base_pos = st["base_pos"]
        self.base_quat = st["base_quat"]
        self.base_vel_world = st["base_vel_world"]
        self.base_lin_vel = st["base_lin_vel"]
        self.base_ang_vel = st["base_ang_vel"]
        self.base_lin_acc = st["base_lin_acc"]

        if self.has_opponent:
            self.opp_base_pos = st["opp_base_pos"]
            self.opp_base_quat = st["opp_base_quat"]
            self.opp_vel_world = st["opp_vel_world"]
            self.opp_ang_world = st["opp_ang_world"]

        self._step_state_valid = False
        self._opp_step_state_valid = False
        self._collision_state_valid = False

    def _update_observation(self):
        step_state = self._get_step_state()
        opponent_block = self._ego_opponent_block(step_state)
        self.obs_buf = self._build_observation(
            num_obs=self.num_obs,
            num_envs=self.num_envs,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            base_lin_acc=self.base_lin_acc,
            last_actions=self.last_actions,
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            obs_cfg=self.obs_cfg,
            step_state=step_state,
            device=self.device,
            opponent_block=opponent_block,
        )
        self.obs_buf = apply_obs_dr(self._dr, self.obs_buf)

    def _compute_rewards(self):
        step_state = self._get_step_state()
        step_state["base_lin_vel"] = self.base_lin_vel
        step_state["actions"] = self.actions
        step_state["last_actions"] = self.last_actions
        if self.has_opponent:
            opp_ss = self._opponent_step_state()
            step_state["opp_s"] = opp_ss["frenet"]["s"]
            # World-frame ego and opponent velocities for the GT Sophy rear-end
            # penalty (Rr), which scales with the squared closing speed
            # ||v_ego - v_opp||^2. Both must be in the same (world) frame; note
            # base_lin_vel is the body-frame velocity and must not be used here.
            step_state["ego_vel_world"] = self.base_vel_world
            step_state["opp_vel_world"] = self.opp_vel_world
            # Same ego-frame box overlap predicate used for collision termination,
            # exposed to the reward path for the GT Sophy any-collision penalty.
            step_state["car_collision"] = self._get_collision_state()["overlap"]
        self.reward_buf, self._step_state = compute_rewards(
            step_state=step_state,
            reward_cfg=self.reward_cfg,
            reward_state=self.reward_state,
            episode_steps_buf=self.episode_steps_buf,
            lap_count_buf=self.lap_count_buf,
        )
        self.extras["rewards"]["total"] = self.reward_buf
        self.extras["rewards"]["terms"] = self.reward_state.get("last_reward_terms", {})

    def _compute_terminations(self):
        step_state = self._get_step_state()
        self.reset_buf, self.extras["termination"], self.extras["time_outs"] = (
            compute_terminations(
                step_state=step_state,
                episode_steps_buf=self.episode_steps_buf,
                max_episode_steps=self.max_episode_steps,
                base_pos=self.base_pos,
                base_quat=self.base_quat,
                base_lin_vel=self.base_lin_vel,
                base_ang_vel=self.base_ang_vel,
                term_state=self.term_state,
                term_params=self.term_params,
            )
        )

        # Collision termination (1v1): anisotropic ego-frame box overlap, gated by
        # closing speed so only high-speed impacts end the episode. Low-speed taps
        # still incur the collision/rear-end penalties and contact physics but let
        # the agent keep driving.
        if self.has_opponent and bool(
            self.env_cfg.get("term_on_collision", True)
        ):
            collision_state = self._get_collision_state()
            overlap = collision_state["overlap"]
            term_speed = float(self.env_cfg.get("collision_term_speed_mps", 0.0))
            if term_speed > 0.0:
                collision = overlap & (collision_state["closing_speed"] > term_speed)
            else:
                collision = overlap
            self.reset_buf = self.reset_buf | collision
            self.extras["termination"]["collision"] = collision.to(dtype=rt.tc_float)

        boundary = step_state["boundary"]
        oob_mask, oob_dist = compute_oob_from_boundary_state(
            boundary,
            margin_m=float(self.term_params["term_oob_margin_m"]),
        )
        progress_ds = step_state.get(
            "progress_ds",
            torch.zeros((self.num_envs,), dtype=rt.tc_float, device=self.device),
        )
        speed_xy = torch.linalg.norm(self.base_lin_vel[:, :2], dim=-1)

        self.extras["metrics"] = {
            "progress_ds": progress_ds,
            "oob_dist": oob_dist,
            "oob_mask": oob_mask.to(dtype=rt.tc_float),
            "boundary_dist": boundary["boundary_dist"],
            "lateral_error": boundary["ey"],
            "speed_xy": speed_xy,
            "episode_steps": self.episode_steps_buf.to(dtype=rt.tc_float),
            "lap_count": self.lap_count_buf.to(dtype=rt.tc_float),
            "laps_completed": self.reward_state.get(
                "last_lap_cross",
                torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            ).to(dtype=rt.tc_float),
        }
        if self.has_opponent:
            opp_ss = self._opponent_step_state()
            seg_dir = opp_ss["frenet"]["seg_dir"]
            seg_dir = seg_dir / torch.linalg.norm(
                seg_dir, dim=-1, keepdim=True
            ).clamp_min(1e-6)
            opp_speed = (self.opp_vel_world[:, :2] * seg_dir).sum(dim=-1)
            self.extras["metrics"]["opp_speed"] = opp_speed
        self.extras["metrics"].update(dr_metrics(self._dr))

    def _record_nonfinite_metrics(self) -> None:
        """Per-step counts of env rows with non-finite obs/reward/physics state."""
        nf_obs = (~torch.isfinite(self.obs_buf)).any(dim=1)
        nf_reward = ~torch.isfinite(self.reward_buf)
        nf_state = ~(
            torch.isfinite(self.base_pos).all(dim=1)
            & torch.isfinite(self.base_quat).all(dim=1)
            & torch.isfinite(self.base_lin_vel).all(dim=1)
            & torch.isfinite(self.base_ang_vel).all(dim=1)
        )
        if self.has_opponent:
            nf_state = nf_state | ~(
                torch.isfinite(self.opp_base_pos).all(dim=1)
                & torch.isfinite(self.opp_base_quat).all(dim=1)
            )
        self.extras["metrics"]["nonfinite_obs_envs"] = nf_obs.to(dtype=rt.tc_float)
        self.extras["metrics"]["nonfinite_reward_envs"] = nf_reward.to(
            dtype=rt.tc_float
        )
        self.extras["metrics"]["nonfinite_state_envs"] = nf_state.to(dtype=rt.tc_float)

    def _normalize_reset_mask(self, envs_idx) -> torch.Tensor:
        """Coerce a None / index-list / index-tensor / bool-mask into a bool mask."""
        if envs_idx is None:
            return torch.ones((self.num_envs,), dtype=torch.bool, device=rt.device)
        if isinstance(envs_idx, (list, tuple, np.ndarray)):
            mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=rt.device)
            if len(envs_idx) > 0:
                mask[list(envs_idx)] = True
            return mask
        if envs_idx.dtype == torch.bool:
            return envs_idx
        mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=rt.device)
        mask[envs_idx] = True
        return mask

    def reset(self, envs_idx=None):
        """Sync-free reset. Safe to call unconditionally every step with a (possibly
        all-False) done mask: masked envs are teleported/zeroed and the rest are
        untouched, with no host-device synchronization."""
        mask = self._normalize_reset_mask(envs_idx)
        if not bool(mask.any()):
            self._update_observation()
            self.extras["observations"]["critic"] = self.obs_buf
            return self.obs_buf, self.extras

        self._reset_envs(mask)

        # Re-read state after the masked teleport (immediate FK, no scene.step) and
        # force fresh kinematics for the reset envs.
        self._update_state_buffers()
        self.base_lin_acc = torch.where(
            mask.unsqueeze(1), torch.zeros_like(self.base_lin_acc), self.base_lin_acc
        )

        step_state = self._get_step_state()
        sync_progress_state_for_resets(
            reward_state=self.reward_state,
            step_state=step_state,
            episode_steps_buf=self.episode_steps_buf,
            reset_mask=mask,
        )

        self._update_observation()
        self.extras["observations"]["critic"] = self.obs_buf
        self.extras.setdefault("metrics", {}).update(dr_metrics(self._dr))
        return self.obs_buf, self.extras

    def _apply_actions(
        self, exec_actions: torch.Tensor, env_ids: torch.Tensor | None = None
    ):
        """Apply throttle/brake and lagged steering to the ego car via the backend."""
        self.backend.apply_ego_actions(exec_actions, self.base_lin_vel, self._dr)

    def _opponent_step_state(
        self, pos: torch.Tensor | None = None
    ) -> dict[str, Any]:
        target_pos = self.opp_base_pos if pos is None else pos
        use_cache = pos is None or pos is self.opp_base_pos
        if not use_cache:
            return build_step_state(
                base_pos=target_pos,
                episode_steps_buf=self.episode_steps_buf,
                track_state=self.track_state,
                device=self.device,
                cache_id="opponent",
            )
        if not self._opp_step_state_valid:
            self._opp_step_state = build_step_state(
                base_pos=target_pos,
                episode_steps_buf=self.episode_steps_buf,
                track_state=self.track_state,
                device=self.device,
                cache_id="opponent",
            )
            self._opp_step_state_valid = True
        return self._opp_step_state

    def _get_collision_state(self) -> dict[str, torch.Tensor]:
        if not self._collision_state_valid:
            ego_yaw = gu.quat_to_xyz(self.base_quat, rpy=True, degrees=False)[:, 2]
            opp_yaw = gu.quat_to_xyz(
                self.opp_base_quat, rpy=True, degrees=False
            )[:, 2]
            self._collision_state = {
                "overlap": collision_mask(
                    self.base_pos[:, :2],
                    self.opp_base_pos[:, :2],
                    ego_yaw,
                    opp_yaw,
                    car_length=float(self.env_cfg.get("car_length", 0.568)),
                    car_width=float(self.env_cfg.get("car_width", 0.296)),
                    collision_margin_m=float(
                        self.env_cfg.get("collision_margin_m", 0.0)
                    ),
                ),
                "closing_speed": torch.linalg.norm(
                    self.base_vel_world[:, :2] - self.opp_vel_world[:, :2], dim=-1
                ),
            }
            self._collision_state_valid = True
        return self._collision_state

    def _agent_state(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        vel_world: torch.Tensor,
        step_state: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Pack the per-car fields the symmetric opponent-obs block needs."""
        yaw = gu.quat_to_xyz(quat, rpy=True, degrees=False)[:, 2]
        return {
            "pos_xy": pos[:, :2],
            "yaw": yaw,
            "vel_xy": vel_world[:, :2],
            "s": step_state["frenet"]["s"],
            "ey": step_state["boundary"]["ey"],
            "L": step_state["frenet"]["L"],
        }

    def _wrapped_track_gap(
        self, s_self: torch.Tensor, s_other: torch.Tensor, track_len: torch.Tensor
    ) -> torch.Tensor:
        gap = s_other.reshape(-1) - s_self.reshape(-1)
        half = 0.5 * track_len.reshape(-1).to(gap.dtype)
        gap = torch.where(gap > half, gap - track_len.reshape(-1).to(gap.dtype), gap)
        gap = torch.where(gap < -half, gap + track_len.reshape(-1).to(gap.dtype), gap)
        return gap

    def _apply_opponent_range_mask(
        self,
        block: torch.Tensor,
        s_self: torch.Tensor,
        s_other: torch.Tensor,
        track_len: torch.Tensor,
    ) -> torch.Tensor:
        ahead_m = float(self.obs_cfg.get("opp_obs_ahead_m", 40.0))
        behind_m = float(self.obs_cfg.get("opp_obs_behind_m", 20.0))
        gap = self._wrapped_track_gap(s_self, s_other, track_len)
        in_range = (gap <= ahead_m) & (gap >= -behind_m)
        return block * in_range.to(block.dtype).unsqueeze(-1)

    def _ego_opponent_block(self, ego_step_state: dict[str, Any]) -> torch.Tensor | None:
        """Opponent-relative observation block from the ego's frame (or None)."""
        if not self.has_opponent or not bool(
            self.obs_cfg.get("enable_opponent_obs", False)
        ):
            return None
        opp_ss = self._opponent_step_state()
        block = obs_opponent(
            self._agent_state(
                self.base_pos, self.base_quat, self.base_vel_world, ego_step_state
            ),
            self._agent_state(
                self.opp_base_pos, self.opp_base_quat, self.opp_vel_world, opp_ss
            ),
            self.obs_cfg,
        )
        return self._apply_opponent_range_mask(
            block,
            ego_step_state["frenet"]["s"],
            opp_ss["frenet"]["s"],
            ego_step_state["frenet"]["L"],
        )

    def _build_opponent_obs(self, opp_step_state: dict[str, Any]) -> torch.Tensor:
        """Opponent's own egocentric observation (only for a PolicyOpponent)."""
        opp_body_vel = gu.inv_transform_by_quat(self.opp_vel_world, self.opp_base_quat)
        opp_ang_vel = gu.inv_transform_by_quat(self.opp_ang_world, self.opp_base_quat)
        opp_acc = torch.zeros_like(opp_body_vel)

        opp_ss = dict(opp_step_state)
        opp_ws = self.backend.read_wheel_state("opp")
        opp_ss["wheel_state"] = opp_ws
        opp_active = self.opp_last_actions[:, 0].abs() > 1e-3
        opp_ss["tyre_slip"] = self._tyre_slip_from(opp_ws, opp_active)
        opp_ss["tyre_load"] = self._tyre_load_from(opp_ws, opp_acc)

        ego_ss = self._get_step_state()
        opp_block = obs_opponent(
            self._agent_state(
                self.opp_base_pos, self.opp_base_quat, self.opp_vel_world, opp_ss
            ),
            self._agent_state(
                self.base_pos, self.base_quat, self.base_vel_world, ego_ss
            ),
            self.obs_cfg,
        )
        opp_block = self._apply_opponent_range_mask(
            opp_block,
            opp_ss["frenet"]["s"],
            ego_ss["frenet"]["s"],
            opp_ss["frenet"]["L"],
        )
        return build_observation(
            num_obs=self.num_obs,
            num_envs=self.num_envs,
            base_lin_vel=opp_body_vel,
            base_ang_vel=opp_ang_vel,
            base_lin_acc=opp_acc,
            last_actions=self.opp_last_actions,
            base_pos=self.opp_base_pos,
            base_quat=self.opp_base_quat,
            obs_cfg=self.obs_cfg,
            step_state=opp_ss,
            device=self.device,
            opponent_block=opp_block,
        )

    def _apply_opponent_actions(self) -> None:
        """Query the opponent controller and actuate the opponent entity."""
        if not self.has_opponent or self.opponent_ctrl is None:
            return
        opp_ss = self._opponent_step_state()
        opp_obs = None
        if self.opponent_ctrl.requires_observation:
            opp_obs = self._build_opponent_obs(opp_ss)
        ctx = OpponentContext(
            step_state=opp_ss,
            opp_pos=self.opp_base_pos,
            opp_vel=self.opp_vel_world,
            opp_quat=self.opp_base_quat,
            opp_last_actions=self.opp_last_actions,
            env_cfg=self.env_cfg,
            device=self.device,
            ego_pos=self.base_pos,
            ego_vel=self.base_vel_world,
            ego_quat=self.base_quat,
            opp_obs=opp_obs,
        )
        opp_actions = self.opponent_ctrl.act(ctx)
        opp_actions = torch.clip(
            opp_actions,
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )
        opp_body_vel = gu.inv_transform_by_quat(
            self.opp_vel_world, self.opp_base_quat
        )
        self.backend.apply_opp_actions(opp_actions, opp_body_vel, self._dr)
        self.opp_last_actions = opp_actions

    def refresh_opponent_policy(
        self,
        state_dict: dict[str, torch.Tensor],
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
    ) -> None:
        """Hot-swap the policy opponent's weights and obs-norm stats (self-play).

        Works for a plain PolicyOpponent and for the MixedOpponentController, which
        forwards the snapshot to its inner policy.
        """
        if not isinstance(self.opponent_ctrl, (PolicyOpponent, MixedOpponentController)):
            return
        self.opponent_ctrl.load_snapshot(state_dict, obs_mean, obs_var)

    def step(self, actions, n_steps=1):
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )
        exec_actions = latency_actions(self._dr, self.actions)

        self._apply_actions(exec_actions)
        self._apply_opponent_actions()
        self.backend.substep(n_steps)
        self.episode_steps_buf += 1
        self._update_state_buffers()

        self._compute_rewards()
        self._compute_terminations()

        done = self.reset_buf.clone()
        # Unconditional, mask-based reset: no per-step host-device sync. reset()
        # recomputes the observation for the full batch (reset envs reflect their
        # new spawn pose), so no separate _update_observation is needed.
        self.reset(done)

        self.last_actions.copy_(self.actions)
        self._record_nonfinite_metrics()
        return self.obs_buf, self.reward_buf, done, self.extras

    def close(self):
        self.backend.close()

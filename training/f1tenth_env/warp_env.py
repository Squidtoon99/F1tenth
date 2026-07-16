from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from f1tenth_sim.dynamics import VehicleBuffers
from f1tenth_sim.params import VehicleParams

from . import runtime as rt
from .kernel import (
    ACTION_HISTORY,
    OBS_DIM,
    EnvBuffers,
    ObsParams,
    OpponentParams,
    PhysicsBuffers,
    ResetParams,
    RewardParams,
    TerminationParams,
    TrackData,
    observation_stage_kernel,
    physics_solo_kernel,
    physics_stage_kernel,
    reset_envs_kernel,
    reset_to_kernel,
    transaction_stage_kernel,
)
from .opponents import MixedOpponentController, PolicyOpponent, make_opponent
from .utils import build_warp_track_data, load_track_state


class _VehicleStorage:
    def __init__(self, num_envs: int, device: torch.device, params: VehicleParams):
        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=torch.float32)

        self.tensor = {
            "x": zeros(num_envs),
            "y": zeros(num_envs),
            "yaw": zeros(num_envs),
            "vx": zeros(num_envs),
            "vy": zeros(num_envs),
            "yaw_rate": zeros(num_envs),
            "steer": zeros(num_envs),
            "effort_state": zeros(num_envs),
            "applied_effort": zeros(num_envs),
            "ax": zeros(num_envs),
            "ay": zeros(num_envs),
            "omega": zeros(num_envs, 4),
            "slip_ratio": zeros(num_envs, 4),
            "slip_angle": zeros(num_envs, 4),
            "load_ratio": torch.ones(
                num_envs, 4, device=device, dtype=torch.float32
            ),
            "fx_lag": zeros(num_envs, 4),
            "fy_lag": zeros(num_envs, 4),
            "mass": torch.full(
                (num_envs,), params.mass, device=device, dtype=torch.float32
            ),
            "mu": torch.full(
                (num_envs,), params.tire_mu, device=device, dtype=torch.float32
            ),
            "drive_scale": torch.ones(
                num_envs, device=device, dtype=torch.float32
            ),
            "steer_bias": zeros(num_envs),
        }
        self.buffers = VehicleBuffers()
        vector_fields = {
            "omega",
            "slip_ratio",
            "slip_angle",
            "load_ratio",
            "fx_lag",
            "fy_lag",
        }
        for name, tensor in self.tensor.items():
            if name in vector_fields:
                setattr(
                    self.buffers, name, wp.from_torch(tensor, dtype=wp.vec4f)
                )
            else:
                setattr(self.buffers, name, wp.from_torch(tensor))


class _EnvironmentStorage:
    _FLOAT_FIELDS = (
        "reward",
        "reward_progress",
        "reward_lateral",
        "reward_oob",
        "reward_slip",
        "reward_smoothness",
        "reward_passing",
        "reward_collision",
        "reward_rear_end",
        "reward_overtake",
        "prev_s",
        "prev_opponent_s",
        "opponent_target_speed",
        "opponent_lateral_offset",
        "opponent_speed_cap",
        "observation_noise",
        "terminal_x",
        "terminal_y",
        "terminal_s",
        "metric_progress",
        "metric_oob",
        "metric_boundary",
        "metric_lateral",
        "metric_speed",
        "metric_opponent_speed",
        "metric_nonfinite",
        "contact_closing_speed",
        "lap_cross",
    )
    _INT_FIELDS = (
        "done_flags",
        "episode_step",
        "episode_id",
        "lap_count",
        "ego_segment",
        "opponent_segment",
        "prev_off_track",
        "prev_opponent_ahead",
        "prev_opponent_in_window",
        "oob_streak",
        "stopped_streak",
        "action_head",
        "action_latency",
        "observation_latency",
        "opponent_mode",
        "contact",
        "valid",
    )
    _BOOL_FIELDS = (
        "done",
        "term_timeout",
        "term_oob",
        "term_stopped",
        "term_invalid",
        "term_collision",
    )

    def __init__(self, num_envs: int, device: torch.device):
        self.tensor: dict[str, torch.Tensor] = {}
        for name in self._FLOAT_FIELDS:
            self.tensor[name] = torch.zeros(
                num_envs, device=device, dtype=torch.float32
            )
        for name in self._INT_FIELDS:
            self.tensor[name] = torch.zeros(
                num_envs, device=device, dtype=torch.int32
            )
        for name in self._BOOL_FIELDS:
            self.tensor[name] = torch.zeros(
                num_envs, device=device, dtype=torch.bool
            )
        self.tensor["last_action"] = torch.zeros(
            num_envs, 2, device=device, dtype=torch.float32
        )
        self.tensor["opponent_last_action"] = torch.zeros(
            num_envs, 2, device=device, dtype=torch.float32
        )
        self.tensor["current_action"] = torch.zeros(
            num_envs, 2, device=device, dtype=torch.float32
        )
        self.tensor["current_opponent_action"] = torch.zeros(
            num_envs, 2, device=device, dtype=torch.float32
        )
        self.tensor["action_history"] = torch.zeros(
            num_envs,
            ACTION_HISTORY,
            2,
            device=device,
            dtype=torch.float32,
        )

        self.buffers = EnvBuffers()
        vector_fields = {
            "last_action",
            "opponent_last_action",
            "current_action",
            "current_opponent_action",
        }
        for name, tensor in self.tensor.items():
            if name in vector_fields:
                setattr(
                    self.buffers, name, wp.from_torch(tensor, dtype=wp.vec2f)
                )
            elif name == "action_history":
                setattr(
                    self.buffers,
                    name,
                    wp.from_torch(tensor, dtype=wp.vec2f),
                )
            else:
                setattr(self.buffers, name, wp.from_torch(tensor))
        self.physics_buffers = PhysicsBuffers()
        for name in (
            "opponent_segment",
            "episode_step",
            "action_history",
            "action_head",
            "action_latency",
            "opponent_mode",
            "opponent_target_speed",
            "opponent_lateral_offset",
            "opponent_speed_cap",
            "current_action",
            "current_opponent_action",
            "contact",
            "contact_closing_speed",
            "valid",
        ):
            tensor = self.tensor[name]
            if name in {
                "action_history",
                "current_action",
                "current_opponent_action",
            }:
                setattr(
                    self.physics_buffers,
                    name,
                    wp.from_torch(tensor, dtype=wp.vec2f),
                )
            else:
                setattr(self.physics_buffers, name, wp.from_torch(tensor))


class _TrackStorage:
    def __init__(self, host, device: str):
        self.array = {
            "point": wp.array(host.point, dtype=wp.vec2f, device=device),
            "tangent": wp.array(host.tangent, dtype=wp.vec2f, device=device),
            "normal": wp.array(host.normal, dtype=wp.vec2f, device=device),
            "segment_length": wp.array(
                host.segment_length, dtype=wp.float32, device=device
            ),
            "cumulative_length": wp.array(
                host.cumulative_length, dtype=wp.float32, device=device
            ),
            "width_left": wp.array(
                host.width_left, dtype=wp.float32, device=device
            ),
            "width_right": wp.array(
                host.width_right, dtype=wp.float32, device=device
            ),
            "nearest_segment_lut": wp.array(
                host.nearest_segment_lut, dtype=wp.int32, device=device
            ),
        }
        self.data = TrackData()
        for name, array in self.array.items():
            setattr(self.data, name, array)
        self.data.count = host.point.shape[0]
        self.data.length = host.length
        self.data.lut_width = host.lut_width
        self.data.lut_height = host.lut_height
        self.data.lut_origin = wp.vec2f(*host.lut_origin)
        self.data.lut_resolution = host.lut_resolution


class WarpF1tenthEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        show_viewer=False,
        enable_recording=False,
    ):
        del show_viewer, enable_recording
        if rt.tc_float != torch.float32:
            raise ValueError("Warp environment supports float32 only")
        self.num_envs = int(num_envs)
        self.num_actions = 2
        self.num_obs = OBS_DIM
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        strategy = env_cfg.get("opponent_strategy")
        self.has_opponent = strategy not in (None, "none")
        self.device = torch.device(rt.device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Warp environment supports CPU and CUDA devices")
        if int(obs_cfg["num_obs"]) != OBS_DIM:
            raise ValueError("Warp environment requires the fixed 390-d observation")
        self.dt = float(env_cfg.get("sim_dt", 0.005))
        self.control_interval = int(env_cfg.get("control_interval", 10))
        if self.dt != 0.005 or self.control_interval != 10:
            raise ValueError("Warp kernel requires sim_dt=0.005 and control_interval=10")
        self.control_dt = self.dt * self.control_interval
        self.step_launch_count = 3
        self.max_episode_steps = int(
            math.ceil(float(env_cfg["episode_length"]) / self.control_dt)
        )
        self.wp_device = (
            "cpu"
            if self.device.type == "cpu"
            else f"cuda:{self.device.index or 0}"
        )

        workspace = str(Path(__file__).resolve().parents[1])
        self.track_state = load_track_state(
            env_cfg.get("track"), workspace, self.device
        )
        self.centerline = self.track_state["centerline"]
        self.w_tr_left = self.track_state["w_tr_left"]
        self.w_tr_right = self.track_state["w_tr_right"]
        self._track_host = build_warp_track_data(self.track_state)
        self.track_length = self._track_host.length
        self._track = _TrackStorage(self._track_host, self.wp_device)

        self.vehicle_params = VehicleParams.from_config(env_cfg)
        self._sim_params = self.vehicle_params.to_warp(
            sim_dt=self.dt, control_dt=self.control_dt
        )
        self._ego = _VehicleStorage(
            self.num_envs, self.device, self.vehicle_params
        )
        self._opponent = _VehicleStorage(
            self.num_envs, self.device, self.vehicle_params
        )
        self._env = _EnvironmentStorage(self.num_envs, self.device)
        self._obs_params = self._build_obs_params()
        self._reward_params = self._build_reward_params()
        self._termination_params = self._build_termination_params()
        self._reset_params = self._build_reset_params()
        self._opponent_params = self._build_opponent_params()

        self._obs = [
            torch.zeros(
                self.num_envs, OBS_DIM, device=self.device, dtype=torch.float32
            )
            for _ in range(2)
        ]
        self._raw_obs = [
            torch.zeros_like(self._obs[0]),
            torch.zeros_like(self._obs[0]),
        ]
        self._opponent_obs = torch.zeros_like(self._obs[0])
        self._obs_wp = [wp.from_torch(tensor) for tensor in self._obs]
        self._raw_obs_wp = [
            wp.from_torch(tensor) for tensor in self._raw_obs
        ]
        self._opponent_obs_wp = wp.from_torch(self._opponent_obs)
        self._active_obs = 0
        self.obs_buf = self._obs[0]
        self.reward_buf = self._env.tensor["reward"]
        self.reset_buf = self._env.tensor["done"]
        self.episode_steps_buf = self._env.tensor["episode_step"]
        self.lap_count_buf = self._env.tensor["lap_count"]
        self.opponent_mode_buf = self._env.tensor["opponent_mode"]
        self.last_actions = self._env.tensor["last_action"]
        self.actions = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float32
        )
        self._policy_opponent_actions = torch.zeros_like(self.actions)
        self._reset_mask = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._reset_mask_wp = wp.from_torch(self._reset_mask)

        def _fixed_buffer(cols):
            shape = (self.num_envs, cols) if cols > 1 else (self.num_envs,)
            return torch.zeros(shape, device=self.device, dtype=torch.float32)

        self._fixed_ego_pose = _fixed_buffer(2)
        self._fixed_ego_yaw = _fixed_buffer(1)
        self._fixed_ego_speed = _fixed_buffer(1)
        self._fixed_opp_pose = _fixed_buffer(2)
        self._fixed_opp_yaw = _fixed_buffer(1)
        self._fixed_opp_speed = _fixed_buffer(1)

        self.opponent_ctrl = (
            make_opponent(env_cfg, obs_cfg, self.device)
            if self.has_opponent
            else None
        )
        self._policy_opponent = None
        if isinstance(self.opponent_ctrl, PolicyOpponent):
            self._policy_opponent = self.opponent_ctrl
        elif isinstance(self.opponent_ctrl, MixedOpponentController):
            self._policy_opponent = self.opponent_ctrl.policy
            self.opponent_ctrl.mode_buf = self._env.tensor["opponent_mode"]

        self.extras = self._build_extras()
        self.reset()

    def _build_obs_params(self):
        params = ObsParams()
        params.future_horizon_seconds = float(
            self.obs_cfg.get("future_track_horizon_s", 6.0)
        )
        params.future_minimum_lookahead = float(
            self.obs_cfg.get("future_track_min_lookahead_m", 5.0)
        )
        params.clip = float(self.obs_cfg.get("clip_obs", 50.0))
        params.opponent_ahead = float(
            self.obs_cfg.get("opp_obs_ahead_m", 40.0)
        )
        params.opponent_behind = float(
            self.obs_cfg.get("opp_obs_behind_m", 20.0)
        )
        params.zero_tyre_slip = int(
            self.obs_cfg.get("zero_tyre_slip_obs", False)
        )
        params.contact_margin = float(
            self.obs_cfg.get("contact_margin_m", 0.08)
        )
        return params

    def _build_reward_params(self):
        scales = self.reward_cfg.get("reward_scales", {})
        # Sophy reward rate is 10 Hz; convert per-step state/event terms to that
        # rate. At the 20 Hz control cadence this is 0.5.
        cadence = self.control_dt / 0.1
        params = RewardParams()
        progress_scale = float(scales.get("progress", 0.0))
        params.progress_forward = progress_scale
        params.progress_backward = progress_scale
        params.progress_max_lateral = float(
            self.reward_cfg.get("progress_max_lateral_m", 1.0)
        )
        params.lateral = float(
            self.reward_cfg.get("lateral_k", 0.5)
        ) * float(scales.get("lateral", 0.0))
        params.oob = (
            float(scales.get("oob_penalty", 0.0))
            * self.control_dt
            * (3.6 * 3.6)
        )
        params.oob_margin = float(
            self.reward_cfg.get("oob_margin_m", 0.2)
        )
        params.slip = float(scales.get("tyre_slip_penalty", 0.0)) * cadence
        params.slip_angle_weight = float(
            self.reward_cfg.get("slip_angle_weight", 1.0)
        )
        params.slip_ratio_deadzone = float(
            self.reward_cfg.get("slip_deadzone_ratio", 0.0)
        )
        params.slip_angle_deadzone = float(
            self.reward_cfg.get("slip_deadzone_angle", 0.0)
        )
        params.smoothness = float(scales.get("smoothness", 0.0))
        params.passing = float(scales.get("passing", 0.0))
        params.passing_ahead = float(
            self.reward_cfg.get("passing_gate_ahead_m", 40.0)
        )
        params.passing_behind = float(
            self.reward_cfg.get("passing_gate_behind_m", 20.0)
        )
        params.collision = float(scales.get("collision", 0.0)) * cadence
        params.rear_end = float(scales.get("rear_end", 0.0)) * cadence
        params.overtake = float(
            self.reward_cfg.get("overtake_bonus_k", 1.0)
        ) * float(scales.get("overtake", 0.0))
        params.overtake_gap = float(
            self.reward_cfg.get("overtake_gap_m", 5.0)
        )
        params.global_scale = float(
            self.reward_cfg.get("global_reward_scale", 1.0)
        )
        return params

    def _build_termination_params(self):
        params = TerminationParams()
        params.maximum_episode_steps = self.max_episode_steps
        params.maximum_oob_steps = int(
            self.env_cfg.get("term_oob_max_consecutive", 2)
        )
        params.maximum_stopped_steps = int(
            round(
                float(self.env_cfg.get("term_not_moving_time_s", 2.0))
                / self.control_dt
            )
        )
        params.speed_threshold = float(
            self.env_cfg.get("term_speed_threshold", 0.2)
        )
        params.minimum_progress = float(
            self.env_cfg.get("term_not_moving_min_ds", 1.0e-3)
        )
        params.maximum_heading_error = float(
            self.env_cfg.get("term_heading_error_rad", 3.0)
        )
        params.collision_speed = float(
            self.env_cfg.get("collision_term_speed_mps", 0.0)
        )
        params.terminate_on_collision = int(
            self.env_cfg.get("term_on_collision", True)
        )
        params.oob_margin = float(
            self.env_cfg.get("term_oob_margin_m", 0.15)
        )
        return params

    def _build_reset_params(self):
        dr = self.env_cfg.get("domain_randomization") or {}
        enabled = bool(dr.get("enabled", False))

        def interval(name, nominal):
            if not enabled:
                return float(nominal[0]), float(nominal[1])
            values = dr.get(name, nominal)
            return float(values[0]), float(values[1])

        params = ResetParams()
        params.seed = int(self.env_cfg.get("seed", 0))
        params.has_opponent = int(self.has_opponent)
        params.spawn_margin = float(
            self.env_cfg.get("reset_spawn_margin_m", 0.2)
        )
        params.speed_min = float(
            self.env_cfg.get("reset_speed_min_mps", 1.0)
        )
        params.speed_max = float(
            self.env_cfg.get("reset_speed_max_mps", 4.0)
        )
        params.opponent_gap_min = float(
            self.env_cfg.get("opponent_spawn_gap_min_m", 3.0)
        )
        params.opponent_gap_max = float(
            self.env_cfg.get("opponent_spawn_gap_max_m", 20.0)
        )
        params.opponent_behind_probability = float(
            self.env_cfg.get("opponent_spawn_behind_prob", 0.3)
        )
        params.opponent_speed_min = float(
            self.env_cfg.get("opponent_reset_speed_min_mps", 1.0)
        )
        params.opponent_speed_max = float(
            self.env_cfg.get("opponent_reset_speed_max_mps", 4.0)
        )
        minimum, maximum = interval(
            "vehicle_mass_range",
            (self.vehicle_params.mass, self.vehicle_params.mass),
        )
        params.mass_min = minimum
        params.mass_max = maximum
        minimum, maximum = interval(
            "tire_friction_range",
            (self.vehicle_params.tire_mu, self.vehicle_params.tire_mu),
        )
        params.friction_min = minimum
        params.friction_max = maximum
        minimum, maximum = interval(
            "drive_scale_range", (1.0, 1.0)
        )
        params.drive_scale_min = minimum
        params.drive_scale_max = maximum
        minimum, maximum = interval(
            "steer_bias_range", (0.0, 0.0)
        )
        params.steer_bias_min = minimum
        params.steer_bias_max = maximum
        action_latency = (
            dr.get("action_latency_steps_range", (0, 0))
            if enabled
            else (int(self.env_cfg.get("simulate_action_latency", False)),) * 2
        )
        observation_latency = (
            dr.get("obs_latency_steps_range", (0, 0))
            if enabled
            else (0, 0)
        )
        if int(observation_latency[1]) > 1:
            raise ValueError("Warp observation latency supports at most one tick")
        params.action_latency_min = int(action_latency[0])
        params.action_latency_max = int(action_latency[1])
        params.observation_latency_min = int(observation_latency[0])
        params.observation_latency_max = int(observation_latency[1])
        minimum, maximum = interval(
            "obs_noise_std_range", (0.0, 0.0)
        )
        params.observation_noise_min = minimum
        params.observation_noise_max = maximum
        params.spawn_yaw_jitter = float(
            self.env_cfg.get("reset_spawn_yaw_jitter_rad", 0.0)
        )
        params.opponent_lateral_spawn = int(
            bool(self.env_cfg.get("opponent_spawn_lateral_independent", False))
        )
        return params

    def _build_opponent_params(self):
        strategy = self.env_cfg.get("opponent_strategy")
        params = OpponentParams()
        params.strategy = {
            None: 0,
            "none": 0,
            "scripted": 0,
            "policy": 1,
            "mixed": 2,
        }[strategy]
        mix = self.env_cfg.get("opponent_mix") or {}
        scripted_weight = float(mix.get("scripted_weight", 0.25))
        policy_weight = float(mix.get("policy_weight", 0.75))
        total_weight = scripted_weight + policy_weight
        if total_weight <= 0.0:
            raise ValueError("opponent_mix weights must have a positive sum")
        params.policy_probability = policy_weight / total_weight
        params.kp_lateral = float(
            self.env_cfg.get("opponent_kp_ey", 1.0)
        )
        params.kp_heading = float(
            self.env_cfg.get("opponent_kh_heading", 1.0)
        )
        params.kp_speed = float(
            self.env_cfg.get("opponent_kp_speed", 1.0)
        )
        speed_range = self.env_cfg.get("opponent_target_speed_range")
        if speed_range:
            params.target_speed_min = float(speed_range[0])
            params.target_speed_max = float(speed_range[1])
        else:
            target = float(self.env_cfg.get("opponent_target_speed", 2.5))
            params.target_speed_min = target
            params.target_speed_max = target
        params.lateral_offset_max = float(
            self.env_cfg.get("opponent_lateral_offset_m", 0.0)
        )
        params.policy_speed_cap_prob = float(
            mix.get("policy_speed_cap_prob", 0.0)
        )
        cap_range = mix.get("policy_speed_cap_range", (2.5, 5.0))
        params.policy_speed_cap_lo = float(cap_range[0])
        params.policy_speed_cap_hi = float(cap_range[1])
        params.car_length = float(self.env_cfg.get("car_length", 0.568))
        params.car_width = float(self.env_cfg.get("car_width", 0.296))
        params.restitution = 0.1
        return params

    def _build_extras(self):
        tensors = self._env.tensor
        return {
            "observations": {"critic": self.obs_buf},
            "rewards": {
                "total": tensors["reward"],
                "terms": {
                    "progress": tensors["reward_progress"],
                    "lateral": tensors["reward_lateral"],
                    "oob_penalty": tensors["reward_oob"],
                    "tyre_slip_penalty": tensors["reward_slip"],
                    "smoothness": tensors["reward_smoothness"],
                    "passing": tensors["reward_passing"],
                    "collision": tensors["reward_collision"],
                    "rear_end": tensors["reward_rear_end"],
                    "overtake": tensors["reward_overtake"],
                },
            },
            "termination": {
                "time_out": tensors["term_timeout"],
                "out_of_bounds": tensors["term_oob"],
                "not_moving": tensors["term_stopped"],
                "invalid_state": tensors["term_invalid"],
                "collision": tensors["term_collision"],
            },
            "time_outs": tensors["term_timeout"],
            "metrics": {
                "progress_ds": tensors["metric_progress"],
                "s": tensors["prev_s"],
                "oob_mask": tensors["metric_oob"],
                "boundary_dist": tensors["metric_boundary"],
                "lateral_error": tensors["metric_lateral"],
                "speed_xy": tensors["metric_speed"],
                "episode_steps": tensors["episode_step"],
                "lap_count": tensors["lap_count"],
                "laps_completed": tensors["lap_cross"],
                "opp_speed": tensors["metric_opponent_speed"],
                "opponent_s": tensors["prev_opponent_s"],
                "nonfinite_obs_envs": tensors["metric_nonfinite"],
                "nonfinite_reward_envs": tensors["metric_nonfinite"],
                "nonfinite_state_envs": tensors["metric_nonfinite"],
                "dr/tire_friction": self._ego.tensor["mu"],
                "dr/vehicle_mass": self._ego.tensor["mass"],
                "dr/drive_scale": self._ego.tensor["drive_scale"],
                "dr/steer_bias": self._ego.tensor["steer_bias"],
                "dr/action_latency_steps": tensors["action_latency"],
                "dr/obs_latency_steps": tensors["observation_latency"],
                "dr/obs_noise_std": tensors["observation_noise"],
            },
        }

    def _stream(self):
        if self.device.type == "cuda":
            return wp.stream_from_torch(torch.cuda.current_stream(self.device))
        return None

    def _policy_actions(self):
        if self._policy_opponent is None:
            return self._policy_opponent_actions
        return self._policy_opponent.act_observation(self._opponent_obs)

    def reset(self, envs_idx=None, *, seed: int | None = None):
        if seed is not None:
            self._reset_params.seed = int(seed)
            self._env.tensor["episode_id"].zero_()
        if envs_idx is None:
            self._reset_mask.fill_(True)
        elif isinstance(envs_idx, (list, tuple, np.ndarray)):
            self._reset_mask.zero_()
            self._reset_mask[envs_idx] = True
        elif envs_idx.dtype == torch.bool:
            self._reset_mask.copy_(envs_idx.to(self.device))
        else:
            self._reset_mask.zero_()
            self._reset_mask[envs_idx] = True
        wp.launch(
            reset_envs_kernel,
            dim=self.num_envs,
            inputs=[
                self._reset_mask_wp,
                self._ego.buffers,
                self._opponent.buffers,
                self._env.buffers,
                self._track.data,
                self._sim_params,
                self._obs_params,
                self._reset_params,
                self._opponent_params,
                self._raw_obs_wp[0],
                self._raw_obs_wp[1],
                self._obs_wp[0],
                self._obs_wp[1],
                self._opponent_obs_wp,
            ],
            device=self.wp_device,
            stream=self._stream(),
        )
        self.obs_buf = self._obs[self._active_obs]
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def reset_to(
        self,
        pose,
        yaw,
        speed,
        *,
        opponent_pose=None,
        opponent_yaw=None,
        opponent_speed=None,
        envs_idx=None,
        seed=None,
    ):
        """Deterministically place cars at explicit world poses (bypass random spawn).

        Seeds all per-env bookkeeping (domain randomization, latencies, episode
        counters) through the normal ``reset`` first, then overwrites the vehicle
        pose/heading/speed and reprojects the Frenet baseline + observations. Used
        for eval / telemetry reproduction and collision fixtures; identical inputs
        yield identical trajectories.
        """
        self.reset(envs_idx, seed=seed)

        def _fill(buffer, values, cols):
            tensor = torch.as_tensor(
                values, device=self.device, dtype=torch.float32
            )
            shape = (self.num_envs, cols) if cols > 1 else (self.num_envs,)
            tensor = tensor.reshape(-1, cols) if cols > 1 else tensor.reshape(-1)
            if tensor.shape[0] == 1:
                tensor = tensor.expand(*shape).contiguous()
            buffer.copy_(tensor)

        _fill(self._fixed_ego_pose, pose, 2)
        _fill(self._fixed_ego_yaw, yaw, 1)
        _fill(self._fixed_ego_speed, speed, 1)
        if opponent_pose is None:
            self._fixed_opp_pose.copy_(self._fixed_ego_pose)
            self._fixed_opp_yaw.copy_(self._fixed_ego_yaw)
            self._fixed_opp_speed.zero_()
        else:
            _fill(self._fixed_opp_pose, opponent_pose, 2)
            _fill(
                self._fixed_opp_yaw,
                self._fixed_ego_yaw if opponent_yaw is None else opponent_yaw,
                1,
            )
            _fill(
                self._fixed_opp_speed,
                0.0 if opponent_speed is None else opponent_speed,
                1,
            )

        wp.launch(
            reset_to_kernel,
            dim=self.num_envs,
            inputs=[
                self._reset_mask_wp,
                wp.from_torch(self._fixed_ego_pose, dtype=wp.vec2f),
                wp.from_torch(self._fixed_ego_yaw),
                wp.from_torch(self._fixed_ego_speed),
                wp.from_torch(self._fixed_opp_pose, dtype=wp.vec2f),
                wp.from_torch(self._fixed_opp_yaw),
                wp.from_torch(self._fixed_opp_speed),
                self._ego.buffers,
                self._opponent.buffers,
                self._env.buffers,
                self._track.data,
                self._sim_params,
                self._obs_params,
                self._reset_params,
                self._opponent_params,
                self._raw_obs_wp[0],
                self._raw_obs_wp[1],
                self._obs_wp[0],
                self._obs_wp[1],
                self._opponent_obs_wp,
            ],
            device=self.wp_device,
            stream=self._stream(),
        )
        self.obs_buf = self._obs[self._active_obs]
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def step(self, actions, n_steps=10):
        if int(n_steps) != self.control_interval:
            raise ValueError("Warp environment requires n_steps=control_interval=10")
        self.actions = actions.to(
            device=self.device, dtype=torch.float32
        ).contiguous()
        if self.actions.shape != (self.num_envs, 2):
            raise ValueError(
                f"actions shape={tuple(self.actions.shape)}; "
                f"expected ({self.num_envs}, 2)"
            )
        opponent_actions = self._policy_actions()
        inactive = 1 - self._active_obs
        if self.has_opponent:
            physics_kernel = physics_stage_kernel
            physics_inputs = [
                wp.from_torch(self.actions, dtype=wp.vec2f),
                wp.from_torch(
                    opponent_actions, dtype=wp.vec2f
                ),
                self._ego.buffers,
                self._opponent.buffers,
                self._env.physics_buffers,
                self._track.data,
                self._sim_params,
                self._reset_params,
                self._opponent_params,
                self.num_envs,
                self.control_interval,
            ]
        else:
            physics_kernel = physics_solo_kernel
            physics_inputs = [
                wp.from_torch(self.actions, dtype=wp.vec2f),
                self._ego.buffers,
                self._env.physics_buffers,
                self._sim_params,
                self.control_interval,
            ]
        wp.launch(
            physics_kernel,
            dim=2 * self.num_envs if self.has_opponent else self.num_envs,
            inputs=physics_inputs,
            device=self.wp_device,
            stream=self._stream(),
        )
        wp.launch(
            transaction_stage_kernel,
            dim=self.num_envs,
            inputs=[
                self._ego.buffers,
                self._opponent.buffers,
                self._env.buffers,
                self._track.data,
                self._sim_params,
                self._reward_params,
                self._termination_params,
                self._reset_params,
                self._opponent_params,
            ],
            device=self.wp_device,
            stream=self._stream(),
        )
        wp.launch(
            observation_stage_kernel,
            dim=self.num_envs,
            inputs=[
                self._ego.buffers,
                self._opponent.buffers,
                self._env.buffers,
                self._track.data,
                self._obs_params,
                self._reset_params,
                self._raw_obs_wp[self._active_obs],
                self._raw_obs_wp[inactive],
                self._obs_wp[inactive],
                self._opponent_obs_wp,
            ],
            device=self.wp_device,
            stream=self._stream(),
        )
        self._active_obs = inactive
        self.obs_buf = self._obs[inactive]
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.reward_buf, self.reset_buf, self.extras

    def refresh_opponent_policy(self, state_dict, obs_mean, obs_var):
        if self._policy_opponent is None:
            raise RuntimeError("Environment has no policy opponent")
        self._policy_opponent.load_snapshot(state_dict, obs_mean, obs_var)

    def read_state(self):
        ego = self._read_vehicle(self._ego.tensor)
        if self.has_opponent:
            opponent = self._read_vehicle(self._opponent.tensor)
            ego.update(
                {
                    "opp_base_pos": opponent["base_pos"],
                    "opp_base_quat": opponent["base_quat"],
                    "opp_vel_world": opponent["base_vel_world"],
                    "opp_ang_world": opponent["base_ang_vel"],
                }
            )
        return ego

    def read_wheel_state(self, which="ego"):
        storage = self._opponent if which == "opp" else self._ego
        return {
            "dof_vel": storage.tensor["omega"],
            "tyre_slip": torch.cat(
                (
                    storage.tensor["slip_ratio"],
                    storage.tensor["slip_angle"],
                ),
                dim=1,
            ),
            "tyre_load": storage.tensor["load_ratio"],
        }

    def _read_vehicle(self, state):
        yaw = state["yaw"]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        zero = torch.zeros_like(yaw)
        return {
            "base_pos": torch.stack((state["x"], state["y"], zero), dim=1),
            "base_quat": torch.stack(
                (torch.cos(0.5 * yaw), zero, zero, torch.sin(0.5 * yaw)),
                dim=1,
            ),
            "base_vel_world": torch.stack(
                (
                    cosine * state["vx"] - sine * state["vy"],
                    sine * state["vx"] + cosine * state["vy"],
                    zero,
                ),
                dim=1,
            ),
            "base_lin_vel": torch.stack(
                (state["vx"], state["vy"], zero), dim=1
            ),
            "base_ang_vel": torch.stack(
                (zero, zero, state["yaw_rate"]), dim=1
            ),
            "base_lin_acc": torch.stack(
                (state["ax"], state["ay"], zero), dim=1
            ),
        }

    @property
    def base_pos(self):
        return self.read_state()["base_pos"]

    @property
    def base_quat(self):
        return self.read_state()["base_quat"]

    @property
    def base_lin_vel(self):
        return self.read_state()["base_lin_vel"]

    @property
    def base_ang_vel(self):
        return self.read_state()["base_ang_vel"]

    def close(self):
        return None

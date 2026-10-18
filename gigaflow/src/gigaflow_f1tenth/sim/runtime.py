"""Production Warp N-agent simulator implementing kernels.Simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp

from gigaflow_f1tenth.buffers import (
    DEFAULT_AGENT_STATE_DIM,
    pack_compact_agent_state,
    unpack_compact_agent_state,
)
from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.kernels import SimulatorState, WorldSlotLayout, world_slot_layout
from gigaflow_f1tenth.rewards import (
    CONDITION_DIM,
    CONDITION_FIELD_NAMES,
    PrivateStyle,
    RewardTerms,
    deployment_style,
    sample_private_styles,
    styles_to_condition_batch,
)
from gigaflow_f1tenth.sim.contact import (
    ContactParams,
    apply_contact_gather_kernel,
    build_world_pairs_kernel,
    clear_contact_accumulators,
    resolve_pairs_jacobi_kernel,
)
from gigaflow_f1tenth.sim.geometry import (
    build_sim_geometry_from_atlas,
    make_synthetic_oval_atlas,
)
from gigaflow_f1tenth.sim.reward_kernels import (
    pack_opponent_progress_kernel,
    racing_reward_kernel,
    slot_reward_geom_kernel,
)
from gigaflow_f1tenth.sim.layout_local import LIDAR_DIM
from gigaflow_f1tenth.sim.sensors import (
    default_sensor_params,
    lidar_and_proprio_kernel,
    lidar_beam_parallel_kernel,
)
from gigaflow_f1tenth.sim.spawn import (
    apply_static_pins,
    assign_world_tracks,
    horizon_for_track,
    mix_seed,
    place_static_opponents,
    sample_active_counts,
    sample_lidar_corruption,
    spawn_agent_pose,
)
from gigaflow_f1tenth.sim.spawn_kernels import (
    async_respawn_kernel,
    style_scatter_kernel,
)
from gigaflow_f1tenth.sim.stages import (
    MAX_PHYSICS_SUBSTEPS,
    deactivate_terminal_rows_kernel,
    physics_control_kernel,
    progress_wall_kernel,
    push_command_history_kernel,
    terminal_mask_kernel,
)
from gigaflow_f1tenth.sim.state import SimulatorBuffers
from gigaflow_f1tenth.sim.vehicle import VehicleParams
from gigaflow_f1tenth.tracks import PackedTrackAtlasView

# Style-pool size for device-side async private-style resampling (setup-time CPU).
_STYLE_POOL_SIZE = 256

# Integration resolved: normals/segment_length derived in sim.geometry;
# conditioned rewards, compact-state packing, and buffer wiring are live.
INTERFACE_GAPS: tuple[str, ...] = ()


class WarpNAgentSimulator:
    """Fixed-capacity multi-world N-car Warp simulator (one production path)."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        atlas: PackedTrackAtlasView,
        device: str,
        *,
        sync_no_respawn: bool | None = None,
    ):
        wp.init()
        self.cfg = cfg
        self.layout = world_slot_layout(cfg)
        self.device = device
        # Training defaults to async respawn. Evaluation suites pass
        # sync_no_respawn=True explicitly; do not inherit evaluation.sync_no_respawn
        # into the learner simulator (that permanently deactivates terminal agents).
        self.sync_no_respawn = False if sync_no_respawn is None else bool(sync_no_respawn)
        self.vehicle_params = VehicleParams()
        self.sim_params = self.vehicle_params.to_warp(
            sim_dt=cfg.agents.sim_dt,
            control_dt=1.0 / cfg.agents.control_hz,
        )
        self.geom = build_sim_geometry_from_atlas(atlas)
        self.buffers = SimulatorBuffers(cfg, device)
        self.buffers.seed_defaults(self.vehicle_params)
        self.sensor_params = default_sensor_params(
            cfg.agents.car_length_m,
            cfg.agents.car_width_m,
            cfg.worlds.max_agents_per_world,
        )
        self.contact_params = ContactParams()
        self.contact_params.car_length = float(cfg.agents.car_length_m)
        self.contact_params.car_width = float(cfg.agents.car_width_m)
        self.contact_params.restitution = 0.1
        self.contact_params.max_agents_per_world = int(cfg.worlds.max_agents_per_world)
        self.contact_params.max_pairs = int(self.buffers.pair_i.shape[0])

        self._upload_geometry()
        self._spawn_stats = {
            "requested": 0,
            "realized": 0,
            "rejects": 0,
        }
        n = self.layout.num_slots
        n_others = max(int(cfg.worlds.max_agents_per_world) - 1, 0)
        d = self.buffers.wp_device
        self._n_others = n_others
        self._wp_track_length_slot = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_half_width = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_tangent_yaw = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_opp_s = wp.zeros((n, max(n_others, 1)), dtype=wp.float32, device=d)
        self._wp_opp_ds = wp.zeros((n, max(n_others, 1)), dtype=wp.float32, device=d)
        self._wp_opp_act = wp.zeros((n, max(n_others, 1)), dtype=wp.uint8, device=d)
        self._wp_gate = wp.zeros((n, max(n_others, 1)), dtype=wp.uint8, device=d)
        self._wp_prev_gate = wp.zeros((n, max(n_others, 1)), dtype=wp.uint8, device=d)
        self._wp_r_progress = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_r_collision = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_r_boundary = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_r_center = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_r_passing = wp.zeros(n, dtype=wp.float32, device=d)
        self._wp_styles = wp.zeros((n, CONDITION_DIM), dtype=wp.float32, device=d)
        self._torch_styles = wp.to_torch(self._wp_styles)
        self._wp_style_pool = wp.zeros(
            (_STYLE_POOL_SIZE, CONDITION_DIM), dtype=wp.float32, device=d
        )
        self._torch_style_pool = wp.to_torch(self._wp_style_pool)
        self._style_alpha_names = (
            "alpha_collision",
            "alpha_boundary",
            "alpha_l_center",
            "alpha_center_bias",
            "alpha_passing",
        )
        self._wp_alpha = {
            name: wp.zeros(n, dtype=wp.float32, device=d)
            for name in self._style_alpha_names
        }
        self._torch_alpha = {
            name: wp.to_torch(arr) for name, arr in self._wp_alpha.items()
        }
        self.styles: list[PrivateStyle] = [
            deployment_style(cfg.evaluation.conservative_deployment_style)
            for _ in range(n)
        ]
        self._style_tensors: dict[str, torch.Tensor] | None = None
        self._torch_prev_gate = wp.to_torch(self._wp_prev_gate)
        self._passing_gate = self._torch_prev_gate
        self._overflow_torch = wp.to_torch(self.buffers.broadphase_overflow)
        self.last_reward_terms: dict[str, torch.Tensor] | None = None
        if str(device).startswith("cuda") and self.buffers.device_str != "cuda":
            raise RuntimeError(
                "CUDA device requested but simulator buffers fell back to CPU; "
                "refusing host-resident runtime path"
            )
        self._control_interval = int(cfg.agents.control_interval)
        if self._control_interval > MAX_PHYSICS_SUBSTEPS:
            raise ValueError(
                f"control_interval={self._control_interval} exceeds "
                f"MAX_PHYSICS_SUBSTEPS={MAX_PHYSICS_SUBSTEPS}"
            )
        n_slots = self.layout.num_slots
        self._action_buf = torch.zeros(
            (n_slots, 2),
            device=self.buffers.torch_device,
            dtype=torch.float32,
        )
        self._wp_actions = wp.from_torch(self._action_buf, dtype=wp.vec2f)
        # Optional static-shape Warp CUDA graphs. Physics graph stays on for CUDA.
        # Sensor-graph capture was benchmarked and regressed a single fast launch, so
        # it stays off unless explicitly enabled for experiments.
        self._cuda_graph = None
        self._sensor_cuda_graph = None
        self._cuda_graph_enabled = bool(str(device).startswith("cuda"))
        self._sensor_cuda_graph_enabled = False
        self._cuda_graph_warmup_left = 2
        self._sensor_cuda_graph_warmup_left = 2
        self._cuda_graph_status = "disabled_cpu" if not self._cuda_graph_enabled else "pending"
        self._sensor_cuda_graph_status = "disabled"
        # Production path: beam-parallel LiDAR. Serial kernel remains for parity tests.
        self._use_beam_parallel_lidar = True

        # Static-opponent pin state (populated by reset_all when configured;
        # None keeps step() a no-op for simulators without static opponents).
        self._static_mask: torch.Tensor | None = None
        self._static_pin: tuple[torch.Tensor, ...] | None = None

        self._rebuild_style_pool(cfg.seed + 404)
        self.reset_all(cfg.seed)
        self.apply_styles(self.styles)

    def _upload_geometry(self) -> None:
        d = self.buffers.wp_device
        g = self.geom
        self.wp_point = wp.array(g.centerline_xy, dtype=wp.vec2f, device=d)
        self.wp_tangent = wp.array(g.tangents_xy, dtype=wp.vec2f, device=d)
        self.wp_normal = wp.array(g.normals_xy, dtype=wp.vec2f, device=d)
        self.wp_seg_len = wp.array(g.segment_length, dtype=wp.float32, device=d)
        self.wp_cum = wp.array(g.cum_length, dtype=wp.float32, device=d)
        # widths_rl is [right, left]
        self.wp_width_right = wp.array(g.widths_rl[:, 0], dtype=wp.float32, device=d)
        self.wp_width_left = wp.array(g.widths_rl[:, 1], dtype=wp.float32, device=d)
        self.wp_offsets = wp.array(g.offsets, dtype=wp.int32, device=d)
        self.wp_track_length = wp.array(g.track_length, dtype=wp.float32, device=d)
        self.wp_edt = wp.array(g.edt_distance, dtype=wp.float32, device=d)
        self.wp_edt_offsets = wp.array(g.edt_offsets, dtype=wp.int32, device=d)
        self.wp_edt_width = wp.array(g.edt_width, dtype=wp.int32, device=d)
        self.wp_edt_height = wp.array(g.edt_height, dtype=wp.int32, device=d)
        self.wp_edt_origin = wp.array(g.edt_origin, dtype=wp.vec2f, device=d)
        self.wp_edt_resolution = wp.array(g.edt_resolution, dtype=wp.float32, device=d)
        # Device-resident tables for reward geometry gathers (no per-slot .item()).
        td = self.buffers.torch_device
        self._torch_track_length = torch.as_tensor(
            g.track_length, device=td, dtype=torch.float32
        )
        self._torch_offsets = torch.as_tensor(g.offsets, device=td, dtype=torch.int64)
        self._torch_tangents = torch.as_tensor(
            g.tangents_xy, device=td, dtype=torch.float32
        )
        self._torch_widths = torch.as_tensor(g.widths_rl, device=td, dtype=torch.float32)

    def interface_gaps(self) -> tuple[str, ...]:
        return INTERFACE_GAPS

    @property
    def state_dim(self) -> int:
        return DEFAULT_AGENT_STATE_DIM

    def pack_state(self) -> torch.Tensor:
        return pack_compact_agent_state(self.buffers.torch_arrays)

    def restore_state(self, state: torch.Tensor) -> None:
        unpack_compact_agent_state(state, self.buffers.torch_arrays)

    def _rebuild_style_pool(self, seed: int) -> None:
        """Offline CPU sample of private styles → device pool for async scatter."""
        cfg = self.cfg
        rng = np.random.default_rng(int(seed))
        randomize = bool(
            cfg.reward_conditioning.enabled and cfg.ablations.reward_randomization
        )
        if randomize and bool(cfg.reward_conditioning.randomize_styles):
            pool = sample_private_styles(cfg, _STYLE_POOL_SIZE, rng=rng)
        else:
            style = deployment_style(cfg.evaluation.conservative_deployment_style)
            pool = [style for _ in range(_STYLE_POOL_SIZE)]
        mat = styles_to_condition_batch(pool, normalize=False)
        self._torch_style_pool.copy_(
            torch.as_tensor(mat, device=self.buffers.torch_device, dtype=torch.float32)
        )

    def condition_tensor(self) -> torch.Tensor:
        """Normalized private-condition side channel [S, 10] (device-resident)."""
        from gigaflow_f1tenth.rewards import normalize_condition_vector

        return normalize_condition_vector(self._torch_styles)

    def raw_styles(self) -> torch.Tensor:
        """Authoritative unnormalized per-slot style rows [S, 10] on device.

        Device-side respawn scatter updates these; ``self.styles`` is only the
        host-side metadata snapshot and goes stale after the first respawn.
        """
        return self._torch_styles

    def apply_styles(self, styles: list[PrivateStyle]) -> None:
        if len(styles) != self.layout.num_slots:
            raise ValueError(
                f"styles len {len(styles)} != num_slots {self.layout.num_slots}"
            )
        self.styles = list(styles)
        device = self.buffers.torch_device
        mat = torch.as_tensor(
            styles_to_condition_batch(styles, normalize=False),
            device=device,
            dtype=torch.float32,
        )
        self._torch_styles.copy_(mat)
        self._style_tensors = {
            name: self._torch_alpha[name]
            for name in self._style_alpha_names
        }
        for j, name in enumerate(CONDITION_FIELD_NAMES):
            if name in self._torch_alpha:
                self._torch_alpha[name].copy_(mat[:, j])
        t = self.buffers.torch_arrays
        t.drive_scale.copy_(mat[:, CONDITION_FIELD_NAMES.index("drive_scale")])
        t.mass.copy_(mat[:, CONDITION_FIELD_NAMES.index("mass_kg")])
        t.steer_scale.copy_(mat[:, CONDITION_FIELD_NAMES.index("steer_scale")])
        t.accel_scale.copy_(mat[:, CONDITION_FIELD_NAMES.index("accel_scale")])
        t.vmax_scale.copy_(mat[:, CONDITION_FIELD_NAMES.index("vmax_scale")])

    def resample_styles_for_mask(
        self, mask: torch.Tensor, seed: int
    ) -> list[PrivateStyle]:
        """Device-side style scatter for reset rows; host list kept for metadata."""
        n = self.layout.num_slots
        t = self.buffers.torch_arrays
        m = torch.as_tensor(mask, device=self.buffers.torch_device).view(-1)
        saved = t.reset_mask.clone()
        t.reset_mask.copy_(m.to(dtype=t.reset_mask.dtype))
        wp.launch(
            style_scatter_kernel,
            dim=n,
            inputs=[
                self.buffers.reset_mask,
                self._wp_style_pool,
                _STYLE_POOL_SIZE,
                int(seed),
                self.buffers.world_id,
                self.buffers.slot_id,
                self.buffers.episode_id,
                self._wp_styles,
                self.buffers.drive_scale,
                self.buffers.mass,
                self.buffers.steer_scale,
                self.buffers.accel_scale,
                self.buffers.vmax_scale,
            ],
            device=self.buffers.wp_device,
        )
        t.reset_mask.copy_(saved)
        mat = self._torch_styles
        for j, name in enumerate(CONDITION_FIELD_NAMES):
            if name in self._torch_alpha:
                self._torch_alpha[name].copy_(mat[:, j])
        self._style_tensors = {
            name: self._torch_alpha[name] for name in self._style_alpha_names
        }
        return self.styles

    def _launch_reward_kernels(self) -> RewardTerms:
        cfg = self.cfg
        n = self.layout.num_slots
        n_others = self._n_others
        vehicles = self.buffers.vehicle_buffers()
        control_dt = 1.0 / float(cfg.agents.control_hz)
        wp.launch(
            slot_reward_geom_kernel,
            dim=n,
            inputs=[
                self.buffers.active,
                self.buffers.track_id,
                self.buffers.frenet_segment,
                self.wp_offsets,
                self.wp_track_length,
                self.wp_tangent,
                self.wp_width_right,
                self.wp_width_left,
                self._wp_track_length_slot,
                self._wp_half_width,
                self._wp_tangent_yaw,
            ],
            device=self.buffers.wp_device,
        )
        if n_others > 0:
            wp.launch(
                pack_opponent_progress_kernel,
                dim=n,
                inputs=[
                    self.buffers.active,
                    self.buffers.world_id,
                    self.buffers.frenet_s,
                    self.buffers.rewards,
                    int(cfg.worlds.max_agents_per_world),
                    int(n_others),
                    self._wp_opp_s,
                    self._wp_opp_ds,
                    self._wp_opp_act,
                ],
                device=self.buffers.wp_device,
            )
        wp.launch(
            racing_reward_kernel,
            dim=n,
            inputs=[
                vehicles,
                self.buffers.active,
                self.buffers.rewards,
                self.buffers.wall_contact,
                self.buffers.contact,
                self.buffers.frenet_s,
                self.buffers.frenet_ey,
                self.buffers.reset_mask,
                self._wp_track_length_slot,
                self._wp_half_width,
                self._wp_tangent_yaw,
                self._wp_alpha["alpha_collision"],
                self._wp_alpha["alpha_boundary"],
                self._wp_alpha["alpha_l_center"],
                self._wp_alpha["alpha_center_bias"],
                self._wp_alpha["alpha_passing"],
                self._wp_opp_s,
                self._wp_opp_ds,
                self._wp_opp_act,
                self._wp_prev_gate,
                int(n_others),
                float(control_dt),
                self.buffers.rewards,
                self._wp_r_progress,
                self._wp_r_collision,
                self._wp_r_boundary,
                self._wp_r_center,
                self._wp_r_passing,
                self._wp_gate,
            ],
            device=self.buffers.wp_device,
        )
        wp.copy(self._wp_prev_gate, self._wp_gate)
        return RewardTerms(
            total=self.buffers.torch_arrays.rewards,
            progress=wp.to_torch(self._wp_r_progress),
            collision=wp.to_torch(self._wp_r_collision),
            boundary=wp.to_torch(self._wp_r_boundary),
            lane_center=wp.to_torch(self._wp_r_center),
            passing=wp.to_torch(self._wp_r_passing),
        )

    def _launch_async_respawn(self, seed: int) -> None:
        cfg = self.cfg
        n = self.layout.num_slots
        vehicles = self.buffers.vehicle_buffers()
        wp.launch(
            async_respawn_kernel,
            dim=n,
            inputs=[
                vehicles,
                self.buffers.active,
                self.buffers.trainable,
                self.buffers.done,
                self.buffers.timeout,
                self.buffers.reset_mask,
                self.buffers.world_id,
                self.buffers.slot_id,
                self.buffers.track_id,
                self.buffers.episode_id,
                self.buffers.episode_step,
                self.buffers.stalled_steps,
                self.buffers.frenet_segment,
                self.buffers.frenet_s,
                self.buffers.frenet_ey,
                self.buffers.prev_s,
                self.buffers.progress_s,
                self.buffers.boundary_distance,
                self.buffers.wall_contact,
                self.buffers.contact,
                self.buffers.contact_counterpart,
                self.buffers.contact_closing_speed,
                self.buffers.executed_long_0,
                self.buffers.executed_long_1,
                self.buffers.executed_steer_0,
                self.buffers.executed_steer_1,
                self.buffers.executed_steer_2,
                self.buffers.executed_steer_3,
                self.buffers.sensor_noise_seed,
                self.buffers.lidar_range_noise_std,
                self.buffers.lidar_dropout_prob,
                self.buffers.lidar_far_dropout_prob,
                self.buffers.lidar_angle_bias,
                self.buffers.lidar_extrinsic_x,
                self.buffers.lidar_extrinsic_y,
                self.buffers.lidar_extrinsic_yaw,
                self.buffers.lidar_sector_start,
                self.buffers.lidar_sector_width,
                self.wp_point,
                self.wp_tangent,
                self.wp_normal,
                self.wp_seg_len,
                self.wp_cum,
                self.wp_width_left,
                self.wp_width_right,
                self.wp_offsets,
                self.wp_track_length,
                float(self.vehicle_params.mass),
                float(self.vehicle_params.tire_mu),
                float(self.vehicle_params.wheel_radius),
                float(cfg.agents.car_length_m),
                float(cfg.agents.car_width_m),
                int(cfg.worlds.max_agents_per_world),
                int(seed),
                0.2,
                0.25,
                64,
            ],
            device=self.buffers.wp_device,
        )
        reset_rows = self.buffers.torch_arrays.reset_mask > 0
        self.resample_styles_for_mask(reset_rows, seed + 7)
        # Only respawned rows lose their latched passing gate; clearing every row
        # would make the reward depend on world count and reset rate.
        self._torch_prev_gate[reset_rows] = 0

    def rebuild_sensors(self) -> torch.Tensor:
        """Re-run LiDAR/proprio from current SoA state (reconstruction path)."""
        self._launch_sensors()
        # No host synchronize: consumers use zero-copy Torch views / later CUDA ops.
        return self.buffers.torch_arrays.sensor_obs

    def action_observation(self) -> torch.Tensor:
        """Sensors matching the current SoA pose (built by last step/reset)."""
        return self.buffers.torch_arrays.sensor_obs

    def _invalidate_cuda_graph(self) -> None:
        if self._cuda_graph is not None:
            self._cuda_graph = None
            if self._cuda_graph_enabled:
                self._cuda_graph_warmup_left = 2
                self._cuda_graph_status = "pending"
        if self._sensor_cuda_graph is not None:
            self._sensor_cuda_graph = None
            if self._sensor_cuda_graph_enabled:
                self._sensor_cuda_graph_warmup_left = 2
                self._sensor_cuda_graph_status = "pending"

    def _sensor_kernel_inputs(self):
        vehicles = self.buffers.vehicle_buffers()
        return [
            vehicles,
            self.buffers.active,
            self.buffers.world_id,
            self.buffers.track_id,
            self.buffers.episode_id,
            self.buffers.episode_step,
            self.buffers.sensor_noise_seed,
            self.buffers.lidar_range_noise_std,
            self.buffers.lidar_dropout_prob,
            self.buffers.lidar_far_dropout_prob,
            self.buffers.lidar_angle_bias,
            self.buffers.lidar_extrinsic_x,
            self.buffers.lidar_extrinsic_y,
            self.buffers.lidar_extrinsic_yaw,
            self.buffers.lidar_sector_start,
            self.buffers.lidar_sector_width,
            self.buffers.executed_long_0,
            self.buffers.executed_long_1,
            self.buffers.executed_steer_0,
            self.buffers.executed_steer_1,
            self.buffers.executed_steer_2,
            self.buffers.executed_steer_3,
            self.wp_edt,
            self.wp_edt_offsets,
            self.wp_edt_width,
            self.wp_edt_height,
            self.wp_edt_origin,
            self.wp_edt_resolution,
            self.sensor_params,
            self.buffers.sensor_obs,
        ]

    def _launch_sensors_eager(self) -> None:
        n = self.layout.num_slots
        inputs = self._sensor_kernel_inputs()
        if self._use_beam_parallel_lidar:
            wp.launch(
                lidar_beam_parallel_kernel,
                dim=(n, int(LIDAR_DIM)),
                inputs=inputs,
                device=self.buffers.wp_device,
            )
        else:
            wp.launch(
                lidar_and_proprio_kernel,
                dim=n,
                inputs=inputs,
                device=self.buffers.wp_device,
            )

    def _launch_sensors(self) -> None:
        """Replay sensor CUDA graph when captured; otherwise eager launch."""
        if self._sensor_cuda_graph is not None:
            wp.capture_launch(self._sensor_cuda_graph)
            self._sensor_cuda_graph_status = "replaying"
            return
        if (
            self._sensor_cuda_graph_enabled
            and self._sensor_cuda_graph_warmup_left <= 0
            and self._sensor_cuda_graph is None
        ):
            try:
                with wp.ScopedCapture() as capture:
                    self._launch_sensors_eager()
                self._sensor_cuda_graph = capture.graph
                self._sensor_cuda_graph_status = "captured"
                # ScopedCapture only records the launches; this step's sensor
                # rebuild has not run yet, so replay the graph once before returning.
                wp.capture_launch(self._sensor_cuda_graph)
                return
            except Exception as exc:  # noqa: BLE001 — probe unsupported capture
                self._sensor_cuda_graph = None
                self._sensor_cuda_graph_enabled = False
                self._sensor_cuda_graph_status = f"unsupported:{type(exc).__name__}"
        self._launch_sensors_eager()
        if self._sensor_cuda_graph_enabled and self._sensor_cuda_graph is None:
            self._sensor_cuda_graph_warmup_left = max(
                0, self._sensor_cuda_graph_warmup_left - 1
            )
            self._sensor_cuda_graph_status = "warming"

    def state(self) -> SimulatorState:
        t = self.buffers.torch_arrays
        return SimulatorState(
            layout=WorldSlotLayout(
                num_worlds=self.layout.num_worlds,
                max_agents_per_world=self.layout.max_agents_per_world,
                num_slots=self.layout.num_slots,
            ),
            active=t.active,
            trainable=t.trainable,
            done=t.done,
            track_id=t.track_id,
            episode_id=t.episode_id,
            arrays=self.buffers.as_state_dict(),
        )

    def reset_all(self, seed: int) -> None:
        cfg = self.cfg
        track_ids = assign_world_tracks(cfg, self.geom, seed)
        counts = sample_active_counts(cfg, self.geom, track_ids, seed + 17)
        n_worlds = cfg.worlds.num_worlds
        n_agents = cfg.worlds.max_agents_per_world
        static_count = int(cfg.worlds.static_opponents_per_world)
        t = self.buffers.torch_arrays
        t.active.zero_()
        t.trainable.zero_()
        t.done.zero_()
        t.timeout.zero_()
        t.reset_mask.zero_()
        t.episode_id.zero_()
        t.episode_step.zero_()
        t.progress_s.zero_()
        self._spawn_stats = {"requested": int(counts.sum()), "realized": 0, "rejects": 0}

        device = self.buffers.torch_device
        n_slots = self.layout.num_slots
        static_mask = None
        pins: tuple[torch.Tensor, ...] | None = None
        if static_count > 0:
            static_mask = torch.zeros(n_slots, dtype=torch.bool, device=device)
            pin_x = torch.zeros(n_slots, dtype=torch.float32, device=device)
            pin_y = torch.zeros(n_slots, dtype=torch.float32, device=device)
            pin_yaw = torch.zeros(n_slots, dtype=torch.float32, device=device)
            pin_segment = torch.zeros(n_slots, dtype=torch.int32, device=device)
            pin_s = torch.zeros(n_slots, dtype=torch.float32, device=device)
            pin_ey = torch.zeros(n_slots, dtype=torch.float32, device=device)
            pin_boundary = torch.zeros(n_slots, dtype=torch.float32, device=device)
            pins = (pin_x, pin_y, pin_yaw, pin_segment, pin_s, pin_ey, pin_boundary)

        for w in range(n_worlds):
            tid = int(track_ids[w])
            horizon = horizon_for_track(cfg, self.geom, tid)
            existing: list[tuple[float, float, float]] = []
            for slot in range(n_agents):
                flat = w * n_agents + slot
                t.track_id[flat] = tid
                t.episode_horizon[flat] = horizon
                if slot >= int(counts[w]):
                    continue
                self._spawn_slot(flat, w, slot, episode_id=0, seed=seed, existing=existing)
                pose = (float(t.x[flat]), float(t.y[flat]), float(t.yaw[flat]))
                existing.append(pose)
            if static_mask is not None:
                assert pins is not None
                pin_x, pin_y, pin_yaw, pin_segment, pin_s, pin_ey, pin_boundary = pins
                poses = place_static_opponents(
                    self.geom,
                    tid,
                    static_count,
                    seed=mix_seed(seed, w, n_agents, 0),
                    racer_poses=existing,
                    car_length=cfg.agents.car_length_m,
                    car_width=cfg.agents.car_width_m,
                )
                if len(poses) != static_count:
                    raise RuntimeError(
                        f"could not deterministically place {static_count} static "
                        f"opponents on track {tid} (world {w})"
                    )
                for local, pose in enumerate(poses):
                    flat = w * n_agents + int(counts[w]) + local
                    t.episode_horizon[flat] = horizon
                    static_mask[flat] = True
                    pin_x[flat] = float(pose["x"])
                    pin_y[flat] = float(pose["y"])
                    pin_yaw[flat] = float(pose["yaw"])
                    pin_segment[flat] = int(pose["segment"])
                    pin_s[flat] = float(pose["s"])
                    pin_ey[flat] = float(pose["ey"])
                    pin_boundary[flat] = float(pose["boundary_distance"])
        if static_mask is not None:
            assert pins is not None
            apply_static_pins(t, static_mask, *pins, reset_contact=True)
        self._static_mask = static_mask
        self._static_pin = pins
        self.resample_styles_for_mask(torch.ones(self.layout.num_slots), seed + 91)
        self._passing_gate.zero_()
        # Action-time observations are the current SoA sensors (no collect rebuild).
        self.rebuild_sensors()
        self._invalidate_cuda_graph()

    def _pin_static_slots(self) -> None:
        """Re-clamp static-opponent slots to their placement pose every step.

        Runs after physics/contact/terminal but before the reward kernels, so
        a contact impulse this step cannot leak into the passing reward's
        opponent-progress computation or into next step's LiDAR trace.
        """
        if self._static_mask is None:
            return
        assert self._static_pin is not None
        t = self.buffers.torch_arrays
        apply_static_pins(t, self._static_mask, *self._static_pin, reset_contact=False)

    def _spawn_slot(
        self,
        flat: int,
        world_id: int,
        slot: int,
        *,
        episode_id: int,
        seed: int,
        existing: list[tuple[float, float, float]] | None = None,
    ) -> None:
        cfg = self.cfg
        t = self.buffers.torch_arrays
        tid = int(t.track_id[flat].item())
        rng = np.random.default_rng(mix_seed(seed, world_id, slot, episode_id))
        if existing is None:
            existing = self._active_poses_in_world(world_id, exclude=flat)
        pose = spawn_agent_pose(
            rng,
            self.geom,
            tid,
            existing,
            car_length=cfg.agents.car_length_m,
            car_width=cfg.agents.car_width_m,
        )
        corr = sample_lidar_corruption(rng)
        wheel = float(self.vehicle_params.wheel_radius)
        speed = float(pose["vx"])
        t.x[flat] = float(pose["x"])
        t.y[flat] = float(pose["y"])
        t.yaw[flat] = float(pose["yaw"])
        t.vx[flat] = speed
        t.vy[flat] = 0.0
        t.yaw_rate[flat] = 0.0
        t.steer[flat] = 0.0
        t.effort_state[flat] = 0.0
        t.applied_effort[flat] = 0.0
        t.ax[flat] = 0.0
        t.ay[flat] = 0.0
        t.mass[flat] = float(self.vehicle_params.mass)
        t.mu[flat] = float(self.vehicle_params.tire_mu)
        t.drive_scale[flat] = 1.0
        t.steer_scale[flat] = 1.0
        t.accel_scale[flat] = 1.0
        t.vmax_scale[flat] = 1.0
        t.frenet_segment[flat] = int(pose["segment"])
        t.frenet_s[flat] = float(pose["s"])
        t.prev_s[flat] = float(pose["s"])
        t.frenet_ey[flat] = float(pose["ey"])
        t.boundary_distance[flat] = float(pose["boundary_distance"])
        t.progress_s[flat] = 0.0
        t.episode_id[flat] = episode_id
        t.episode_step[flat] = 0
        t.stalled_steps[flat] = 0
        t.done[flat] = 0
        t.timeout[flat] = 0
        t.reset_mask[flat] = 0
        t.active[flat] = 1
        t.trainable[flat] = 1
        t.contact[flat] = 0
        t.wall_contact[flat] = 0
        t.executed_long_0[flat] = 0.0
        t.executed_long_1[flat] = 0.0
        t.executed_steer_0[flat] = 0.0
        t.executed_steer_1[flat] = 0.0
        t.executed_steer_2[flat] = 0.0
        t.executed_steer_3[flat] = 0.0
        t.sensor_noise_seed[flat] = mix_seed(seed, world_id, slot, episode_id + 99)
        t.lidar_range_noise_std[flat] = float(corr["range_noise_std"])
        t.lidar_dropout_prob[flat] = float(corr["dropout_prob"])
        t.lidar_far_dropout_prob[flat] = float(corr["far_dropout_prob"])
        t.lidar_angle_bias[flat] = float(corr["angle_bias"])
        t.lidar_extrinsic_x[flat] = float(corr["extrinsic_x"])
        t.lidar_extrinsic_y[flat] = float(corr["extrinsic_y"])
        t.lidar_extrinsic_yaw[flat] = float(corr["extrinsic_yaw"])
        t.lidar_sector_start[flat] = int(corr["sector_start"])
        t.lidar_sector_width[flat] = int(corr["sector_width"])
        om = np.array(self.buffers.omega.numpy(), copy=True)
        om[flat] = np.full(4, speed / wheel, dtype=np.float32)
        wp.copy(
            self.buffers.omega,
            wp.array(om, dtype=wp.vec4f, device=self.buffers.wp_device),
        )
        self._spawn_stats["realized"] += 1
        self._spawn_stats["rejects"] += int(pose["rejected"])

    def _active_poses_in_world(
        self, world_id: int, exclude: int | None = None
    ) -> list[tuple[float, float, float]]:
        t = self.buffers.torch_arrays
        n_agents = self.layout.max_agents_per_world
        out: list[tuple[float, float, float]] = []
        base = world_id * n_agents
        for slot in range(n_agents):
            flat = base + slot
            if exclude is not None and flat == exclude:
                continue
            if int(t.active[flat].item()) == 0:
                continue
            out.append((float(t.x[flat]), float(t.y[flat]), float(t.yaw[flat])))
        return out

    def reset_agents(self, agent_mask: Any, seed: int) -> None:
        mask = torch.as_tensor(agent_mask, device=self.buffers.torch_device).view(-1)
        t = self.buffers.torch_arrays
        n_agents = self.layout.max_agents_per_world
        for flat in range(self.layout.num_slots):
            if int(mask[flat].item()) == 0:
                continue
            world_id = flat // n_agents
            slot = flat % n_agents
            ep = int(t.episode_id[flat].item()) + 1
            self._spawn_slot(
                flat,
                world_id,
                slot,
                episode_id=ep,
                seed=seed,
            )

    def _launch_physics_contact_terminal(self) -> None:
        """Pure Warp launches with static shapes (safe CUDA-graph candidates)."""
        cfg = self.cfg
        n = self.layout.num_slots
        vehicles = self.buffers.vehicle_buffers()
        half_w = 0.5 * cfg.agents.car_width_m
        wp_actions = self._wp_actions

        wp.launch(
            physics_control_kernel,
            dim=n,
            inputs=[
                vehicles,
                self.buffers.active,
                wp_actions,
                self.buffers.prev_x,
                self.buffers.prev_y,
                self.buffers.prev_yaw,
                self.sim_params,
                int(self._control_interval),
            ],
            device=self.buffers.wp_device,
        )
        wp.launch(
            clear_contact_accumulators,
            dim=n,
            inputs=[
                self.buffers.impulse_x,
                self.buffers.impulse_y,
                self.buffers.correction_x,
                self.buffers.correction_y,
                self.buffers.contact,
                self.buffers.contact_counterpart,
                self.buffers.contact_closing_speed,
                self.buffers.pair_count,
                self.buffers.broadphase_overflow,
            ],
            device=self.buffers.wp_device,
        )
        wp.launch(
            build_world_pairs_kernel,
            dim=n,
            inputs=[
                self.buffers.active,
                self.buffers.world_id,
                self.buffers.x,
                self.buffers.y,
                self.buffers.prev_x,
                self.buffers.prev_y,
                self.contact_params,
                self.buffers.pair_i,
                self.buffers.pair_j,
                self.buffers.pair_count,
                self.buffers.broadphase_overflow,
            ],
            device=self.buffers.wp_device,
        )
        max_pairs = int(self.contact_params.max_pairs)
        wp.launch(
            resolve_pairs_jacobi_kernel,
            dim=max_pairs,
            inputs=[
                vehicles,
                self.buffers.pair_i,
                self.buffers.pair_j,
                self.buffers.pair_count,
                self.contact_params,
                self.buffers.impulse_x,
                self.buffers.impulse_y,
                self.buffers.correction_x,
                self.buffers.correction_y,
                self.buffers.contact,
                self.buffers.contact_counterpart,
                self.buffers.contact_closing_speed,
            ],
            device=self.buffers.wp_device,
        )
        wp.launch(
            apply_contact_gather_kernel,
            dim=n,
            inputs=[
                vehicles,
                self.buffers.active,
                self.buffers.impulse_x,
                self.buffers.impulse_y,
                self.buffers.correction_x,
                self.buffers.correction_y,
            ],
            device=self.buffers.wp_device,
        )
        wp.launch(
            progress_wall_kernel,
            dim=n,
            inputs=[
                vehicles,
                self.buffers.active,
                self.buffers.track_id,
                self.buffers.frenet_segment,
                self.buffers.frenet_s,
                self.buffers.frenet_ey,
                self.buffers.prev_s,
                self.buffers.progress_s,
                self.buffers.boundary_distance,
                self.buffers.wall_contact,
                self.buffers.rewards,
                self.wp_point,
                self.wp_tangent,
                self.wp_normal,
                self.wp_seg_len,
                self.wp_cum,
                self.wp_width_left,
                self.wp_width_right,
                self.wp_offsets,
                self.wp_track_length,
                half_w,
            ],
            device=self.buffers.wp_device,
        )
        wp.launch(
            push_command_history_kernel,
            dim=n,
            inputs=[
                self.buffers.active,
                self.buffers.applied_effort,
                self.buffers.steer,
                self.buffers.executed_long_0,
                self.buffers.executed_long_1,
                self.buffers.executed_steer_0,
                self.buffers.executed_steer_1,
                self.buffers.executed_steer_2,
                self.buffers.executed_steer_3,
            ],
            device=self.buffers.wp_device,
        )
        async_respawn = 1 if cfg.agents.async_respawn else 0
        sync_flag = 1 if self.sync_no_respawn else 0
        wp.launch(
            terminal_mask_kernel,
            dim=n,
            inputs=[
                vehicles,
                self.buffers.active,
                self.buffers.trainable,
                self.buffers.done,
                self.buffers.timeout,
                self.buffers.reset_mask,
                self.buffers.episode_step,
                self.buffers.episode_horizon,
                self.buffers.boundary_distance,
                self.buffers.wall_contact,
                self.buffers.contact,
                self.buffers.contact_closing_speed,
                self.buffers.stalled_steps,
                async_respawn,
                sync_flag,
                half_w,
                4.0,
                0.15,
                50,
            ],
            device=self.buffers.wp_device,
        )

    def _run_graphable_launches(self) -> None:
        """Replay captured graph or launch; capture once after warmup if supported."""
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
            self._cuda_graph_status = "replaying"
            return
        if (
            self._cuda_graph_enabled
            and self._cuda_graph_warmup_left <= 0
            and self._cuda_graph is None
        ):
            try:
                with wp.ScopedCapture() as capture:
                    self._launch_physics_contact_terminal()
                self._cuda_graph = capture.graph
                self._cuda_graph_status = "captured"
                # ScopedCapture only records the launches; this step's physics
                # has not run yet, so replay the graph once before returning.
                wp.capture_launch(self._cuda_graph)
                return
            except Exception as exc:  # noqa: BLE001 — probe unsupported capture paths
                self._cuda_graph = None
                self._cuda_graph_enabled = False
                self._cuda_graph_status = f"unsupported:{type(exc).__name__}"
                # Capture may not have executed; fall through to eager launch.
        self._launch_physics_contact_terminal()
        if self._cuda_graph_enabled and self._cuda_graph is None:
            self._cuda_graph_warmup_left = max(0, self._cuda_graph_warmup_left - 1)
            self._cuda_graph_status = "warming"

    def step(self, actions: Any) -> dict[str, Any]:
        cfg = self.cfg
        n = self.layout.num_slots
        act = torch.as_tensor(
            actions, dtype=torch.float32, device=self.buffers.torch_device
        ).reshape(n, 2)
        act = torch.clamp(act, -1.0, 1.0)
        self._action_buf.copy_(act)

        if self._style_tensors is None:
            self.apply_styles(self.styles)

        # Physics/contact/terminal: static-shape Warp graph when supported.
        self._run_graphable_launches()
        # Undo any contact-impulse motion on static opponents before rewards
        # or sensors read this step's position/frenet state.
        self._pin_static_slots()

        # Rewards / async respawn / sensors stay eager (Torch interop + style scatter).
        terms = self._launch_reward_kernels()
        # Clone term views: callers retain them across later steps / respawns.
        term_dict = {k: v.detach().clone() for k, v in terms.as_dict().items()}
        self.last_reward_terms = term_dict
        t = self.buffers.torch_arrays
        # Snapshot transition-time flags before async respawn clears slot state.
        # PPO GAE needs done/timeout from the dying transition; GRU/style use
        # reset_mask (kept 1 through respawn). Contact/wall feed rollout metrics.
        done_t = (t.done > 0).clone()
        timeout_t = (t.timeout > 0).clone()
        reset_t = (t.reset_mask > 0).clone()
        contact_t = (t.contact > 0).clone()
        wall_t = (t.wall_contact > 0).clone()
        # Next state of this transition, before respawn overwrites the slot.
        # PPO bootstraps truncated episodes from V(next_state).
        next_compact = self.pack_state()
        async_respawn = bool(cfg.agents.async_respawn) and not self.sync_no_respawn
        if async_respawn:
            self._launch_async_respawn(cfg.seed)
        else:
            wp.launch(
                deactivate_terminal_rows_kernel,
                dim=n,
                inputs=[
                    self.buffers.active,
                    self.buffers.trainable,
                    self.buffers.done,
                ],
                device=self.buffers.wp_device,
            )
        self._launch_sensors()

        # Return zero-copy views for hot fields. Callers that retain tensors across
        # subsequent step()/sensor rebuilds must copy (rollout buffer store_step does).
        compact = self.pack_state()
        critic_state = {
            "x": t.x,
            "y": t.y,
            "yaw": t.yaw,
            "vx": t.vx,
            "vy": t.vy,
            "frenet_s": t.frenet_s,
            "active": t.active,
            "world_id": t.world_id,
            "track_id": t.track_id,
            "compact_state": compact,
        }
        return {
            "rewards": t.rewards,
            "done": done_t,
            "timeout": timeout_t,
            "reset_mask": reset_t,
            "contact": contact_t,
            "wall_contact": wall_t,
            "sensor_obs": t.sensor_obs,
            "compact_state": compact,
            "next_compact_state": next_compact,
            "critic_state": critic_state,
            "reward_terms": dict(term_dict),
            "spawn_stats": dict(self._spawn_stats),
            "cuda_graph_status": self._cuda_graph_status,
            "sensor_cuda_graph_status": self._sensor_cuda_graph_status,
            # Device-resident overflow flag (zero-copy). Tests may .item().
            "broadphase_overflow": self._overflow_torch[0],
        }


def build_n_agent_simulator(
    cfg: ExperimentConfig,
    atlas: PackedTrackAtlasView | None,
    device: str,
    *,
    sync_no_respawn: bool | None = None,
) -> WarpNAgentSimulator:
    if atlas is None:
        atlas = make_synthetic_oval_atlas(
            max_agents=cfg.worlds.max_agents_per_world,
        )
    return WarpNAgentSimulator(
        cfg, atlas, device, sync_no_respawn=sync_no_respawn
    )

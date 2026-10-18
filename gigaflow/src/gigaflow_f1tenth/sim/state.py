"""Fixed-capacity SoA world/agent storage for N-car worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.sim.layout_local import SENSOR_OBS_DIM
from gigaflow_f1tenth.sim.vehicle import VehicleBuffers, VehicleParams


@dataclass(frozen=True)
class SlotLayout:
    num_worlds: int
    max_agents_per_world: int
    num_slots: int


def slot_layout_from_config(cfg: ExperimentConfig) -> SlotLayout:
    n_worlds = cfg.worlds.num_worlds
    n_agents = cfg.worlds.max_agents_per_world
    return SlotLayout(
        num_worlds=n_worlds,
        max_agents_per_world=n_agents,
        num_slots=n_worlds * n_agents,
    )


def _device_str(device: str) -> str:
    wanted = str(device)
    if wanted.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={wanted!r} requested but torch.cuda.is_available() is False; "
                "refusing silent CPU fallback for simulator buffers"
            )
        return "cuda"
    return "cpu"


def _wp_device(device: str):
    return wp.get_device("cuda:0" if _device_str(device) == "cuda" else "cpu")


@dataclass
class SoaArrays:
    """Torch views (zero-copy where possible) over simulator SoA buffers."""

    active: torch.Tensor
    trainable: torch.Tensor
    done: torch.Tensor
    timeout: torch.Tensor
    reset_mask: torch.Tensor
    track_id: torch.Tensor
    episode_id: torch.Tensor
    episode_step: torch.Tensor
    episode_horizon: torch.Tensor
    world_id: torch.Tensor
    slot_id: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    yaw: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor
    yaw_rate: torch.Tensor
    steer: torch.Tensor
    effort_state: torch.Tensor
    applied_effort: torch.Tensor
    ax: torch.Tensor
    ay: torch.Tensor
    omega: torch.Tensor
    mass: torch.Tensor
    mu: torch.Tensor
    drive_scale: torch.Tensor
    steer_scale: torch.Tensor
    accel_scale: torch.Tensor
    vmax_scale: torch.Tensor
    frenet_s: torch.Tensor
    frenet_ey: torch.Tensor
    frenet_segment: torch.Tensor
    progress_s: torch.Tensor
    prev_s: torch.Tensor
    boundary_distance: torch.Tensor
    wall_contact: torch.Tensor
    contact: torch.Tensor
    contact_counterpart: torch.Tensor
    contact_closing_speed: torch.Tensor
    prev_x: torch.Tensor
    prev_y: torch.Tensor
    prev_yaw: torch.Tensor
    sensor_obs: torch.Tensor
    sensor_noise_seed: torch.Tensor
    lidar_range_noise_std: torch.Tensor
    lidar_dropout_prob: torch.Tensor
    lidar_far_dropout_prob: torch.Tensor
    lidar_angle_bias: torch.Tensor
    lidar_extrinsic_x: torch.Tensor
    lidar_extrinsic_y: torch.Tensor
    lidar_extrinsic_yaw: torch.Tensor
    lidar_sector_start: torch.Tensor
    lidar_sector_width: torch.Tensor
    rewards: torch.Tensor
    stalled_steps: torch.Tensor
    executed_long_0: torch.Tensor
    executed_long_1: torch.Tensor
    executed_steer_0: torch.Tensor
    executed_steer_1: torch.Tensor
    executed_steer_2: torch.Tensor
    executed_steer_3: torch.Tensor


class SimulatorBuffers:
    def __init__(self, cfg: ExperimentConfig, device: str):
        self.cfg = cfg
        self.layout = slot_layout_from_config(cfg)
        self.device_str = _device_str(device)
        self.wp_device = _wp_device(self.device_str)
        self.torch_device = torch.device(self.device_str)
        n = self.layout.num_slots
        wp.set_device(self.wp_device)

        def f32(fill: float = 0.0):
            return wp.zeros(n, dtype=wp.float32, device=self.wp_device)

        def i32(fill: int = 0):
            return wp.zeros(n, dtype=wp.int32, device=self.wp_device)

        def u8(fill: int = 0):
            return wp.zeros(n, dtype=wp.uint8, device=self.wp_device)

        self.active = u8()
        self.trainable = u8()
        self.done = u8()
        self.timeout = u8()
        self.reset_mask = u8()
        self.track_id = i32()
        self.episode_id = i32()
        self.episode_step = i32()
        self.episode_horizon = i32()
        self.world_id = i32()
        self.slot_id = i32()

        self.x = f32()
        self.y = f32()
        self.yaw = f32()
        self.vx = f32()
        self.vy = f32()
        self.yaw_rate = f32()
        self.steer = f32()
        self.effort_state = f32()
        self.applied_effort = f32()
        self.ax = f32()
        self.ay = f32()
        self.omega = wp.zeros(n, dtype=wp.vec4f, device=self.wp_device)
        self.slip_ratio = wp.zeros(n, dtype=wp.vec4f, device=self.wp_device)
        self.slip_angle = wp.zeros(n, dtype=wp.vec4f, device=self.wp_device)
        self.load_ratio = wp.zeros(n, dtype=wp.vec4f, device=self.wp_device)
        self.fx_lag = wp.zeros(n, dtype=wp.vec4f, device=self.wp_device)
        self.fy_lag = wp.zeros(n, dtype=wp.vec4f, device=self.wp_device)
        self.mass = f32()
        self.mu = f32()
        self.drive_scale = f32()
        self.steer_scale = f32()
        self.accel_scale = f32()
        self.vmax_scale = f32()

        self.frenet_s = f32()
        self.frenet_ey = f32()
        self.frenet_segment = i32()
        self.progress_s = f32()
        self.prev_s = f32()
        self.boundary_distance = f32()
        self.wall_contact = u8()
        self.contact = u8()
        self.contact_counterpart = i32()
        self.contact_closing_speed = f32()
        self.prev_x = f32()
        self.prev_y = f32()
        self.prev_yaw = f32()

        self.sensor_obs = wp.zeros(
            (n, SENSOR_OBS_DIM), dtype=wp.float32, device=self.wp_device
        )
        self.sensor_noise_seed = i32()
        self.lidar_range_noise_std = f32()
        self.lidar_dropout_prob = f32()
        self.lidar_far_dropout_prob = f32()
        self.lidar_angle_bias = f32()
        self.lidar_extrinsic_x = f32()
        self.lidar_extrinsic_y = f32()
        self.lidar_extrinsic_yaw = f32()
        self.lidar_sector_start = i32()
        self.lidar_sector_width = i32()
        self.rewards = f32()
        self.stalled_steps = i32()
        self.executed_long_0 = f32()
        self.executed_long_1 = f32()
        self.executed_steer_0 = f32()
        self.executed_steer_1 = f32()
        self.executed_steer_2 = f32()
        self.executed_steer_3 = f32()

        # Broadphase scratch (fixed capacity).
        max_pairs = max(n * max(cfg.worlds.max_agents_per_world, 1), 1)
        self.pair_i = wp.zeros(max_pairs, dtype=wp.int32, device=self.wp_device)
        self.pair_j = wp.zeros(max_pairs, dtype=wp.int32, device=self.wp_device)
        self.pair_count = wp.zeros(1, dtype=wp.int32, device=self.wp_device)
        self.impulse_x = f32()
        self.impulse_y = f32()
        self.correction_x = f32()
        self.correction_y = f32()
        self.broadphase_overflow = wp.zeros(1, dtype=wp.int32, device=self.wp_device)

        self._init_index_maps()
        self._torch = self._build_torch_views()

    def _init_index_maps(self) -> None:
        n_worlds = self.layout.num_worlds
        n_agents = self.layout.max_agents_per_world
        world = np.repeat(np.arange(n_worlds, dtype=np.int32), n_agents)
        slot = np.tile(np.arange(n_agents, dtype=np.int32), n_worlds)
        wp.copy(self.world_id, wp.array(world, dtype=wp.int32, device=self.wp_device))
        wp.copy(self.slot_id, wp.array(slot, dtype=wp.int32, device=self.wp_device))

    def vehicle_buffers(self) -> VehicleBuffers:
        buf = VehicleBuffers()
        buf.x = self.x
        buf.y = self.y
        buf.yaw = self.yaw
        buf.vx = self.vx
        buf.vy = self.vy
        buf.yaw_rate = self.yaw_rate
        buf.steer = self.steer
        buf.effort_state = self.effort_state
        buf.applied_effort = self.applied_effort
        buf.ax = self.ax
        buf.ay = self.ay
        buf.omega = self.omega
        buf.slip_ratio = self.slip_ratio
        buf.slip_angle = self.slip_angle
        buf.load_ratio = self.load_ratio
        buf.fx_lag = self.fx_lag
        buf.fy_lag = self.fy_lag
        buf.mass = self.mass
        buf.mu = self.mu
        buf.drive_scale = self.drive_scale
        buf.steer_scale = self.steer_scale
        buf.accel_scale = self.accel_scale
        buf.vmax_scale = self.vmax_scale
        return buf

    def _wp_to_torch(self, arr) -> torch.Tensor:
        return wp.to_torch(arr)

    def _build_torch_views(self) -> SoaArrays:
        return SoaArrays(
            active=self._wp_to_torch(self.active),
            trainable=self._wp_to_torch(self.trainable),
            done=self._wp_to_torch(self.done),
            timeout=self._wp_to_torch(self.timeout),
            reset_mask=self._wp_to_torch(self.reset_mask),
            track_id=self._wp_to_torch(self.track_id),
            episode_id=self._wp_to_torch(self.episode_id),
            episode_step=self._wp_to_torch(self.episode_step),
            episode_horizon=self._wp_to_torch(self.episode_horizon),
            world_id=self._wp_to_torch(self.world_id),
            slot_id=self._wp_to_torch(self.slot_id),
            x=self._wp_to_torch(self.x),
            y=self._wp_to_torch(self.y),
            yaw=self._wp_to_torch(self.yaw),
            vx=self._wp_to_torch(self.vx),
            vy=self._wp_to_torch(self.vy),
            yaw_rate=self._wp_to_torch(self.yaw_rate),
            steer=self._wp_to_torch(self.steer),
            effort_state=self._wp_to_torch(self.effort_state),
            applied_effort=self._wp_to_torch(self.applied_effort),
            ax=self._wp_to_torch(self.ax),
            ay=self._wp_to_torch(self.ay),
            omega=self._wp_to_torch(self.omega),
            mass=self._wp_to_torch(self.mass),
            mu=self._wp_to_torch(self.mu),
            drive_scale=self._wp_to_torch(self.drive_scale),
            steer_scale=self._wp_to_torch(self.steer_scale),
            accel_scale=self._wp_to_torch(self.accel_scale),
            vmax_scale=self._wp_to_torch(self.vmax_scale),
            frenet_s=self._wp_to_torch(self.frenet_s),
            frenet_ey=self._wp_to_torch(self.frenet_ey),
            frenet_segment=self._wp_to_torch(self.frenet_segment),
            progress_s=self._wp_to_torch(self.progress_s),
            prev_s=self._wp_to_torch(self.prev_s),
            boundary_distance=self._wp_to_torch(self.boundary_distance),
            wall_contact=self._wp_to_torch(self.wall_contact),
            contact=self._wp_to_torch(self.contact),
            contact_counterpart=self._wp_to_torch(self.contact_counterpart),
            contact_closing_speed=self._wp_to_torch(self.contact_closing_speed),
            prev_x=self._wp_to_torch(self.prev_x),
            prev_y=self._wp_to_torch(self.prev_y),
            prev_yaw=self._wp_to_torch(self.prev_yaw),
            sensor_obs=self._wp_to_torch(self.sensor_obs),
            sensor_noise_seed=self._wp_to_torch(self.sensor_noise_seed),
            lidar_range_noise_std=self._wp_to_torch(self.lidar_range_noise_std),
            lidar_dropout_prob=self._wp_to_torch(self.lidar_dropout_prob),
            lidar_far_dropout_prob=self._wp_to_torch(self.lidar_far_dropout_prob),
            lidar_angle_bias=self._wp_to_torch(self.lidar_angle_bias),
            lidar_extrinsic_x=self._wp_to_torch(self.lidar_extrinsic_x),
            lidar_extrinsic_y=self._wp_to_torch(self.lidar_extrinsic_y),
            lidar_extrinsic_yaw=self._wp_to_torch(self.lidar_extrinsic_yaw),
            lidar_sector_start=self._wp_to_torch(self.lidar_sector_start),
            lidar_sector_width=self._wp_to_torch(self.lidar_sector_width),
            rewards=self._wp_to_torch(self.rewards),
            stalled_steps=self._wp_to_torch(self.stalled_steps),
            executed_long_0=self._wp_to_torch(self.executed_long_0),
            executed_long_1=self._wp_to_torch(self.executed_long_1),
            executed_steer_0=self._wp_to_torch(self.executed_steer_0),
            executed_steer_1=self._wp_to_torch(self.executed_steer_1),
            executed_steer_2=self._wp_to_torch(self.executed_steer_2),
            executed_steer_3=self._wp_to_torch(self.executed_steer_3),
        )

    @property
    def torch_arrays(self) -> SoaArrays:
        return self._torch

    def as_state_dict(self) -> dict[str, Any]:
        t = self._torch
        return {
            "active": t.active,
            "trainable": t.trainable,
            "done": t.done,
            "timeout": t.timeout,
            "reset_mask": t.reset_mask,
            "track_id": t.track_id,
            "episode_id": t.episode_id,
            "x": t.x,
            "y": t.y,
            "yaw": t.yaw,
            "vx": t.vx,
            "vy": t.vy,
            "frenet_s": t.frenet_s,
            "progress_s": t.progress_s,
            "boundary_distance": t.boundary_distance,
            "contact": t.contact,
            "sensor_obs": t.sensor_obs,
            "rewards": t.rewards,
        }

    def seed_defaults(self, params: VehicleParams) -> None:
        n = self.layout.num_slots
        self._torch.mass.fill_(float(params.mass))
        self._torch.mu.fill_(float(params.tire_mu))
        self._torch.drive_scale.fill_(1.0)
        self._torch.steer_scale.fill_(1.0)
        self._torch.accel_scale.fill_(1.0)
        self._torch.vmax_scale.fill_(1.0)
        ones = torch.ones(n, 4, device=self.torch_device, dtype=torch.float32)
        # load_ratio default 1 via warp array init is zero; set via numpy copy.
        lr = np.ones((n, 4), dtype=np.float32)
        wp.copy(self.load_ratio, wp.array(lr, dtype=wp.vec4f, device=self.wp_device))
        del ones

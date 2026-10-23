"""20 Hz Gigaflow inference: 1097-D sensor obs + frozen 10-D IRL condition."""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
import torch

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.sensor_action import (
    integrate_steering_delta,
    normalized_action_to_physical,
)
from f1tenth_rl_agent.sensor_inference_runtime import (
    InferenceTimings,
    RealizedActuatorCommand,
)
from f1tenth_rl_agent.sensor_preprocessing import (
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    observation_is_finite,
    pack_actor_observation,
    pack_lidar_from_scan,
)

HISTORY_STRIDE = 2
STEER_RING = 1 + 3 * HISTORY_STRIDE  # 7 samples @ 20 Hz → 10 Hz-equivalent slots
LONG_RING = 1 + HISTORY_STRIDE
STEERING_DELTA_MAX_RAD = float(np.pi / 120.0)
FOLDED_CONDITION_DIM = 10


class FoldedLidarGruAdapter:
    """Call a 272-D GRU-in ``lidar_cnn_gru`` actor from GigaflowInferenceRuntime."""

    def __init__(self, actor, obs_normalizer=None) -> None:
        self.actor = actor
        self.obs_normalizer = obs_normalizer

    def initial_hidden(self, batch_size: int, device: str = "cpu"):
        return self.actor.initial_hidden(
            int(batch_size), device=device, dtype=torch.float32
        )

    def forward(self, sensor_obs, private_condition, hidden, deterministic=True):
        del private_condition
        obs = sensor_obs
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        action, _, next_hidden = self.actor.step(
            obs, hidden, deterministic=deterministic, with_logprob=False
        )
        return type(self)._Out(actions=action, hidden=next_hidden)

    class _Out:
        __slots__ = ("actions", "hidden")

        def __init__(self, actions, hidden) -> None:
            self.actions = actions
            self.hidden = hidden


def frozen_condition_from_payload(payload: Mapping) -> torch.Tensor | None:
    """Resolve the 10-D IRL condition, including ``actor_state_dict`` buffers."""
    condition = payload.get("deployment_condition_normalized")
    if condition is None:
        condition = payload.get("baked_condition")
    if condition is None:
        state = payload.get("actor_state_dict")
        if isinstance(state, Mapping) and "baked_condition" in state:
            condition = state["baked_condition"]
    if condition is None:
        return None
    vec = torch.as_tensor(condition, dtype=torch.float32).reshape(-1)
    return vec.view(1, -1)


class GigaflowInferenceRuntime:
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device, **kwargs):
        from f1tenth_rl_agent.policy_model import load_gigaflow_actor

        actor, condition, _payload = load_gigaflow_actor(checkpoint_path, device)
        return cls(actor, condition, device, **kwargs)

    def __init__(
        self,
        actor,
        condition: torch.Tensor,
        device: torch.device,
        *,
        deterministic: bool = True,
        warmup_iters: int = 5,
        steering_delta_max_rad: float = STEERING_DELTA_MAX_RAD,
        max_steer: float = si.MAX_STEER_RAD,
        i_drive_max_a: float = 40.0,
        i_brake_max_a: float = 40.0,
        steering_angle_to_servo_gain: float = 1.0,
        steering_angle_to_servo_offset: float = 0.5,
    ) -> None:
        self.actor = actor
        if condition.dim() == 1:
            condition = condition.view(1, -1)
        self.condition = condition.to(device=device, dtype=torch.float32)
        self.device = device
        self.deterministic = bool(deterministic)
        self.steering_delta_max_rad = float(steering_delta_max_rad)
        self.max_steer = float(max_steer)
        self.i_drive_max_a = float(i_drive_max_a)
        self.i_brake_max_a = float(i_brake_max_a)
        self.steering_angle_to_servo_gain = float(steering_angle_to_servo_gain)
        self.steering_angle_to_servo_offset = float(steering_angle_to_servo_offset)
        self.realized_steer_rad = 0.0
        self._lidar_buf = np.full(si.LIDAR_DIM, si.LIDAR_RANGE_MAX, dtype=np.float32)
        self._steer_ring = np.zeros(STEER_RING, dtype=np.float32)
        self._long_ring = np.zeros(LONG_RING, dtype=np.float32)
        self._strided_steer = np.zeros(si.STEER_HISTORY, dtype=np.float32)
        self._action_host = np.zeros(si.NUM_ACTIONS, dtype=np.float32)
        self._hidden = actor.initial_hidden(1, device=str(device))
        self._obs_device = torch.zeros(
            (1, si.NUM_OBS), dtype=torch.float32, device=device
        )
        self._action_device = torch.zeros(
            (1, si.NUM_ACTIONS), dtype=torch.float32, device=device
        )
        self._use_cuda = device.type == "cuda"
        self._obs_host_np = np.zeros(si.NUM_OBS, dtype=np.float32)
        self.last_log_prob: float | None = None
        self.warmup(warmup_iters)

    def reset_hidden(self) -> None:
        self._hidden.zero_()
        self.realized_steer_rad = 0.0
        self._steer_ring.fill(0.0)
        self._long_ring.fill(0.0)

    def pack_observation(
        self,
        *,
        scan_angle_min: float,
        scan_angle_increment: float,
        scan_ranges,
        scan_range_min: float,
        imu_interval: list[RawImuSample],
        imu_cal: ImuCalibration,
        speed: float,
        applied_long: float,
        applied_steer_rad: float,
        vesc_current: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        pack_lidar_from_scan(
            scan_angle_min,
            scan_angle_increment,
            scan_ranges,
            scan_range_min=scan_range_min,
            out=self._lidar_buf,
        )
        actor_imu, raw_imu = actor_imu_from_interval(
            imu_interval, imu_cal, freeze_const_channels=True
        )
        steer = self._steer_ring
        long_h = self._long_ring
        # Newest sample at index 0; stride-2 slots match Gigaflow 20 Hz packing.
        self._strided_steer[0] = float(applied_steer_rad)
        self._strided_steer[1] = steer[1]
        self._strided_steer[2] = steer[3]
        self._strided_steer[3] = steer[5]
        raw_obs = self._obs_host_np
        pack_actor_observation(
            self._lidar_buf,
            actor_imu,
            speed,
            vesc_current,
            float(applied_long),
            float(long_h[1]),
            self._strided_steer,
            out=raw_obs,
        )
        finite = bool(observation_is_finite(raw_obs))
        if finite:
            steer[1:] = steer[:-1]
            steer[0] = float(applied_steer_rad)
            long_h[1:] = long_h[:-1]
            long_h[0] = float(applied_long)
        return raw_obs, actor_imu, raw_imu, finite

    def realize_action(self, action_np: np.ndarray) -> RealizedActuatorCommand:
        t0 = time.perf_counter()
        self.realized_steer_rad = integrate_steering_delta(
            self.realized_steer_rad,
            float(action_np[1]),
            delta_max_rad=self.steering_delta_max_rad,
            max_steer=self.max_steer,
            clip_actions=si.CLIP_ACTIONS,
        )
        steering_command = self.realized_steer_rad / self.max_steer
        drive_a, brake_a, servo, long_norm, steer_norm = (
            normalized_action_to_physical(
                float(action_np[0]),
                steering_command,
                i_drive_max_a=self.i_drive_max_a,
                i_brake_max_a=self.i_brake_max_a,
                max_steer=self.max_steer,
                steering_angle_to_servo_gain=self.steering_angle_to_servo_gain,
                steering_angle_to_servo_offset=self.steering_angle_to_servo_offset,
                clip_actions=si.CLIP_ACTIONS,
            )
        )
        return RealizedActuatorCommand(
            drive_a=drive_a,
            brake_a=brake_a,
            servo=servo,
            long_norm=long_norm,
            steer_norm=steer_norm,
            realized_steer_rad=self.realized_steer_rad,
            map_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def warmup(self, iters: int) -> None:
        if iters <= 0:
            return
        with torch.inference_mode():
            for _ in range(iters):
                self._obs_device.normal_()
                out = self.actor.forward(
                    self._obs_device,
                    self.condition,
                    self._hidden,
                    deterministic=self.deterministic,
                )
                self._hidden.copy_(out.hidden)
                if self._use_cuda:
                    torch.cuda.synchronize(self.device)
        self.reset_hidden()

    def infer_host_obs(self) -> tuple[np.ndarray, InferenceTimings]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            self._obs_device[0].copy_(
                torch.as_tensor(self._obs_host_np, dtype=torch.float32)
            )
            h2d_ms = (time.perf_counter() - t0) * 1000.0
            infer_t0 = time.perf_counter()
            out = self.actor.forward(
                self._obs_device,
                self.condition,
                self._hidden,
                deterministic=self.deterministic,
            )
            self._hidden.copy_(out.hidden)
            if self._use_cuda:
                torch.cuda.synchronize(self.device)
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0
            d2h_t0 = time.perf_counter()
            self._action_device.copy_(out.actions)
            self._action_host[:] = self._action_device[0].detach().cpu().numpy()
            logp = getattr(out, "log_prob", None)
            if logp is None:
                self.last_log_prob = None
            else:
                self.last_log_prob = float(logp.reshape(-1)[0].detach().cpu())
            d2h_ms = (time.perf_counter() - d2h_t0) * 1000.0
        return self._action_host, InferenceTimings(
            h2d_ms=h2d_ms, infer_ms=infer_ms, d2h_ms=d2h_ms
        )

"""Preallocated recurrent inference runtime for sensor_racer."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.policy_model import ObsNormalizer, SquashedGaussianLidarGRUActor
from f1tenth_rl_agent.sensor_action import (
    integrate_steering_delta,
    normalized_action_to_physical,
)
from f1tenth_rl_agent.sensor_preprocessing import (
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    observation_is_finite,
    pack_actor_observation,
    pack_lidar_from_scan,
)


@dataclass(frozen=True)
class InferenceTimings:
    h2d_ms: float = 0.0
    infer_ms: float = 0.0
    d2h_ms: float = 0.0


@dataclass(frozen=True)
class RealizedActuatorCommand:
    drive_a: float
    brake_a: float
    servo: float
    long_norm: float
    steer_norm: float
    realized_steer_rad: float
    map_ms: float


class SensorInferenceRuntime:
    def __init__(
        self,
        actor: SquashedGaussianLidarGRUActor,
        normalizer: ObsNormalizer,
        device: torch.device,
        *,
        deterministic: bool = True,
        warmup_iters: int = 5,
        use_pinned_h2d: bool | None = None,
        use_compile: bool = False,
        steering_action_mode: str = "delta",
        steering_delta_max_rad: float = si.STEERING_DELTA_MAX_RAD,
        max_steer: float = si.MAX_STEER_RAD,
        i_drive_max_a: float = 40.0,
        i_brake_max_a: float = 40.0,
        steering_angle_to_servo_gain: float = 1.0,
        steering_angle_to_servo_offset: float = 0.5,
    ) -> None:
        self.actor = actor
        self.normalizer = normalizer
        self.device = device
        self.deterministic = bool(deterministic)
        self.steering_action_mode = str(steering_action_mode)
        self.steering_delta_max_rad = float(steering_delta_max_rad)
        self.max_steer = float(max_steer)
        self.i_drive_max_a = float(i_drive_max_a)
        self.i_brake_max_a = float(i_brake_max_a)
        self.steering_angle_to_servo_gain = float(steering_angle_to_servo_gain)
        self.steering_angle_to_servo_offset = float(
            steering_angle_to_servo_offset
        )
        self.realized_steer_rad = 0.0
        self._lidar_buf = np.full(si.LIDAR_DIM, si.LIDAR_RANGE_MAX, dtype=np.float32)
        self._steer_hist_view = np.zeros(4, dtype=np.float32)
        self._executed_steer_history = np.zeros(si.STEER_HISTORY, dtype=np.float32)
        self._prev_applied_long = 0.0
        self._action_host = np.zeros(si.NUM_ACTIONS, dtype=np.float32)
        self._hidden = actor.initial_hidden(1, device=device, dtype=torch.float32)
        self._obs_device = torch.zeros(
            (1, si.NUM_OBS), dtype=torch.float32, device=device
        )
        self._norm_device = torch.zeros(
            (1, si.NUM_OBS), dtype=torch.float32, device=device
        )
        self._action_device = torch.zeros(
            (1, si.NUM_ACTIONS), dtype=torch.float32, device=device
        )
        self._use_cuda = device.type == "cuda"
        self._stream: torch.cuda.Stream | None = None
        self._obs_host: torch.Tensor | None = None
        self._obs_host_np = np.zeros(si.NUM_OBS, dtype=np.float32)
        if self._use_cuda:
            if use_pinned_h2d is None:
                use_pinned_h2d = True
            if use_pinned_h2d:
                self._obs_host = torch.zeros(
                    (1, si.NUM_OBS), dtype=torch.float32, pin_memory=True
                )
                self._obs_host_np = self._obs_host.numpy().reshape(si.NUM_OBS)
            self._stream = torch.cuda.Stream(device=device)
        self._step_fn = self._make_step_fn(use_compile)
        self.warmup(warmup_iters)

    def _make_step_fn(self, use_compile: bool):
        def _step(obs, hidden):
            return self.actor.step(
                obs,
                hidden,
                reset_mask=None,
                deterministic=self.deterministic,
                with_logprob=False,
            )

        if not use_compile:
            return _step
        return torch.compile(_step, mode="default")

    @property
    def host_obs_buffer(self) -> np.ndarray:
        return self._obs_host_np

    @property
    def hidden(self) -> torch.Tensor:
        return self._hidden

    def reset_hidden(self) -> None:
        self._hidden.zero_()
        self.realized_steer_rad = 0.0
        self._prev_applied_long = 0.0
        self._executed_steer_history.fill(0.0)

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
        """Pack lidar/IMU/applied history into the host obs buffer."""
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
        hist = self._executed_steer_history
        self._steer_hist_view[0] = applied_steer_rad
        self._steer_hist_view[1] = hist[0]
        self._steer_hist_view[2] = hist[1]
        self._steer_hist_view[3] = hist[2]
        raw_obs = self._obs_host_np
        pack_actor_observation(
            self._lidar_buf,
            actor_imu,
            speed,
            vesc_current,
            applied_long,
            self._prev_applied_long,
            self._steer_hist_view,
            out=raw_obs,
        )
        finite = bool(observation_is_finite(raw_obs))
        if finite:
            hist[3] = hist[2]
            hist[2] = hist[1]
            hist[1] = hist[0]
            hist[0] = applied_steer_rad
            self._prev_applied_long = applied_long
        return raw_obs, actor_imu, raw_imu, finite

    def realize_action(self, action_np: np.ndarray) -> RealizedActuatorCommand:
        """Integrate delta steering and map to VESC/servo (delta-only)."""
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
                self.normalizer.normalize_into(self._obs_device, self._norm_device)
                action, _, next_hidden = self._step_fn(
                    self._norm_device, self._hidden
                )
                self._hidden.copy_(next_hidden)
                if self._use_cuda:
                    torch.cuda.synchronize(self.device)
                _ = action
        self.reset_hidden()

    def infer_host_obs(self) -> tuple[np.ndarray, InferenceTimings]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self._obs_host is not None and self._stream is not None:
                with torch.cuda.stream(self._stream):
                    self._obs_device.copy_(self._obs_host, non_blocking=True)
                self._stream.synchronize()
                h2d_ms = (time.perf_counter() - t0) * 1000.0
                self.normalizer.normalize_into(self._obs_device, self._norm_device)
            elif self._use_cuda:
                self._obs_device[0].copy_(
                    torch.as_tensor(self._obs_host_np, dtype=torch.float32)
                )
                if self._stream is not None:
                    self._stream.synchronize()
                h2d_ms = (time.perf_counter() - t0) * 1000.0
                self.normalizer.normalize_into(self._obs_device, self._norm_device)
            else:
                self._obs_device[0].copy_(
                    torch.as_tensor(self._obs_host_np, dtype=torch.float32)
                )
                h2d_ms = (time.perf_counter() - t0) * 1000.0
                self.normalizer.normalize_into(self._obs_device, self._norm_device)

            infer_t0 = time.perf_counter()
            action, _, next_hidden = self._step_fn(
                self._norm_device, self._hidden
            )
            self._hidden.copy_(next_hidden)
            if self._use_cuda:
                torch.cuda.synchronize(self.device)
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0

            d2h_t0 = time.perf_counter()
            self._action_device.copy_(action)
            self._action_host[:] = self._action_device[0].detach().cpu().numpy()
            d2h_ms = (time.perf_counter() - d2h_t0) * 1000.0

        return self._action_host, InferenceTimings(
            h2d_ms=h2d_ms, infer_ms=infer_ms, d2h_ms=d2h_ms
        )

    def infer_from_host_obs(
        self, raw_obs: np.ndarray
    ) -> tuple[np.ndarray, InferenceTimings]:
        np.copyto(self._obs_host_np, np.asarray(raw_obs, dtype=np.float32))
        return self.infer_host_obs()

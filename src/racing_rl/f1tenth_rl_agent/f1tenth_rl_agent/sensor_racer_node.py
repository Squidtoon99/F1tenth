"""Standalone 1,097-D recurrent sensor racer (10 Hz, no localization)."""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
import torch
from f1tenth_interfaces.msg import ActuatorCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32MultiArray

from f1tenth_policy import assert_artifact_current_limits_match

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.policy_model import (
    ObsNormalizer,
    SquashedGaussianLidarGRUActor,
    load_sensor_actor,
    load_sensor_obs_norm,
)
from f1tenth_rl_agent.sensor_inference_runtime import SensorInferenceRuntime
from f1tenth_rl_agent.sensor_preprocessing import ImuCalibration, RawImuSample


class SensorRacerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("sensor_racer", **kwargs)

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("require_checkpoint", True)
        self.declare_parameter("state_dict_key", "actor")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("control_hz", si.CONTROL_HZ)
        self.declare_parameter("norm_clip", si.OBS_NORM_CLIP)
        self.declare_parameter("norm_eps", si.OBS_NORM_EPS)
        self.declare_parameter("scan_topic", si.TOPIC_SCAN)
        self.declare_parameter("imu_topic", si.TOPIC_IMU)
        self.declare_parameter("odom_topic", si.TOPIC_ODOM)
        self.declare_parameter("applied_actuator_topic", si.TOPIC_APPLIED_ACTUATOR)
        self.declare_parameter("sensor_max_age_s", 0.15)
        self.declare_parameter("twist_vx_sign", 1.0)
        self.declare_parameter("i_drive_max_a", 80.0)
        self.declare_parameter("i_brake_max_a", 20.0)
        self.declare_parameter("max_steer", 0.33)
        self.declare_parameter("steering_action_mode", "delta")
        self.declare_parameter("steering_delta_max_rad", math.pi / 60.0)
        self.declare_parameter("steering_angle_to_servo_gain", -1.2135)
        self.declare_parameter("steering_angle_to_servo_offset", 0.4495)
        self.declare_parameter("imu_accel_to_ms2", 9.80665)
        self.declare_parameter("imu_gyro_to_rads", math.pi / 180.0)
        for _imu_key, _imu_default in (
            ("imu_ax_sign", 1.0),
            ("imu_ay_sign", 1.0),
            ("imu_az_sign", 1.0),
            ("imu_gx_sign", 1.0),
            ("imu_gy_sign", 1.0),
            ("imu_gz_sign", 1.0),
            ("imu_ax_bias", 0.0),
            ("imu_ay_bias", 0.0),
            ("imu_az_bias", 0.0),
            ("imu_gx_bias", 0.0),
            ("imu_gy_bias", 0.0),
            ("imu_gz_bias", 0.0),
        ):
            self.declare_parameter(_imu_key, _imu_default)
        self.declare_parameter("recording_mode", False)
        self.declare_parameter("publish_raw_observation", False)
        self.declare_parameter("publish_diagnostics", True)
        self.declare_parameter("inference_warmup_iters", 5)
        self.declare_parameter("use_torch_compile", False)

        gp = self.get_parameter
        checkpoint_path = gp("checkpoint_path").get_parameter_value().string_value
        self.require_checkpoint = gp("require_checkpoint").get_parameter_value().bool_value
        state_dict_key = gp("state_dict_key").get_parameter_value().string_value
        device_str = gp("device").get_parameter_value().string_value
        self.deterministic = gp("deterministic").get_parameter_value().bool_value
        self.control_hz = float(gp("control_hz").get_parameter_value().double_value)
        self.norm_clip = gp("norm_clip").get_parameter_value().double_value
        self.norm_eps = gp("norm_eps").get_parameter_value().double_value
        self.sensor_max_age_s = float(
            gp("sensor_max_age_s").get_parameter_value().double_value
        )
        self.twist_vx_sign = float(gp("twist_vx_sign").get_parameter_value().double_value)
        self.i_drive_max_a = float(gp("i_drive_max_a").get_parameter_value().double_value)
        self.i_brake_max_a = float(gp("i_brake_max_a").get_parameter_value().double_value)
        if (
            not math.isfinite(self.i_drive_max_a)
            or self.i_drive_max_a <= 0.0
            or not math.isfinite(self.i_brake_max_a)
            or self.i_brake_max_a <= 0.0
        ):
            raise ValueError(
                "i_drive_max_a and i_brake_max_a must be finite and > 0 "
                f"(got {self.i_drive_max_a!r}, {self.i_brake_max_a!r})"
            )
        self.max_steer = float(gp("max_steer").get_parameter_value().double_value)
        self.steering_action_mode = (
            gp("steering_action_mode").get_parameter_value().string_value
        )
        if self.steering_action_mode != "delta":
            raise ValueError(
                "steering_action_mode must be 'delta' (Lee/ADR-0011); got "
                f"{self.steering_action_mode!r}"
            )
        self.steering_delta_max_rad = float(
            gp("steering_delta_max_rad").get_parameter_value().double_value
        )
        if self.steering_delta_max_rad <= 0.0:
            raise ValueError("steering_delta_max_rad must be positive")
        self.steering_angle_to_servo_gain = float(
            gp("steering_angle_to_servo_gain").get_parameter_value().double_value
        )
        self.steering_angle_to_servo_offset = float(
            gp("steering_angle_to_servo_offset").get_parameter_value().double_value
        )
        self.recording_mode = gp("recording_mode").get_parameter_value().bool_value
        self.publish_raw_observation = (
            gp("publish_raw_observation").get_parameter_value().bool_value
        )
        self.publish_diagnostics = (
            gp("publish_diagnostics").get_parameter_value().bool_value
        )
        warmup_iters = int(gp("inference_warmup_iters").get_parameter_value().integer_value)
        use_torch_compile = gp("use_torch_compile").get_parameter_value().bool_value

        def _imu_f(name: str) -> float:
            return float(gp(name).get_parameter_value().double_value)

        self._imu_cal = ImuCalibration(
            accel_to_ms2=_imu_f("imu_accel_to_ms2"),
            gyro_to_rads=_imu_f("imu_gyro_to_rads"),
            ax_sign=_imu_f("imu_ax_sign"),
            ay_sign=_imu_f("imu_ay_sign"),
            az_sign=_imu_f("imu_az_sign"),
            gx_sign=_imu_f("imu_gx_sign"),
            gy_sign=_imu_f("imu_gy_sign"),
            gz_sign=_imu_f("imu_gz_sign"),
            ax_bias=_imu_f("imu_ax_bias"),
            ay_bias=_imu_f("imu_ay_bias"),
            az_bias=_imu_f("imu_az_bias"),
            gx_bias=_imu_f("imu_gx_bias"),
            gy_bias=_imu_f("imu_gy_bias"),
            gz_bias=_imu_f("imu_gz_bias"),
        )

        self.device = torch.device(device_str)
        self.actor, self.obs_normalizer = self._load_policy(
            checkpoint_path, state_dict_key
        )
        self._runtime: SensorInferenceRuntime | None = None
        if self.actor is not None and self.obs_normalizer is not None:
            self._runtime = SensorInferenceRuntime(
                self.actor,
                self.obs_normalizer,
                self.device,
                deterministic=self.deterministic,
                warmup_iters=warmup_iters,
                use_compile=use_torch_compile,
                steering_action_mode=self.steering_action_mode,
                steering_delta_max_rad=self.steering_delta_max_rad,
                max_steer=self.max_steer,
                i_drive_max_a=self.i_drive_max_a,
                i_brake_max_a=self.i_brake_max_a,
                steering_angle_to_servo_gain=self.steering_angle_to_servo_gain,
                steering_angle_to_servo_offset=self.steering_angle_to_servo_offset,
            )

        self._scan: LaserScan | None = None
        self._scan_time = None
        self._odom: Odometry | None = None
        self._odom_time = None
        self._applied: ActuatorCommand | None = None
        self._applied_time = None
        self._imu_interval: list[RawImuSample] = []
        self._imu_time = None
        self._last_tick_time = self.get_clock().now()
        self._deadline_misses = 0
        self._gru_resets = 0
        self._consecutive_safe_ticks = 0
        self._last_applied_source: int | None = None
        self._unusable_reset_active = False
        self._generation = 0
        self._diag_buf = np.zeros(si.DIAG_LEN, dtype=np.float32)
        self._cmd_msg = ActuatorCommand()
        self._diag_msg = Float32MultiArray()
        self._obs_msg = Float32MultiArray()

        scan_topic = gp("scan_topic").get_parameter_value().string_value
        imu_topic = gp("imu_topic").get_parameter_value().string_value
        odom_topic = gp("odom_topic").get_parameter_value().string_value
        applied_topic = gp("applied_actuator_topic").get_parameter_value().string_value

        self.create_subscription(
            LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu, imu_topic, self._on_imu, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(ActuatorCommand, applied_topic, self._on_applied, 10)

        self._desired_pub = self.create_publisher(
            ActuatorCommand, si.TOPIC_DESIRED_ACTUATOR, 10
        )
        if self.publish_diagnostics:
            self._diag_pub = self.create_publisher(
                Float32MultiArray, si.TOPIC_DIAGNOSTICS, 10
            )
        else:
            self._diag_pub = None
        if self.publish_raw_observation:
            self._obs_pub = self.create_publisher(
                Float32MultiArray, si.TOPIC_OBSERVATION, 10
            )
        else:
            self._obs_pub = None
        if self.recording_mode:
            self._imu_raw_pub = self.create_publisher(
                Float32MultiArray, si.TOPIC_IMU_RAW_RECORD, 10
            )
            self._imu_actor_pub = self.create_publisher(
                Float32MultiArray, si.TOPIC_IMU_ACTOR_RECORD, 10
            )
        else:
            self._imu_raw_pub = None
            self._imu_actor_pub = None

        period = 1.0 / max(self.control_hz, 1.0)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"sensor_racer {si.NUM_OBS}-D @ {self.control_hz:.1f} Hz "
            f"device={device_str} delta_max={self.steering_delta_max_rad:.6f}"
        )

    def _load_policy(
        self, checkpoint_path: str, state_dict_key: str
    ) -> tuple[SquashedGaussianLidarGRUActor | None, ObsNormalizer | None]:
        if not checkpoint_path:
            if self.require_checkpoint:
                raise RuntimeError(
                    "No checkpoint_path provided and require_checkpoint is true; "
                    "refusing to run a random-init sensor actor."
                )
            self.get_logger().warn(
                "No checkpoint_path provided; sensor racer will not infer actions."
            )
            return None, None
        try:
            actor = load_sensor_actor(
                checkpoint_path=checkpoint_path,
                state_dict_key=state_dict_key,
                device=self.device,
                expected_steering_action_mode=self.steering_action_mode,
                expected_steering_delta_max_rad=self.steering_delta_max_rad,
            )
            arch = actor.actor_architecture
            if arch.get("name") != si.ACTOR_ARCHITECTURE_NAME:
                raise ValueError(f"expected {si.ACTOR_ARCHITECTURE_NAME!r} actor")
            if int(arch.get("gru_hidden_dim", 0)) != si.GRU_HIDDEN_DIM:
                raise ValueError(f"expected gru_hidden_dim={si.GRU_HIDDEN_DIM}")
            normalizer = load_sensor_obs_norm(
                checkpoint_path=checkpoint_path,
                device=self.device,
                eps=self.norm_eps,
                clip=self.norm_clip,
                expected_steering_action_mode=self.steering_action_mode,
                expected_steering_delta_max_rad=self.steering_delta_max_rad,
            )
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            assert_artifact_current_limits_match(
                payload, self.i_drive_max_a, self.i_brake_max_a
            )
            self.get_logger().info(f"Loaded sensor racer checkpoint: {checkpoint_path}")
            return actor, normalizer
        except Exception as exc:  # noqa: BLE001
            if self.require_checkpoint:
                raise RuntimeError(
                    f"Failed to load sensor checkpoint '{checkpoint_path}': {exc}"
                ) from exc
            self.get_logger().error(
                f"Failed to load sensor checkpoint '{checkpoint_path}': {exc}"
            )
            return None, None

    def _stamp_age_s(self, stamp_time) -> float:
        if stamp_time is None:
            return float("inf")
        return (self.get_clock().now() - stamp_time).nanoseconds * 1e-9

    def _applied_source(self) -> float:
        if self._applied is None:
            return float("nan")
        return float(self._applied.source)

    def _applied_values_finite(self) -> bool:
        if self._applied is None:
            return False
        values = (
            self._applied.longitudinal,
            self._applied.steering,
            self._applied.drive_current_a,
            self._applied.brake_current_a,
            self._applied.servo_position,
        )
        return all(math.isfinite(float(v)) for v in values)

    def _reset_runtime(self, reason: int) -> None:
        if self._runtime is not None:
            self._runtime.reset_hidden()
        self._gru_resets += 1
        self._diag_buf[si.DIAG_GRU_RESET] = 1.0
        self._diag_buf[si.DIAG_GRU_RESET_REASON] = float(reason)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_time = self.get_clock().now()

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg
        self._odom_time = self.get_clock().now()

    def _on_applied(self, msg: ActuatorCommand) -> None:
        self._applied = msg
        self._applied_time = self.get_clock().now()

    def _on_imu(self, msg: Imu) -> None:
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        sample = RawImuSample(
            stamp_s=stamp_s,
            ax=float(msg.linear_acceleration.x),
            ay=float(msg.linear_acceleration.y),
            az=float(msg.linear_acceleration.z),
            gx=float(msg.angular_velocity.x),
            gy=float(msg.angular_velocity.y),
            gz=float(msg.angular_velocity.z),
        )
        self._imu_interval.append(sample)
        self._imu_time = self.get_clock().now()

    def _sensors_fresh(self) -> bool:
        ages = (
            self._stamp_age_s(self._scan_time),
            self._stamp_age_s(self._odom_time),
            self._stamp_age_s(self._applied_time),
            self._stamp_age_s(self._imu_time),
        )
        return all(age <= self.sensor_max_age_s for age in ages)

    def _applied_blocks_rl_publish(self) -> bool:
        if self._applied is None:
            return True
        return self._applied.source in (
            ActuatorCommand.SOURCE_TELEOP,
            ActuatorCommand.SOURCE_SAFETY,
        )

    def _ownership_reset_reason(self, *, deadline_miss: bool) -> int:
        if self._applied is None or not self._applied_values_finite():
            return si.GRU_RESET_APPLIED_INVALID
        source = int(self._applied.source)
        source_changed = source != self._last_applied_source
        self._last_applied_source = source
        if deadline_miss:
            return si.GRU_RESET_DEADLINE_MISS
        if source == ActuatorCommand.SOURCE_RL or not source_changed:
            return si.GRU_RESET_NONE
        return {
            ActuatorCommand.SOURCE_SAFE: si.GRU_RESET_APPLIED_SAFE,
            ActuatorCommand.SOURCE_TELEOP: si.GRU_RESET_APPLIED_TELEOP,
            ActuatorCommand.SOURCE_SAFETY: si.GRU_RESET_APPLIED_SAFETY,
        }.get(source, si.GRU_RESET_APPLIED_INVALID)

    def _tick_unusable(self) -> bool:
        return (
            self._scan is None
            or self._odom is None
            or self._applied is None
            or not self._imu_interval
            or not self._sensors_fresh()
            or self._runtime is None
        )

    def _begin_diag(self, *, deadline_miss: bool) -> None:
        diag = self._diag_buf
        diag.fill(0.0)
        diag[si.DIAG_DEADLINE_MISS] = 1.0 if deadline_miss else 0.0
        diag[si.DIAG_SCAN_AGE_S] = self._stamp_age_s(self._scan_time)
        diag[si.DIAG_IMU_AGE_S] = self._stamp_age_s(self._imu_time)
        diag[si.DIAG_ODOM_AGE_S] = self._stamp_age_s(self._odom_time)
        diag[si.DIAG_APPLIED_AGE_S] = self._stamp_age_s(self._applied_time)
        diag[si.DIAG_APPLIED_SOURCE] = self._applied_source()
        safe = (
            self._applied is not None
            and self._applied.source == ActuatorCommand.SOURCE_SAFE
        )
        self._consecutive_safe_ticks = self._consecutive_safe_ticks + 1 if safe else 0
        diag[si.DIAG_CONSECUTIVE_SAFE] = float(self._consecutive_safe_ticks)
        diag[si.DIAG_GRU_RESETS] = float(self._gru_resets)

    def _finish_tick(self, now, **timings) -> None:
        diag = self._diag_buf
        for key, idx in (
            ("preprocess_ms", si.DIAG_PREPROCESS_MS),
            ("h2d_ms", si.DIAG_H2D_MS),
            ("infer_ms", si.DIAG_INFER_MS),
            ("d2h_ms", si.DIAG_D2H_MS),
            ("map_ms", si.DIAG_MAP_MS),
            ("publish_ms", si.DIAG_PUBLISH_MS),
            ("total_ms", si.DIAG_TOTAL_MS),
            ("stale_sensors", si.DIAG_STALE_SENSORS),
            ("valid_tick", si.DIAG_VALID_TICK),
        ):
            if key in timings:
                diag[idx] = float(timings[key])
        if "gru_reset" in timings and timings["gru_reset"] is not None:
            diag[si.DIAG_GRU_RESET] = float(timings["gru_reset"])
        diag[si.DIAG_GRU_RESETS] = float(self._gru_resets)
        self._publish_diagnostics()
        self._imu_interval.clear()
        self._last_tick_time = now

    def _on_timer(self) -> None:
        t0 = time.perf_counter()
        now = self.get_clock().now()
        period_s = 1.0 / max(self.control_hz, 1.0)
        elapsed = (now - self._last_tick_time).nanoseconds * 1e-9
        deadline_miss = elapsed > (1.25 * period_s)
        if deadline_miss:
            self._deadline_misses += 1

        self._begin_diag(deadline_miss=deadline_miss)
        unusable = self._tick_unusable()
        if unusable:
            if not self._unusable_reset_active:
                self._reset_runtime(si.GRU_RESET_UNUSABLE_INPUT)
                self._unusable_reset_active = True
            total_ms = (time.perf_counter() - t0) * 1000.0
            self._finish_tick(
                now,
                preprocess_ms=total_ms,
                total_ms=total_ms,
                stale_sensors=1.0,
            )
            return
        self._unusable_reset_active = False

        reset_reason = self._ownership_reset_reason(deadline_miss=deadline_miss)
        if reset_reason != si.GRU_RESET_NONE:
            self._reset_runtime(reset_reason)

        assert self._scan is not None
        assert self._odom is not None
        assert self._applied is not None
        assert self._runtime is not None

        speed = self.twist_vx_sign * float(self._odom.twist.twist.linear.x)
        applied_long = float(self._applied.longitudinal)
        applied_steer = float(self._applied.steering) * self.max_steer
        signed_current_a = float(
            self._applied.drive_current_a - self._applied.brake_current_a
        )
        vesc_current = si.applied_current_fraction(
            signed_current_a, self.i_drive_max_a, self.i_brake_max_a
        )
        raw_obs, actor_imu, raw_imu, finite = self._runtime.pack_observation(
            scan_angle_min=float(self._scan.angle_min),
            scan_angle_increment=float(self._scan.angle_increment),
            scan_ranges=self._scan.ranges,
            scan_range_min=float(self._scan.range_min),
            imu_interval=self._imu_interval,
            imu_cal=self._imu_cal,
            speed=speed,
            applied_long=applied_long,
            applied_steer_rad=applied_steer,
            vesc_current=vesc_current,
        )
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        if not finite:
            self._reset_runtime(si.GRU_RESET_OBSERVATION_NONFINITE)
            total_ms = (time.perf_counter() - t0) * 1000.0
            self._finish_tick(
                now,
                preprocess_ms=preprocess_ms,
                total_ms=total_ms,
            )
            return

        if self._obs_pub is not None:
            self._obs_msg.data = raw_obs.tolist()
            self._obs_pub.publish(self._obs_msg)
        if self.recording_mode and self._imu_raw_pub is not None and self._imu_actor_pub:
            raw_msg = Float32MultiArray()
            raw_msg.data = raw_imu.astype(np.float32).tolist()
            self._imu_raw_pub.publish(raw_msg)
            actor_msg = Float32MultiArray()
            actor_msg.data = actor_imu.astype(np.float32).tolist()
            self._imu_actor_pub.publish(actor_msg)

        action_np, infer_timings = self._runtime.infer_host_obs()
        np.clip(action_np, -si.CLIP_ACTIONS, si.CLIP_ACTIONS, out=action_np)
        if not bool(np.isfinite(action_np).all()):
            self._reset_runtime(si.GRU_RESET_ACTION_NONFINITE)
            total_ms = (time.perf_counter() - t0) * 1000.0
            self._finish_tick(
                now,
                preprocess_ms=preprocess_ms,
                h2d_ms=infer_timings.h2d_ms,
                infer_ms=infer_timings.infer_ms,
                d2h_ms=infer_timings.d2h_ms,
                total_ms=total_ms,
            )
            return

        if self._applied_blocks_rl_publish():
            total_ms = (time.perf_counter() - t0) * 1000.0
            self._finish_tick(
                now,
                preprocess_ms=preprocess_ms,
                h2d_ms=infer_timings.h2d_ms,
                infer_ms=infer_timings.infer_ms,
                d2h_ms=infer_timings.d2h_ms,
                total_ms=total_ms,
            )
            return

        realized = self._runtime.realize_action(action_np)
        drive_a = realized.drive_a
        brake_a = realized.brake_a
        servo = realized.servo
        long_norm = realized.long_norm
        steer_norm = realized.steer_norm
        map_ms = realized.map_ms

        if drive_a > 0.0 and brake_a > 0.0:
            self._reset_runtime(si.GRU_RESET_DUAL_CURRENT)
            total_ms = (time.perf_counter() - t0) * 1000.0
            self._finish_tick(
                now,
                preprocess_ms=preprocess_ms,
                h2d_ms=infer_timings.h2d_ms,
                infer_ms=infer_timings.infer_ms,
                d2h_ms=infer_timings.d2h_ms,
                map_ms=map_ms,
                total_ms=total_ms,
            )
            return

        pub_t0 = time.perf_counter()
        self._generation += 1
        cmd = self._cmd_msg
        cmd.header.stamp = now.to_msg()
        cmd.generation = self._generation
        cmd.observation_stamp = self._scan.header.stamp
        cmd.drive_current_a = drive_a
        cmd.brake_current_a = brake_a
        cmd.servo_position = servo
        cmd.longitudinal = float(long_norm)
        cmd.steering = float(steer_norm)
        cmd.source = ActuatorCommand.SOURCE_RL
        self._desired_pub.publish(cmd)
        publish_ms = (time.perf_counter() - pub_t0) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        self._finish_tick(
            now,
            preprocess_ms=preprocess_ms,
            h2d_ms=infer_timings.h2d_ms,
            infer_ms=infer_timings.infer_ms,
            d2h_ms=infer_timings.d2h_ms,
            map_ms=map_ms,
            publish_ms=publish_ms,
            total_ms=total_ms,
            valid_tick=1.0,
        )

    def _publish_diagnostics(self) -> None:
        if self._diag_pub is None:
            return
        self._diag_msg.data = self._diag_buf.astype(float).tolist()
        self._diag_pub.publish(self._diag_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorRacerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

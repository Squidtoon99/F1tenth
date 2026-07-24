"""Shared rosbag2 reader for offline calibration (macOS-friendly, no rclpy)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

ACKERMANN_DRIVE = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""
ACKERMANN_DRIVE_STAMPED = """
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""
VESC_STATE = """
int32 FAULT_CODE_NONE=0
int32 FAULT_CODE_OVER_VOLTAGE=1
int32 FAULT_CODE_UNDER_VOLTAGE=2
int32 FAULT_CODE_DRV8302=3
int32 FAULT_CODE_ABS_OVER_CURRENT=4
int32 FAULT_CODE_OVER_TEMP_FET=5
int32 FAULT_CODE_OVER_TEMP_MOTOR=6
float64 temp_fet
float64 temp_motor
float64 current_motor
float64 current_input
float64 avg_id
float64 avg_iq
float64 duty_cycle
float64 speed
float64 voltage_input
float64 charge_drawn
float64 charge_regen
float64 energy_drawn
float64 energy_regen
int32 displacement
int32 distance_traveled
int32 fault_code
float64 pid_pos_now
int32 controller_id
float64 ntc_temp_mos1
float64 ntc_temp_mos2
float64 ntc_temp_mos3
float64 avg_vd
float64 avg_vq
"""
VESC_STATE_STAMPED = """
std_msgs/Header header
vesc_msgs/VescState state
"""
ACTUATOR_COMMAND = """
std_msgs/Header header
uint64 generation
builtin_interfaces/Time observation_stamp
float64 drive_current_a
float64 brake_current_a
float64 servo_position
float32 longitudinal
float32 steering
uint8 SOURCE_SAFE=0
uint8 SOURCE_RL=1
uint8 SOURCE_TELEOP=2
uint8 SOURCE_SAFETY=3
uint8 source
"""

TOPIC_IMU_RAW_RECORD = "/sensor_policy/imu_raw_record"
TOPIC_IMU_ACTOR_RECORD = "/sensor_policy/imu_actor_record"

# Topics used by the calibration fits.
DEFAULT_TOPICS = (
    "/odom",
    "/sensors/imu/raw",
    TOPIC_IMU_RAW_RECORD,
    TOPIC_IMU_ACTOR_RECORD,
    "/pf/pose/odom",
    "/ackermann_cmd",
    "/teleop",
    "/drive",
    "/commands/motor/speed",
    "/commands/motor/current",
    "/commands/motor/brake",
    "/commands/servo/position",
    "/sensors/servo_position_command",
    "/sensors/core",
)


def build_typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(get_types_from_msg(ACKERMANN_DRIVE, "ackermann_msgs/msg/AckermannDrive"))
    types.update(
        get_types_from_msg(
            ACKERMANN_DRIVE_STAMPED, "ackermann_msgs/msg/AckermannDriveStamped"
        )
    )
    types.update(get_types_from_msg(VESC_STATE, "vesc_msgs/msg/VescState"))
    types.update(
        get_types_from_msg(VESC_STATE_STAMPED, "vesc_msgs/msg/VescStateStamped")
    )
    types.update(
        get_types_from_msg(ACTUATOR_COMMAND, "f1tenth_interfaces/msg/ActuatorCommand")
    )
    ts.register(types)
    return ts


G = 9.81


def detect_accel_scale(az: float) -> tuple[float, str]:
    """Infer the IMU accel unit from a stationary z-axis reading.

    ``|az| ~ 1`` means g units (scale to m/s^2); ``|az| ~ 9.81`` is already
    m/s^2. Anything else is treated as unknown and normalized against gravity.
    """
    az_abs = abs(float(az))
    if 0.5 < az_abs < 1.5:
        return G, "g"
    if 5.0 < az_abs < 15.0:
        return 1.0, "m_s2"
    return G / max(az_abs, 1e-6), "unknown"


def yaw_from_quat(z: float, w: float) -> float:
    return float(np.arctan2(2.0 * w * z, 1.0 - 2.0 * z * z))


def load_series(
    bag_dir: Path | str,
    topics: Iterable[str] | None = None,
    typestore=None,
) -> dict[str, np.ndarray]:
    """Load selected topics into arrays with columns ``[t_sec, ...]``.

    Column layouts:
      /odom                         -> t, vx, yaw_rate
      /sensors/imu/raw              -> t, ax, ay, az, gx, gy, gz
      /sensor_policy/imu_*_record   -> t, ax, ay, az, gx, gy, gz
      /pf/pose/odom                 -> t, x, y, yaw
      /ackermann_cmd|/teleop|/drive -> t, speed, steering, acceleration
      /commands/motor/speed|current|brake
      /commands/servo/position|/sensors/servo_position_command -> t, value
      /sensors/core -> t, temp_fet, temp_motor, current_motor, current_input,
                       avg_id, avg_iq, duty, erpm, voltage, charge_drawn,
                       charge_regen, energy_drawn, energy_regen, fault, avg_vd, avg_vq
    """
    bag_dir = Path(bag_dir)
    wanted = set(topics if topics is not None else DEFAULT_TOPICS)
    ts = typestore or build_typestore()

    odom, imu, pf = [], [], []
    imu_raw_record, imu_actor_record = [], []
    ackermann, teleop, drive = [], [], []
    motor_speed, motor_current, motor_brake = [], [], []
    servo, servo_cmd = [], []
    core = []

    with AnyReader([bag_dir], default_typestore=ts) as reader:
        for conn, t_ns, raw in reader.messages():
            if conn.topic not in wanted:
                continue
            t = t_ns * 1e-9
            m = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/odom":
                odom.append((t, m.twist.twist.linear.x, m.twist.twist.angular.z))
            elif conn.topic == "/sensors/imu/raw":
                a = m.linear_acceleration
                g = m.angular_velocity
                imu.append((t, a.x, a.y, a.z, g.x, g.y, g.z))
            elif conn.topic in (TOPIC_IMU_RAW_RECORD, TOPIC_IMU_ACTOR_RECORD):
                data = list(m.data)
                if len(data) < 6:
                    continue
                row = (t, *data[:6])
                if conn.topic == TOPIC_IMU_RAW_RECORD:
                    imu_raw_record.append(row)
                else:
                    imu_actor_record.append(row)
            elif conn.topic == "/pf/pose/odom":
                p = m.pose.pose
                pf.append(
                    (
                        t,
                        p.position.x,
                        p.position.y,
                        yaw_from_quat(p.orientation.z, p.orientation.w),
                    )
                )
            elif conn.topic in ("/ackermann_cmd", "/teleop", "/drive"):
                row = (t, m.drive.speed, m.drive.steering_angle, m.drive.acceleration)
                if conn.topic == "/ackermann_cmd":
                    ackermann.append(row)
                elif conn.topic == "/teleop":
                    teleop.append(row)
                else:
                    drive.append(row)
            elif conn.topic == "/commands/motor/speed":
                motor_speed.append((t, float(m.data)))
            elif conn.topic == "/commands/motor/current":
                motor_current.append((t, float(m.data)))
            elif conn.topic == "/commands/motor/brake":
                motor_brake.append((t, float(m.data)))
            elif conn.topic == "/commands/servo/position":
                servo.append((t, float(m.data)))
            elif conn.topic == "/sensors/servo_position_command":
                servo_cmd.append((t, float(m.data)))
            elif conn.topic == "/sensors/core":
                s = m.state
                core.append(
                    (
                        t,
                        s.temp_fet,
                        s.temp_motor,
                        s.current_motor,
                        s.current_input,
                        s.avg_id,
                        s.avg_iq,
                        s.duty_cycle,
                        s.speed,
                        s.voltage_input,
                        s.charge_drawn,
                        s.charge_regen,
                        s.energy_drawn,
                        s.energy_regen,
                        s.fault_code,
                        s.avg_vd,
                        s.avg_vq,
                    )
                )

    def arr(rows, ncols):
        if not rows:
            return np.zeros((0, ncols), dtype=float)
        return np.asarray(rows, dtype=float)

    return {
        "/odom": arr(odom, 3),
        "/sensors/imu/raw": arr(imu, 7),
        TOPIC_IMU_RAW_RECORD: arr(imu_raw_record, 7),
        TOPIC_IMU_ACTOR_RECORD: arr(imu_actor_record, 7),
        "/pf/pose/odom": arr(pf, 4),
        "/ackermann_cmd": arr(ackermann, 4),
        "/teleop": arr(teleop, 4),
        "/drive": arr(drive, 4),
        "/commands/motor/speed": arr(motor_speed, 2),
        "/commands/motor/current": arr(motor_current, 2),
        "/commands/motor/brake": arr(motor_brake, 2),
        "/commands/servo/position": arr(servo, 2),
        "/sensors/servo_position_command": arr(servo_cmd, 2),
        "/sensors/core": arr(core, 17),
    }


def load_sensor_policy_series(
    bag_dir: Path | str,
    typestore=None,
) -> dict[str, np.ndarray]:
    """Load sensor-policy replay topics without requiring a ROS installation.

    Scan columns are ``t, header_t, angle_min, angle_increment, range_min,
    range_max, ranges...``. Actuator columns are ``t, generation,
    observation_t, drive_a, brake_a, servo, longitudinal, steering, source``.
    Other variable-length array topics use ``t, data...``.
    """
    bag_dir = Path(bag_dir)
    ts = typestore or build_typestore()
    rows = {
        "/scan": [],
        "/sensors/imu/raw": [],
        "/odom": [],
        "/rl/actuator/desired": [],
        "/rl/actuator/applied": [],
        "/sensor_racer/diagnostics": [],
        "/sensor_racer/observation": [],
    }

    with AnyReader([bag_dir], default_typestore=ts) as reader:
        for conn, t_ns, raw in reader.messages():
            if conn.topic not in rows:
                continue
            t = t_ns * 1e-9
            m = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/scan":
                header_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                rows[conn.topic].append(
                    (
                        t,
                        header_t,
                        m.angle_min,
                        m.angle_increment,
                        m.range_min,
                        m.range_max,
                        *m.ranges,
                    )
                )
            elif conn.topic == "/sensors/imu/raw":
                a = m.linear_acceleration
                g = m.angular_velocity
                rows[conn.topic].append((t, a.x, a.y, a.z, g.x, g.y, g.z))
            elif conn.topic == "/odom":
                rows[conn.topic].append((t, m.twist.twist.linear.x))
            elif conn.topic in ("/rl/actuator/desired", "/rl/actuator/applied"):
                observation_t = (
                    m.observation_stamp.sec + m.observation_stamp.nanosec * 1e-9
                )
                rows[conn.topic].append(
                    (
                        t,
                        m.generation,
                        observation_t,
                        m.drive_current_a,
                        m.brake_current_a,
                        m.servo_position,
                        m.longitudinal,
                        m.steering,
                        m.source,
                    )
                )
            else:
                rows[conn.topic].append((t, *m.data))

    widths = {
        "/scan": 6,
        "/sensors/imu/raw": 7,
        "/odom": 2,
        "/rl/actuator/desired": 9,
        "/rl/actuator/applied": 9,
        "/sensor_racer/diagnostics": 1,
        "/sensor_racer/observation": 1,
    }
    return {
        topic: (
            np.asarray(values, dtype=float)
            if values
            else np.zeros((0, widths[topic]), dtype=float)
        )
        for topic, values in rows.items()
    }


def inventory(bag_dir: Path | str, series: dict[str, np.ndarray] | None = None) -> dict:
    """Return duration, counts, and approximate rates for a bag."""
    bag_dir = Path(bag_dir)
    series = series if series is not None else load_series(bag_dir)
    times = []
    topics = {}
    for name, data in series.items():
        n = int(data.shape[0])
        if n == 0:
            topics[name] = {"count": 0, "rate_hz": 0.0, "t0": None, "t1": None}
            continue
        t0, t1 = float(data[0, 0]), float(data[-1, 0])
        dt = max(t1 - t0, 1e-9)
        topics[name] = {
            "count": n,
            "rate_hz": float((n - 1) / dt) if n > 1 else 0.0,
            "t0": t0,
            "t1": t1,
        }
        times.extend([t0, t1])
    duration = float(max(times) - min(times)) if times else 0.0
    return {
        "bag": bag_dir.name,
        "path": str(bag_dir),
        "duration_s": duration,
        "topics": topics,
        "has_sensors_core": series["/sensors/core"].shape[0] > 0,
    }


def resample(series: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """Linearly resample columns 1.. of ``series`` onto ``t_ref``."""
    if series.size == 0 or t_ref.size == 0:
        return np.zeros((t_ref.size, max(series.shape[1] if series.ndim == 2 else 1, 1)))
    out = np.zeros((t_ref.size, series.shape[1]), dtype=float)
    out[:, 0] = t_ref
    for c in range(1, series.shape[1]):
        out[:, c] = np.interp(t_ref, series[:, 0], series[:, c])
    return out


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return ``(slope, intercept, r2)`` for y ≈ slope*x + intercept."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2:
        return 0.0, 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(slope), float(intercept), float(r2)

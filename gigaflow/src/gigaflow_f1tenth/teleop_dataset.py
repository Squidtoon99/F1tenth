"""Build 10 Hz Gigaflow (obs, action) windows from manual / teleop rosbags."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gigaflow_f1tenth.irl_identity import IRL_STEERING_DELTA_MAX_RAD
from gigaflow_f1tenth.sim.layout_local import (
    GRAVITY,
    IMU_DIM,
    IMU_START,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    SENSOR_OBS_DIM,
    STEER_DELTA0,
    STEER_DELTA1,
    STEER_DELTA2,
    STEER_HISTORY,
    STEER_T,
    STEER_T1,
    STEER_T2,
    THROTTLE_CURRENT,
    THROTTLE_PRED,
    VESC_CURRENT,
    VESC_CURRENT_SCALE_A,
    VESC_SPEED,
)

CONTROL_HZ = 10.0
CONTROL_DT = 1.0 / CONTROL_HZ
ODOM_VX_SIGN = -1.0
STATIONARY_VX = 0.2
MIN_WINDOW_S = 1.0
DEFAULT_BAG_ROOTS = (
    Path("/home/ubuntu/f1tenth-bags/courtyard_arjun_day"),
    Path("/home/ubuntu/f1tenth-bags/skidpad_2026-09-21"),
)
SKIP_BAG_NAMES = {
    "rl_auton_20260622_191834",
    "gap_follow_2",
    "autonomous_policy_3",
}
SKIP_BAG_PREFIXES = (
    "autonomous_policy",
    "gap_follow",
    "gengar_model",
    "sample_policy",
    "rl_auton",
)

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


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def resample_lidar(
    angle_min: float,
    angle_increment: float,
    ranges: Sequence[float],
    out: np.ndarray | None = None,
) -> np.ndarray:
    dest = np.zeros(LIDAR_DIM, dtype=np.float32) if out is None else out
    src = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if src.size < 2 or not np.isfinite(angle_increment) or angle_increment == 0.0:
        dest.fill(LIDAR_RANGE_MAX)
        return dest
    src_angles = float(angle_min) + np.arange(src.size, dtype=np.float64) * float(
        angle_increment
    )
    dst_angles = LIDAR_ANGLE_MIN + np.arange(LIDAR_DIM, dtype=np.float64) * (
        LIDAR_ANGLE_INCREMENT
    )
    finite = np.isfinite(src)
    filled = np.where(finite, src, LIDAR_RANGE_MAX)
    filled = np.clip(filled, LIDAR_RANGE_MIN, LIDAR_RANGE_MAX)
    dest[:] = np.interp(dst_angles, src_angles, filled).astype(np.float32)
    return dest


def pack_step(
    lidar: np.ndarray,
    vx: float,
    vy: float,
    yaw_rate: float,
    vx_prev: float | None,
    vy_prev: float | None,
    long_current: float,
    long_pred: float,
    steer_hist: np.ndarray,
    dt: float = CONTROL_DT,
    out: np.ndarray | None = None,
) -> np.ndarray:
    obs = np.zeros(SENSOR_OBS_DIM, dtype=np.float32) if out is None else out
    obs[:LIDAR_DIM] = np.asarray(lidar, dtype=np.float32).reshape(-1)[:LIDAR_DIM]
    if vx_prev is None or vy_prev is None or dt <= 0.0:
        ax = ay = 0.0
    else:
        inv_dt = 1.0 / float(dt)
        ax = (float(vx) - float(vx_prev)) * inv_dt - float(yaw_rate) * float(vy)
        ay = (float(vy) - float(vy_prev)) * inv_dt + float(yaw_rate) * float(vx)
    obs[IMU_START : IMU_START + IMU_DIM] = (
        ax,
        ay,
        GRAVITY,
        0.0,
        0.0,
        float(yaw_rate),
    )
    hist = np.asarray(steer_hist, dtype=np.float32).reshape(-1)
    if hist.size < 4:
        raise ValueError("steer history must have at least 4 entries")
    obs[VESC_SPEED] = float(vx)
    obs[VESC_CURRENT] = VESC_CURRENT_SCALE_A * float(long_current)
    obs[THROTTLE_CURRENT] = float(long_current)
    obs[THROTTLE_PRED] = float(long_pred)
    obs[STEER_T] = hist[0]
    obs[STEER_T1] = hist[1]
    obs[STEER_T2] = hist[2]
    obs[STEER_DELTA0] = hist[0] - hist[1]
    obs[STEER_DELTA1] = hist[1] - hist[2]
    obs[STEER_DELTA2] = hist[2] - hist[3]
    return obs


def teleop_to_action(
    acceleration: float,
    steering_angle: float,
    prev_steer: float,
    delta_max: float = IRL_STEERING_DELTA_MAX_RAD,
) -> tuple[np.ndarray, float]:
    force = _clip(float(acceleration), -1.0, 1.0)
    if delta_max <= 0.0:
        raise ValueError("delta_max must be positive")
    delta = _clip((float(steering_angle) - float(prev_steer)) / delta_max, -1.0, 1.0)
    steer = float(steering_angle)
    return np.asarray([force, delta], dtype=np.float32), steer


def synthetic_trajectories(
    *,
    n_traj: int = 2,
    length: int = 32,
    seed: int = 0,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, np.ndarray]] = []
    for _ in range(int(n_traj)):
        t = int(length)
        obs = rng.normal(0.0, 0.1, size=(t, SENSOR_OBS_DIM)).astype(np.float32)
        obs[:, :LIDAR_DIM] = np.clip(3.0 + obs[:, :LIDAR_DIM], 0.06, 30.0)
        actions = rng.uniform(-0.4, 0.4, size=(t, 2)).astype(np.float32)
        vx = np.clip(1.0 + np.cumsum(actions[:, 0]) * 0.05, 0.0, 6.0)
        rewards = (vx * CONTROL_DT).astype(np.float32)
        dones = np.zeros(t, dtype=np.float32)
        dones[-1] = 1.0
        resets = np.zeros(t, dtype=np.float32)
        resets[0] = 1.0
        out.append(
            {
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "resets": resets,
            }
        )
    return out


def discover_manual_bags(roots: Iterable[Path] | None = None) -> list[Path]:
    found: list[Path] = []
    for root in roots or DEFAULT_BAG_ROOTS:
        root = Path(root)
        if not root.exists():
            continue
        for meta in sorted(root.rglob("metadata.yaml")):
            bag = meta.parent
            if bag.name in SKIP_BAG_NAMES:
                continue
            if any(bag.name.startswith(prefix) for prefix in SKIP_BAG_PREFIXES):
                continue
            found.append(bag)
    return found


def _typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(
        get_types_from_msg(ACKERMANN_DRIVE, "ackermann_msgs/msg/AckermannDrive")
    )
    types.update(
        get_types_from_msg(
            ACKERMANN_DRIVE_STAMPED, "ackermann_msgs/msg/AckermannDriveStamped"
        )
    )
    ts.register(types)
    return ts


def _stamp_s(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def load_bag_trajectory(
    bag_dir: Path,
    *,
    odom_vx_sign: float = ODOM_VX_SIGN,
) -> dict[str, np.ndarray] | None:
    from rosbags.highlevel import AnyReader

    bag_dir = Path(bag_dir)
    scans: list[tuple[float, np.ndarray]] = []
    odoms: list[tuple[float, float, float, float]] = []
    teleops: list[tuple[float, float, float, float]] = []
    with AnyReader([bag_dir], default_typestore=_typestore()) as reader:
        connections = [c for c in reader.connections if c.topic in {"/scan", "/odom", "/teleop"}]
        if not connections:
            return None
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            t = float(timestamp) * 1e-9
            if conn.topic == "/scan":
                lidar = resample_lidar(msg.angle_min, msg.angle_increment, msg.ranges)
                scans.append((t, lidar.copy()))
            elif conn.topic == "/odom":
                twist = msg.twist.twist
                odoms.append(
                    (
                        t,
                        odom_vx_sign * float(twist.linear.x),
                        float(twist.linear.y),
                        float(twist.angular.z),
                    )
                )
            elif conn.topic == "/teleop":
                drive = msg.drive
                teleops.append(
                    (
                        t,
                        float(drive.acceleration),
                        float(drive.steering_angle),
                        float(drive.speed),
                    )
                )
    if not scans or not odoms or not teleops:
        return None
    t0 = max(scans[0][0], odoms[0][0], teleops[0][0])
    t1 = min(scans[-1][0], odoms[-1][0], teleops[-1][0])
    if t1 - t0 < MIN_WINDOW_S:
        return None
    times = np.arange(t0, t1, CONTROL_DT, dtype=np.float64)
    scan_t = np.asarray([s[0] for s in scans])
    odom_t = np.asarray([o[0] for o in odoms])
    tele_t = np.asarray([x[0] for x in teleops])

    def _hold(src_t, src_rows, idx):
        j = np.clip(
            np.searchsorted(src_t, times, side="right") - 1, 0, len(src_rows) - 1
        )
        return np.stack([src_rows[int(k)][idx] for k in np.atleast_1d(j)], axis=0)

    lidar_idx = np.clip(np.searchsorted(scan_t, times, side="right") - 1, 0, len(scans) - 1)
    lidar = np.stack([scans[int(i)][1] for i in lidar_idx])
    vx = _hold(odom_t, odoms, 1).astype(np.float32)
    vy = _hold(odom_t, odoms, 2).astype(np.float32)
    yaw_rate = _hold(odom_t, odoms, 3).astype(np.float32)
    accel = _hold(tele_t, teleops, 1).astype(np.float32)
    steer = _hold(tele_t, teleops, 2).astype(np.float32)
    speed_cmd = _hold(tele_t, teleops, 3).astype(np.float32)
    if np.max(np.abs(speed_cmd)) > 0.05 and np.max(np.abs(accel)) < 1e-3:
        return None

    moving = np.abs(vx) > STATIONARY_VX
    if int(moving.sum()) < int(MIN_WINDOW_S * CONTROL_HZ):
        return None

    n = times.size
    obs = np.zeros((n, SENSOR_OBS_DIM), dtype=np.float32)
    actions = np.zeros((n, 2), dtype=np.float32)
    steer_hist = np.zeros(STEER_HISTORY, dtype=np.float32)
    long_current = 0.0
    long_pred = 0.0
    prev_steer = float(steer[0])
    vx_prev = vy_prev = None
    for i in range(n):
        pack_step(
            lidar[i],
            float(vx[i]),
            float(vy[i]),
            float(yaw_rate[i]),
            vx_prev,
            vy_prev,
            long_current,
            long_pred,
            steer_hist,
            out=obs[i],
        )
        action, realized = teleop_to_action(float(accel[i]), float(steer[i]), prev_steer)
        actions[i] = action
        long_pred = long_current
        long_current = float(action[0])
        steer_hist[1:] = steer_hist[:-1]
        steer_hist[0] = realized
        prev_steer = realized
        vx_prev = float(vx[i])
        vy_prev = float(vy[i])
    rewards = (vx * CONTROL_DT).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[-1] = 1.0
    resets = np.zeros(n, dtype=np.float32)
    resets[0] = 1.0
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "resets": resets,
        "vx": vx,
    }


def load_manual_dataset(
    roots: Iterable[Path] | None = None,
    *,
    max_bags: int | None = None,
) -> list[dict[str, np.ndarray]]:
    trajs: list[dict[str, np.ndarray]] = []
    bags = discover_manual_bags(roots)
    if max_bags is not None:
        bags = bags[: int(max_bags)]
    for bag in bags:
        try:
            traj = load_bag_trajectory(bag)
        except Exception as exc:
            print(f"skip {bag}: {type(exc).__name__}: {exc}")
            continue
        if traj is None:
            continue
        traj["bag"] = np.asarray([str(bag)])
        trajs.append(traj)
    return trajs


def split_holdout(
    trajs: Sequence[dict[str, np.ndarray]], *, holdout_frac: float = 0.25, seed: int = 0
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    if not trajs:
        raise ValueError("no trajectories")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(trajs))
    n_hold = max(1, int(round(len(trajs) * float(holdout_frac))))
    if n_hold >= len(trajs):
        n_hold = max(1, len(trajs) // 4) if len(trajs) > 1 else 0
    hold_idx = set(order[:n_hold].tolist()) if n_hold else set()
    train = [trajs[i] for i in range(len(trajs)) if i not in hold_idx]
    hold = [trajs[i] for i in range(len(trajs)) if i in hold_idx]
    if not train:
        train, hold = list(trajs), []
    return train, hold


def windows_from_trajectories(
    trajs: Sequence[dict[str, np.ndarray]],
    *,
    window: int = 32,
    stride: int = 16,
) -> dict[str, np.ndarray]:
    obs_w: list[np.ndarray] = []
    act_w: list[np.ndarray] = []
    rew_w: list[np.ndarray] = []
    done_w: list[np.ndarray] = []
    reset_w: list[np.ndarray] = []
    w = int(window)
    s = int(stride)
    for traj in trajs:
        t = int(traj["obs"].shape[0])
        if t < 2:
            continue
        width = min(w, t)
        last = max(1, t - width + 1)
        for start in range(0, last, s):
            sl = slice(start, start + width)
            obs_w.append(traj["obs"][sl])
            act_w.append(traj["actions"][sl])
            rew_w.append(traj["rewards"][sl])
            done_w.append(traj["dones"][sl])
            reset_w.append(traj["resets"][sl])
    if not obs_w:
        raise ValueError("no windows")
    return {
        "obs": np.stack(obs_w, axis=0),
        "actions": np.stack(act_w, axis=0),
        "rewards": np.stack(rew_w, axis=0),
        "dones": np.stack(done_w, axis=0),
        "resets": np.stack(reset_w, axis=0),
    }

"""Open-loop trajectory replay: native Warp actor obs vs ROS packing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "training"
for path in (
    _ROOT,
    _TRAINING,
    _ROOT / "src/racing_rl/f1tenth_rl_agent",
    _ROOT / "libs/f1tenth_policy",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_full_lap import (  # noqa: E402
    CRASH_TERM_KEYS,
    load_actor_bundle,
    spawn_speed_mps,
)
from f1tenth_policy.layout import (  # noqa: E402
    ACTOR_OBS_DIM,
    CONTROL_HZ,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
)
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.gym_certify import spawn_pose_from_centerline  # noqa: E402
from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    ImuCalibration,
    RawImuSample,
    beam_angles_rad,
    body_accel_to_driver,
    pack_lidar_from_scan,
)
from analysis.warp_sensor_lockstep import (  # noqa: E402
    _AUSTIN_CL,
    load_runtime,
    make_austin_env,
)

CHANNEL_NAMES = [f"lidar[{i}]" for i in range(LIDAR_DIM)] + [
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "vesc_speed",
    "vesc_current",
    "throttle_current",
    "throttle_pred",
    "steer_t",
    "steer_t1",
    "steer_t2",
    "steer_d0",
    "steer_d1",
    "steer_d2",
]
assert len(CHANNEL_NAMES) == ACTOR_OBS_DIM
PROPRIO_NAMES = CHANNEL_NAMES[LIDAR_DIM:]


def _ego_np(env, key: str) -> float:
    return float(env._ego.tensor[key][0])


def snapshot_state(env) -> dict:
    ego = env._ego.tensor
    hist = env._env.tensor["executed_steer_history"][0].detach().cpu().numpy()
    long_hist = env._env.tensor["executed_longitudinal_history"][0].detach().cpu().numpy()
    obs = env.actor_obs_buf[0].detach().cpu().numpy().astype(np.float32)
    return {
        "x": float(ego["x"][0]),
        "y": float(ego["y"][0]),
        "yaw": float(ego["yaw"][0]),
        "vx": float(ego["vx"][0]),
        "vy": float(ego["vy"][0]),
        "ax": float(ego["ax"][0]),
        "ay": float(ego["ay"][0]),
        "yaw_rate": float(ego["yaw_rate"][0]),
        "imu_gz": float(ego["imu_gz"][0]),
        "steer_logical": float(ego["steer_logical"][0]),
        "applied_effort": float(ego["applied_effort"][0]),
        "steer_hist": hist.astype(np.float32).copy(),
        "long_hist": long_hist.astype(np.float32).copy(),
        "obs": obs.copy(),
        "lidar": obs[:LIDAR_DIM].copy(),
    }


def record_training_traj(checkpoint: str | Path, *, steps: int = 32) -> dict:
    from config import DEFAULT_CONFIG

    pose = spawn_pose_from_centerline(_AUSTIN_CL)
    speed = spawn_speed_mps(DEFAULT_CONFIG["env"])
    env = make_austin_env()
    env.env_cfg["reset_stationary_probability"] = 0.0
    device = torch.device("cpu")
    actor, normalizer, _arch, _payload = load_actor_bundle(
        Path(checkpoint), device, DEFAULT_CONFIG
    )
    frames = []
    try:
        env.reset_to(
            [pose[0], pose[1]], pose[2], speed, seed=0, with_sensors=True
        )
        hidden = actor.initial_hidden(1, device=device, dtype=torch.float32)
        dt = 1.0 / float(CONTROL_HZ)
        prev_vx = _ego_np(env, "vx")
        prev_vy = _ego_np(env, "vy")
        for step in range(int(steps)):
            state = snapshot_state(env)
            state["step"] = step
            state["fd_ax"] = 0.0 if step == 0 else (state["vx"] - prev_vx) / dt
            state["fd_ay"] = 0.0 if step == 0 else (state["vy"] - prev_vy) / dt
            obs_t = torch.as_tensor(state["obs"], device=device).unsqueeze(0)
            norm = normalizer.normalize(obs_t)
            with torch.no_grad():
                action, _, hidden = actor.step(
                    norm,
                    hidden,
                    reset_mask=None,
                    deterministic=True,
                    with_logprob=False,
                )
            action_np = action[0].detach().cpu().numpy().astype(np.float32)
            state["norm"] = norm[0].detach().cpu().numpy().astype(np.float32)
            state["action"] = action_np.copy()
            frames.append(state)
            prev_vx = state["vx"]
            prev_vy = state["vy"]
            _obs, _rew, reset, extras = env.step(
                action.to(torch.float32), with_sensors=True
            )
            term = extras.get("termination", {})
            crashed = bool(reset[0])
            reasons = [
                key
                for key in CRASH_TERM_KEYS
                if key in term and bool(term[key][0])
            ]
            frames[-1]["reset"] = crashed
            frames[-1]["crash_reasons"] = reasons
            if crashed:
                hidden.zero_()
        return {
            "spawn": pose,
            "spawn_speed_mps": float(speed),
            "frames": frames,
        }
    finally:
        env.close()


def ros_pack_frame(runtime, frame: dict, *, imu: str) -> np.ndarray:
    if imu == "physics":
        ax, ay, gz = frame["ax"], frame["ay"], frame["imu_gz"]
    elif imu == "finite_diff":
        ax, ay, gz = frame["fd_ax"], frame["fd_ay"], frame["yaw_rate"]
    else:
        raise ValueError(imu)
    ax_d, ay_d = body_accel_to_driver(ax, ay)
    increment = float(beam_angles_rad()[1] - beam_angles_rad()[0])
    runtime.pack_observation(
        scan_angle_min=float(LIDAR_ANGLE_MIN),
        scan_angle_increment=increment,
        scan_ranges=frame["lidar"],
        scan_range_min=float(LIDAR_RANGE_MIN),
        scan_range_max=float(LIDAR_RANGE_MAX),
        imu_interval=[
            RawImuSample(
                stamp_s=float(frame["step"]) / CONTROL_HZ,
                ax=ax_d,
                ay=ay_d,
                az=float(si.GRAVITY_MS2),
                gx=0.0,
                gy=0.0,
                gz=gz,
            )
        ],
        imu_cal=ImuCalibration(accel_to_ms2=1.0, gyro_to_rads=1.0),
        speed=frame["vx"],
        applied_long=float(frame["long_hist"][0]),
        applied_steer_rad=float(frame["steer_hist"][0]),
        vesc_current=frame["applied_effort"],
    )
    return runtime.host_obs_buffer.copy()


def first_channel_div(
    left: np.ndarray, right: np.ndarray, *, atol: float
) -> dict | None:
    delta = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    hit = np.where(delta > float(atol))[0]
    if hit.size == 0:
        return None
    idx = int(hit[0])
    return {
        "channel": idx,
        "name": CHANNEL_NAMES[idx] if idx < len(CHANNEL_NAMES) else str(idx),
        "left": float(left[idx]),
        "right": float(right[idx]),
        "abs": float(delta[idx]),
    }


def compare_replay(
    checkpoint: str | Path,
    *,
    steps: int = 32,
    imu: str = "physics",
    obs_atol: float = 1e-4,
    act_atol: float = 1e-5,
) -> dict:
    traj = record_training_traj(checkpoint, steps=steps)
    runtime = load_runtime(checkpoint)
    runtime.reset_hidden()
    train_norm = None
    from config import DEFAULT_CONFIG

    _actor, train_norm, _arch, _payload = load_actor_bundle(
        Path(checkpoint), torch.device("cpu"), DEFAULT_CONFIG
    )
    del _actor
    first_obs = None
    first_norm_same_obs = None
    first_raw_action = None
    first_realized = None
    lidar_remap = None
    norm_stats = {
        "mean_mae": float(
            torch.abs(runtime.normalizer.mean - train_norm.mean).mean().item()
        ),
        "var_mae": float(
            torch.abs(runtime.normalizer.var - train_norm.var).mean().item()
        ),
    }
    for frame in traj["frames"]:
        packed_lidar = pack_lidar_from_scan(
            float(LIDAR_ANGLE_MIN),
            float(LIDAR_ANGLE_INCREMENT),
            frame["lidar"],
            scan_range_min=float(LIDAR_RANGE_MIN),
            scan_range_max=float(LIDAR_RANGE_MAX),
        )
        if lidar_remap is None:
            lidar_remap = first_channel_div(
                frame["lidar"], packed_lidar, atol=obs_atol
            )
            if lidar_remap is not None:
                lidar_remap["step"] = frame["step"]
        ros_obs = ros_pack_frame(runtime, frame, imu=imu)
        if first_obs is None:
            hit = first_channel_div(frame["obs"], ros_obs, atol=obs_atol)
            if hit is not None:
                hit["step"] = frame["step"]
                first_obs = hit
        train_obs_t = torch.as_tensor(frame["obs"]).unsqueeze(0)
        ros_norm_on_train = runtime.normalizer.normalize(train_obs_t)[0].numpy()
        if first_norm_same_obs is None:
            hit = first_channel_div(frame["norm"], ros_norm_on_train, atol=obs_atol)
            if hit is not None:
                hit["step"] = frame["step"]
                first_norm_same_obs = hit
        action_np, _t = runtime.infer_host_obs()
        if first_raw_action is None:
            delta = np.abs(frame["action"] - action_np)
            if float(delta.max()) > act_atol:
                axis = int(np.argmax(delta))
                first_raw_action = {
                    "step": frame["step"],
                    "channel": axis,
                    "name": ("long_norm", "delta_norm")[axis],
                    "left": float(frame["action"][axis]),
                    "right": float(action_np[axis]),
                    "abs": float(delta[axis]),
                }
        realized = runtime.realize_action(action_np)
        nxt_idx = int(frame["step"]) + 1
        if (
            first_realized is None
            and not frame["reset"]
            and nxt_idx < len(traj["frames"])
        ):
            nxt = traj["frames"][nxt_idx]
            steer_err = abs(realized.realized_steer_rad - nxt["steer_logical"])
            long_err = abs(realized.long_norm - float(nxt["long_hist"][0]))
            if steer_err > act_atol or long_err > act_atol:
                first_realized = {
                    "step": frame["step"],
                    "steer_train": float(nxt["steer_logical"]),
                    "steer_ros": float(realized.realized_steer_rad),
                    "long_train": float(nxt["long_hist"][0]),
                    "long_ros": float(realized.long_norm),
                }
        if frame["reset"]:
            runtime.reset_hidden()
    return {
        "spawn_speed_mps": traj["spawn_speed_mps"],
        "steps": len(traj["frames"]),
        "imu": imu,
        "norm_stats": norm_stats,
        "lidar_remap": lidar_remap,
        "first_obs": first_obs,
        "first_norm_same_raw_obs": first_norm_same_obs,
        "first_raw_action": first_raw_action,
        "first_realized": first_realized,
        "proprio_train0": {
            name: float(traj["frames"][0]["obs"][LIDAR_DIM + i])
            for i, name in enumerate(PROPRIO_NAMES)
        },
        "crash_step": next(
            (f["step"] for f in traj["frames"] if f["reset"]), None
        ),
        "crash_reasons": next(
            (f["crash_reasons"] for f in traj["frames"] if f["reset"]), []
        ),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--imu", choices=("physics", "finite_diff"), default="physics")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    report = compare_replay(args.ckpt, steps=int(args.steps), imu=args.imu)
    print("PARITY_JSON", json.dumps(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["first_obs"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

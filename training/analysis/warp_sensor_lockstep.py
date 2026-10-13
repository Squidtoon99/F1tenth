"""Lockstep Warp plant using sensor_racer observation packing."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

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

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env import runtime as rt  # noqa: E402
from f1tenth_policy.layout import LIDAR_DIM  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.gym_certify import (  # noqa: E402
    NetLapCounter,
    load_centerline,
    project_s,
    spawn_pose_from_centerline,
)
from f1tenth_rl_agent.policy_model import (  # noqa: E402
    load_sensor_actor,
    load_sensor_obs_norm,
)
from f1tenth_rl_agent.sensor_inference_runtime import (  # noqa: E402
    SensorInferenceRuntime,
)
from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    ImuCalibration,
    RawImuSample,
    beam_angles_rad,
    body_accel_to_driver,
)
from standalone_trainer import build_env_cfg  # noqa: E402

_AUSTIN_YAML = _TRAINING / "assets" / "Austin.yaml"
_AUSTIN_CL = _TRAINING / "assets" / "Austin_centerline.csv"
_AUSTIN_SHA = "16501f0cf2333305c305efb6cc3110c7a31e2843263067ddc52edfe826633318"


def austin_env_cfg():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = "Austin"
    cfg["env"]["opponent_strategy"] = None
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_stationary_probability"] = 1.0
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["sensor"]["lidar_map_yaml"] = str(_AUSTIN_YAML)
    cfg["sensor"]["lidar_map_sha256"] = _AUSTIN_SHA
    return build_env_cfg(cfg)


def make_austin_env() -> F1tenthEnv:
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    return F1tenthEnv(
        num_envs=1,
        env_cfg=austin_env_cfg(),
        obs_cfg=DEFAULT_CONFIG["obs"],
        reward_cfg=DEFAULT_CONFIG["reward"],
    )


def load_runtime(checkpoint: str | Path) -> SensorInferenceRuntime:
    device = torch.device("cpu")
    path = str(checkpoint)
    actor = load_sensor_actor(path, "actor", device)
    normalizer = load_sensor_obs_norm(
        path, device, si.OBS_NORM_EPS, si.OBS_NORM_CLIP
    )
    return SensorInferenceRuntime(
        actor,
        normalizer,
        device,
        deterministic=True,
        warmup_iters=2,
        i_drive_max_a=80.0,
        i_brake_max_a=40.0,
    )


def _spawn_reset(env: F1tenthEnv, runtime: SensorInferenceRuntime, pose) -> None:
    env.reset_to([pose[0], pose[1]], pose[2], 0.0, seed=0, with_sensors=True)
    runtime.reset_hidden()


def run_lockstep(
    checkpoint: str | Path,
    *,
    steps: int = 160,
    offtrack_m: float = 1.2,
    stop_on_lap: bool = False,
) -> dict:
    """Warp physics + sensor_racer packing; one inference per control tick."""
    pose = spawn_pose_from_centerline(_AUSTIN_CL)
    xy, cum, length = load_centerline(_AUSTIN_CL)
    env = make_austin_env()
    runtime = load_runtime(checkpoint)
    imu_cal = ImuCalibration(accel_to_ms2=1.0, gyro_to_rads=1.0)
    angles = beam_angles_rad()
    increment = float(angles[1] - angles[0])
    dt = 1.0 / float(si.CONTROL_HZ)
    counter = NetLapCounter(length)
    try:
        _spawn_reset(env, runtime, pose)
        prev_vx = 0.0
        prev_vy = 0.0
        applied_long = 0.0
        applied_steer = 0.0
        vesc_current = 0.0
        first_offtrack = None
        max_speed = 0.0
        t = 0.0
        x = float(pose[0])
        y = float(pose[1])
        s = 0.0
        step = -1
        for step in range(int(steps)):
            ego = env._ego.tensor
            x = float(ego["x"][0])
            y = float(ego["y"][0])
            vx = float(ego["vx"][0])
            vy = float(ego["vy"][0])
            max_speed = max(max_speed, abs(vx))
            yaw_rate = float(ego["yaw_rate"][0])
            s, dist = project_s(x, y, xy, cum)
            offtrack = dist > float(offtrack_m)
            counter.update(s, offtrack, t)
            if offtrack and first_offtrack is None:
                first_offtrack = {
                    "step": step,
                    "t_s": t,
                    "s_m": s,
                    "x": x,
                    "y": y,
                    "dist_m": dist,
                }
            if stop_on_lap and counter.laps >= 1:
                break
            ax = 0.0 if step == 0 else (vx - prev_vx) / dt
            ay = 0.0 if step == 0 else (vy - prev_vy) / dt
            ax_d, ay_d = body_accel_to_driver(ax, ay)
            lidar = env.actor_obs_buf[0, :LIDAR_DIM].detach().cpu().numpy()
            runtime.pack_observation(
                scan_angle_min=float(si.LIDAR_ANGLE_MIN),
                scan_angle_increment=increment,
                scan_ranges=lidar,
                scan_range_min=float(si.LIDAR_RANGE_MIN),
                scan_range_max=float(si.LIDAR_RANGE_MAX),
                imu_interval=[
                    RawImuSample(
                        stamp_s=t,
                        ax=ax_d,
                        ay=ay_d,
                        az=float(si.GRAVITY_MS2),
                        gx=0.0,
                        gy=0.0,
                        gz=yaw_rate,
                    )
                ],
                imu_cal=imu_cal,
                speed=vx,
                applied_long=applied_long,
                applied_steer_rad=applied_steer,
                vesc_current=vesc_current,
            )
            action_np, _timings = runtime.infer_host_obs()
            realized = runtime.realize_action(action_np)
            applied_long = float(realized.long_norm)
            applied_steer = float(realized.realized_steer_rad)
            signed = float(realized.drive_a - realized.brake_a)
            vesc_current = si.applied_current_fraction(
                signed, runtime.i_drive_max_a, runtime.i_brake_max_a
            )
            action = torch.tensor(
                [[float(action_np[0]), float(action_np[1])]],
                dtype=torch.float32,
            )
            _obs, _rew, reset, _extra = env.step(action, with_sensors=True)
            prev_vx = vx
            prev_vy = vy
            t += dt
            if bool(reset[0]):
                if first_offtrack is None:
                    first_offtrack = {
                        "step": step,
                        "t_s": t,
                        "s_m": s,
                        "x": x,
                        "y": y,
                        "dist_m": dist,
                    }
                _spawn_reset(env, runtime, pose)
                applied_long = 0.0
                applied_steer = 0.0
                vesc_current = 0.0
                prev_vx = 0.0
                prev_vy = 0.0
                spawn_s, _dist = project_s(pose[0], pose[1], xy, cum)
                if not offtrack:
                    counter.crashes += 1
                counter.reset_progress(spawn_s, t)
        progress = counter.laps * length + counter.progress
        return {
            "steps": step + 1,
            "duration_s": float((step + 1) * dt) if step >= 0 else 0.0,
            "progress_m": float(progress),
            "laps": int(counter.laps),
            "lap_times_s": [float(v) for v in counter.lap_times],
            "crashes": int(counter.crashes),
            "x": x,
            "y": y,
            "s_m": s,
            "max_speed_mps": float(max_speed),
            "first_offtrack": first_offtrack,
            "track_length_m": float(length),
        }
    finally:
        env.close()


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Warp plant + sensor_racer packing lockstep on Austin"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--offtrack-m", type=float, default=1.2)
    parser.add_argument("--stop-on-lap", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    result = run_lockstep(
        args.ckpt,
        steps=int(args.steps),
        offtrack_m=float(args.offtrack_m),
        stop_on_lap=bool(args.stop_on_lap),
    )
    print("WARP_ROS_JSON", json.dumps(result))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

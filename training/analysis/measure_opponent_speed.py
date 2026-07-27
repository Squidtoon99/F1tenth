"""Measure solo rollout speed for sensor policy checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

import torch  # noqa: E402

from f1tenth_policy import (  # noqa: E402
    actor_from_architecture,
    load_sensor_artifact,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.normalizer import ObsNormalizer  # noqa: E402
from evaluation import deterministic_rollout  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env import runtime as rt  # noqa: E402
from standalone_trainer import (  # noqa: E402
    DEFAULT_CONFIG,
    _deep_merge,
    episode_length_for_track,
    select_device,
)


def measure_checkpoint_speed(
    checkpoint: str | Path,
    *,
    config: str | Path | None = None,
    track: str = "Austin",
    num_envs: int = 32,
    steps: int = 500,
    seed: int = 0,
    device_str: str = "cuda",
    warmup_steps: int = 50,
) -> dict:
    ckpt_path = Path(checkpoint).resolve()
    if config is not None:
        from analysis.lap_timing import _read_config_json

        patch = _read_config_json(Path(config))
        cfg = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), patch)
        config_source = str(Path(config).resolve())
    else:
        from analysis.lap_timing import load_lap_timing_config

        patch, config_source = load_lap_timing_config(ckpt_path)
        cfg = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), patch)
    cfg["env"]["track"] = track
    cfg["env"]["opponent_strategy"] = None
    cfg["env"]["episode_length"] = episode_length_for_track(
        track=track,
        workspace_dir=str(_TRAINING_DIR),
        ref_lap_speed_mps=float(cfg["env"].get("expected_lap_speed_mps", 3.5)),
        lap_multiplier=float(cfg["env"].get("episode_lap_multiplier", 3.0)),
    )

    device = select_device(device_str)
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    control_interval = int(cfg["env"]["control_interval"])
    clip_actions = float(cfg["env"]["clip_actions"])
    env = F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    payload = load_sensor_artifact(str(ckpt_path), map_location=device)
    architecture = dict(payload["actor_architecture"])
    validate_sensor_policy_artifact(
        payload,
        expected_architecture=architecture,
        expected_actor_obs_dim=int(cfg["obs"]["num_actor_obs"]),
        expected_action_dim=int(cfg["env"]["num_actions"]),
        expected_layout_version=int(cfg["obs"]["actor_layout_version"]),
        expected_critic_obs_dim=int(cfg["obs"]["num_obs"]),
    )
    actor = actor_from_architecture(architecture).to(device=device, dtype=torch.float32)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    normalizer = ObsNormalizer(
        obs_dim=int(cfg["obs"]["num_actor_obs"]),
        device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    normalizer.load_state_dict(payload["obs_norm"])
    speeds: list[float] = []

    def on_step(step, _env, _sb, _reward, _done, extras):
        if step < warmup_steps:
            return
        speed = extras["metrics"]["speed_xy"].detach().cpu()
        speeds.extend(float(x) for x in speed.tolist())

    deterministic_rollout(
        env,
        actor,
        normalizer.normalize,
        num_steps=steps,
        control_interval=control_interval,
        clip_actions=clip_actions,
        seed=seed,
        callback=on_step,
        with_sensors=True,
    )
    env.close()
    mean_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0
    return {
        "checkpoint": str(ckpt_path),
        "transitions": int(payload.get("env_transitions", 0)),
        "config_source": config_source,
        "mean_speed_mps": mean_speed,
        "num_samples": len(speeds),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "num_envs": num_envs,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure solo policy rollout speed")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--track", type=str, default="Austin")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints: list[Path] = [Path(p) for p in args.checkpoint]
    if args.run_dir:
        ckpt_dir = Path(args.run_dir) / "checkpoints"
        found = sorted(ckpt_dir.glob("policy_*.pt"))
        if args.stride > 1:
            found = found[:: args.stride]
        if args.limit:
            found = found[: args.limit]
        checkpoints.extend(found)
    if not checkpoints:
        raise SystemExit("Provide --checkpoint and/or --run-dir")

    results = []
    for ckpt in checkpoints:
        result = measure_checkpoint_speed(
            ckpt,
            config=args.config,
            track=args.track,
            num_envs=args.num_envs,
            steps=args.steps,
            seed=args.seed,
            device_str=args.device,
            warmup_steps=args.warmup_steps,
        )
        results.append(result)
        print(
            f"{Path(result['checkpoint']).name}: "
            f"{result['mean_speed_mps']:.3f} m/s "
            f"(transitions={result['transitions']})"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

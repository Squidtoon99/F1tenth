"""Evaluate a trained policy and visualize the rollout (live and/or mp4).

This is an *eval-only* tool: it loads a checkpoint, runs the policy deterministically
in a single environment, and renders a top-down view of the car on the track. It
never trains, samples the replay buffer, or mutates a training run.

Watch live (opens a Rerun viewer, updates in real time -- no waiting for a file):

    python eval_visualize.py --checkpoint outputs/runs/<id>/checkpoints/ckpt_8000.pt --live

Save an mp4 (e.g. to attach to W&B or share):

    python eval_visualize.py --checkpoint .../ckpt_8000.pt --mp4 outputs/eval.mp4

Both at once are fine. Defaults to the pure-Torch backend (Genesis-free).
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from f1tenth_env.env import F1tenthEnv
from f1tenth_env.eval_viz import RolloutVisualizer, yaw_from_quat_wxyz
from standalone_trainer import (
    DEFAULT_CONFIG,
    ObsNormalizer,
    build_models,
    episode_length_for_track,
    select_device,
    select_genesis_backend,
    _maybe_patch_headless_rasterizer,
)


def build_eval_config(args: argparse.Namespace) -> dict:
    """Minimal 1v0 eval config: track + physics backend + derived episode length."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    cfg["env"]["physics_backend"] = args.physics
    if args.throttle_mode is not None:
        cfg["env"]["throttle_mode"] = args.throttle_mode
    cfg["env"]["episode_length"] = episode_length_for_track(
        track=args.track,
        workspace_dir=str(Path(__file__).resolve().parent),
        ref_lap_speed_mps=float(cfg["env"].get("expected_lap_speed_mps", 3.5)),
        lap_multiplier=float(cfg["env"].get("episode_lap_multiplier", 3.0)),
    )
    return cfg


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    p = argparse.ArgumentParser(description="Visualize a trained policy rollout")
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt")
    p.add_argument("--track", type=str, default=cfg["env"]["track"])
    p.add_argument("--physics", type=str, default="torch",
                   choices=["genesis", "torch"])
    p.add_argument("--throttle-mode", type=str, default=None,
                   choices=["force", "speed"])
    p.add_argument("--steps", type=int, default=1500,
                   help="Number of control steps to roll out.")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-show", type=int, default=1,
                   help="How many env instances to draw overlaid on the track "
                        "(swarm view). Bumps --num-envs up to match if needed.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--backend", type=str, default="cpu",
                   choices=["cpu", "gpu", "cuda", "metal", "auto"])
    p.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of using the deterministic mean.")
    # Visualization sinks (either/both).
    p.add_argument("--live", action="store_true", help="Stream live to Rerun.")
    p.add_argument("--mp4", type=str, default=None, help="Write an mp4 to this path.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--img-size", type=int, default=900)
    p.add_argument("--no-spawn", action="store_true",
                   help="Do not spawn a Rerun viewer (use with --rr-save or a "
                        "manually-connected viewer).")
    p.add_argument("--rr-save", type=str, default=None,
                   help="Save the live stream to a .rrd file instead of spawning.")
    return p.parse_args()


def _init_physics_runtime(args, cfg) -> torch.device:
    if str(cfg["env"].get("physics_backend")) == "torch":
        device = select_device(args.device)
        gs.tc_float = torch.float64 if args.precision == "64" else torch.float32
        gs.tc_int = torch.int32
        gs.device = device
        if getattr(gs, "EPS", None) is None:
            gs.EPS = 1e-12
        return device
    _maybe_patch_headless_rasterizer()
    backend = select_genesis_backend(args.backend)
    gs.init(backend=backend, precision=args.precision, performance_mode=True)
    return select_device(args.device) if backend == gs.cpu else gs.device


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.num_show = max(1, args.num_show)
    args.num_envs = max(args.num_envs, args.num_show)

    cfg = build_eval_config(args)
    device = _init_physics_runtime(args, cfg)

    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": args.num_envs},
        **cfg["env"],
    }
    control_interval = int(cfg["env"]["control_interval"])
    clip_actions = float(cfg["env"]["clip_actions"])

    env = F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )

    models, _ = build_models(cfg, device)
    normalizer = ObsNormalizer(
        obs_dim=cfg["obs"]["num_obs"],
        device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    models.actor.load_state_dict(payload["actor"])
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    else:
        print("WARNING: checkpoint has no obs_norm; using identity normalizer stats.")
    models.actor.eval()

    viz = RolloutVisualizer(
        centerline=env.track_state["centerline"],
        w_tr_left=env.track_state["w_tr_left"],
        w_tr_right=env.track_state["w_tr_right"],
        car_length=float(env_cfg.get("car_length", 0.46)),
        car_width=float(env_cfg.get("car_width", 0.30)),
        num_show=args.num_show,
        live=args.live,
        mp4_path=args.mp4,
        fps=args.fps,
        img_size=args.img_size,
        has_opponent=env.has_opponent,
        rr_spawn=not args.no_spawn,
        rr_save_path=args.rr_save,
    )

    obs, _ = env.reset()
    obs = obs.to(torch.float32)
    print(f"Rolling out {args.steps} steps on '{args.track}' "
          f"(physics={cfg['env']['physics_backend']}, "
          f"live={args.live}, mp4={args.mp4})")

    n = args.num_show

    def _yaws(quat_batch) -> np.ndarray:
        return np.array([yaw_from_quat_wxyz(q.tolist()) for q in quat_batch])

    with torch.no_grad():
        for step in range(args.steps):
            actions, _ = models.actor(
                normalizer.normalize(obs),
                deterministic=not args.stochastic,
                with_logprob=False,
            )
            actions = actions.clamp(-clip_actions, clip_actions)
            obs, _, done, _ = env.step(actions.to(gs.tc_float), n_steps=control_interval)
            obs = obs.to(torch.float32)

            st = env.backend.read_state()
            ego_xy = st["base_pos"][:n, :2].cpu().numpy()
            ego_yaw = _yaws(st["base_quat"][:n])
            speed = torch.linalg.norm(st["base_lin_vel"][:n, :2], dim=-1).cpu().numpy()
            opp_xy = opp_yaw = None
            if env.has_opponent and "opp_base_pos" in st:
                opp_xy = st["opp_base_pos"][:n, :2].cpu().numpy()
                opp_yaw = _yaws(st["opp_base_quat"][:n])
            viz.render(
                ego_xy=ego_xy,
                ego_yaw=ego_yaw,
                speed=speed,
                opp_xy=opp_xy,
                opp_yaw=opp_yaw if opp_yaw is not None else 0.0,
                done=done[:n].cpu().numpy(),
            )

    out = viz.close()
    env.close()
    if out:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Evaluate a trained policy and visualize the rollout (live and/or mp4).

This is an *eval-only* tool: it loads a policy artifact and runs deterministically
in a single environment, and renders a top-down view of the car on the track. It
never trains, samples the replay buffer, or mutates a training run.

Watch live (opens a Rerun viewer, updates in real time -- no waiting for a file):

    python eval_visualize.py --checkpoint outputs/runs/<id>/checkpoints/policy_8000.pt --live

Save an mp4 (e.g. to attach to W&B or share):

    python eval_visualize.py --checkpoint .../policy_8000.pt --mp4 outputs/eval.mp4

Both at once are fine.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import deterministic_rollout
from f1tenth_env import runtime as rt
from f1tenth_env.env import F1tenthEnv
from f1tenth_env.eval_viz import RolloutVisualizer, yaw_from_quat_wxyz
from standalone_trainer import (
    DEFAULT_CONFIG,
    ObsNormalizer,
    build_models,
    episode_length_for_track,
    select_device,
)


def build_eval_config(args: argparse.Namespace) -> dict:
    """Eval config from run config.json when available, else DEFAULT_CONFIG."""
    cfg_path = Path(args.checkpoint).resolve().parent.parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())["config"]
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    cfg["env"]["domain_randomization"] = {
        **cfg["env"]["domain_randomization"],
        "enabled": False,
    }
    if args.throttle_mode is not None:
        cfg["env"]["throttle_mode"] = args.throttle_mode
    if args.opponent_ckpt is not None:
        cfg["env"]["opponent_strategy"] = args.opponent_strategy
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
    p.add_argument("--checkpoint", required=True, help="Path to policy_*.pt")
    p.add_argument("--opponent-ckpt", type=str, default=None,
                   help="Path to opponent policy_*.pt (enables 1v1 self-play eval).")
    p.add_argument("--opponent-strategy", type=str, default="policy",
                   choices=["policy", "mixed"],
                   help="Opponent controller when --opponent-ckpt is set.")
    p.add_argument("--track", type=str, default=cfg["env"]["track"])
    p.add_argument("--throttle-mode", type=str, default=None,
                   choices=["force", "speed"])
    p.add_argument("--steps", type=int, default=1500,
                   help="Number of control steps to roll out.")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-show", type=int, default=1,
                   help="How many env instances to draw overlaid on the track "
                        "(swarm view). Bumps --num-envs up to match if needed.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    p.add_argument("--seed", type=int, default=0)
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
    device = select_device(args.device)
    rt.configure(
        float_dtype=torch.float64 if args.precision == "64" else torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    return device


def main() -> None:
    args = parse_args()

    args.num_show = max(1, args.num_show)
    args.num_envs = max(args.num_envs, args.num_show)
    if args.opponent_ckpt is not None:
        args.num_show = max(1, args.num_show)
        args.num_envs = max(args.num_envs, args.num_show, 1)

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
        print("WARNING: artifact has no obs_norm; using identity normalizer stats.")
    models.actor.eval()

    if args.opponent_ckpt is not None:
        opp_payload = torch.load(
            args.opponent_ckpt, map_location=device, weights_only=False
        )
        opp_actor = {
            k: v.detach().cpu().clone()
            for k, v in opp_payload["actor"].items()
        }
        opp_normalizer = ObsNormalizer(
            obs_dim=cfg["obs"]["num_obs"],
            device=device,
            eps=float(cfg["obs"].get("norm_eps", 1e-8)),
            clip=float(cfg["obs"].get("norm_clip", 10.0)),
        )
        if "obs_norm" in opp_payload:
            opp_normalizer.load_state_dict(opp_payload["obs_norm"])
        else:
            print("WARNING: opponent artifact has no obs_norm; using identity stats.")
        env.refresh_opponent_policy(
            opp_actor,
            opp_normalizer.mean.detach().cpu().clone(),
            opp_normalizer.var.detach().cpu().clone(),
        )

    viz = RolloutVisualizer(
        centerline=env.track_state["centerline"],
        w_tr_left=env.track_state["w_tr_left"],
        w_tr_right=env.track_state["w_tr_right"],
        car_length=float(env_cfg.get("car_length", 0.568)),
        car_width=float(env_cfg.get("car_width", 0.296)),
        num_show=args.num_show,
        live=args.live,
        mp4_path=args.mp4,
        fps=args.fps,
        img_size=args.img_size,
        has_opponent=env.has_opponent,
        rr_spawn=not args.no_spawn,
        rr_save_path=args.rr_save,
    )

    duel = "1v1" if args.opponent_ckpt is not None else "1v0"
    print(f"Rolling out {args.steps} steps on '{args.track}' ({duel}, "
          "physics=torch, "
          f"live={args.live}, mp4={args.mp4})")

    n = args.num_show

    def _yaws(quat_batch) -> np.ndarray:
        return np.array([yaw_from_quat_wxyz(q.tolist()) for q in quat_batch])

    def render_step(_step, rollout_env, _state_before, _reward, done, _extras):
        st = rollout_env.backend.read_state()
        ego_xy = st["base_pos"][:n, :2].cpu().numpy()
        ego_yaw = _yaws(st["base_quat"][:n])
        speed = torch.linalg.norm(st["base_lin_vel"][:n, :2], dim=-1).cpu().numpy()
        opp_xy = opp_yaw = None
        if rollout_env.has_opponent and "opp_base_pos" in st:
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

    deterministic_rollout(
        env,
        models.actor,
        normalizer.normalize,
        num_steps=args.steps,
        control_interval=control_interval,
        clip_actions=clip_actions,
        seed=args.seed,
        callback=render_step,
    )

    out = viz.close()
    env.close()
    if out:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Eval-only top-down rollout visualizer (live Rerun and/or mp4)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import deterministic_rollout
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.eval_viz import RolloutVisualizer, yaw_from_quat_wxyz
from standalone_trainer import (
    DEFAULT_CONFIG,
    ObsNormalizer,
    build_env_cfg,
    build_models,
    episode_length_for_track,
    select_device,
)


def build_eval_config(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.checkpoint).resolve().parent.parent / "config.json"
    cfg = (
        json.loads(cfg_path.read_text())["config"]
        if cfg_path.exists()
        else copy.deepcopy(DEFAULT_CONFIG)
    )
    cfg["env"]["track"] = args.track
    cfg["env"]["domain_randomization"] = {
        **cfg["env"]["domain_randomization"],
        "enabled": False,
    }
    cfg["env"]["opponent_strategy"] = (
        "policy" if args.opponent_ckpt is not None else "none"
    )
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent-ckpt", type=str, default=None)
    p.add_argument("--track", type=str, default=cfg["env"]["track"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-show", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true")
    p.add_argument("--mp4", type=str, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--img-size", type=int, default=900)
    p.add_argument("--no-spawn", action="store_true")
    p.add_argument("--rr-save", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.num_show = max(1, args.num_show)
    args.num_envs = max(args.num_envs, args.num_show)
    cfg = build_eval_config(args)
    device = select_device(args.device)
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": args.num_envs},
    )
    env = F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    models, _ = build_models(cfg, device)
    actor_obs_dim = int(cfg["obs"]["num_actor_obs"])
    normalizer = ObsNormalizer(
        obs_dim=actor_obs_dim,
        device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    models.actor.load_state_dict(payload["actor"])
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    models.actor.eval()
    if args.opponent_ckpt is not None:
        opp_payload = torch.load(
            args.opponent_ckpt, map_location=device, weights_only=False
        )
        opp_norm = ObsNormalizer(
            obs_dim=actor_obs_dim,
            device=device,
            eps=float(cfg["obs"].get("norm_eps", 1e-8)),
            clip=float(cfg["obs"].get("norm_clip", 10.0)),
        )
        if "obs_norm" in opp_payload:
            opp_norm.load_state_dict(opp_payload["obs_norm"])
        env.refresh_opponent_policy(
            {k: v.detach().cpu().clone() for k, v in opp_payload["actor"].items()},
            opp_norm.mean.detach().cpu().clone(),
            opp_norm.var.detach().cpu().clone(),
            actor_architecture=opp_payload.get("actor_architecture"),
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
    n = args.num_show

    def render_step(_step, rollout_env, _state_before, _reward, done, _extras):
        st = rollout_env.read_state()
        ego_xy = st["base_pos"][:n, :2].cpu().numpy()
        ego_yaw = np.array(
            [yaw_from_quat_wxyz(q.tolist()) for q in st["base_quat"][:n]]
        )
        speed = torch.linalg.norm(st["base_lin_vel"][:n, :2], dim=-1).cpu().numpy()
        opp_xy = opp_yaw = None
        if rollout_env.has_opponent and "opp_base_pos" in st:
            opp_xy = st["opp_base_pos"][:n, :2].cpu().numpy()
            opp_yaw = np.array(
                [yaw_from_quat_wxyz(q.tolist()) for q in st["opp_base_quat"][:n]]
            )
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
        control_interval=int(cfg["env"]["control_interval"]),
        clip_actions=float(cfg["env"]["clip_actions"]),
        seed=args.seed,
        callback=render_step,
        with_sensors=True,
    )
    out = viz.close()
    env.close()
    if out:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU/CPU smokes for Gigaflow conditioning before the long Brev run."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))
_POLICY = str(_TRAINING_ROOT.parent / "libs" / "f1tenth_policy")
if _POLICY not in sys.path:
    sys.path.insert(0, _POLICY)

from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env import runtime as rt  # noqa: E402
from f1tenth_env.gigaflow_cond import (  # noqa: E402
    C_VEL,
    GIGAFLOW_ACTOR_OBS_DIM,
    GIGAFLOW_CRITIC_OBS_DIM,
    format_gigaflow_spec,
    speed_cap_from_c,
)
from f1tenth_env.sensors import ACTOR_LIDAR_DIM  # noqa: E402
from f1tenth_policy.normalizer import ObsNormalizer  # noqa: E402
from standalone_trainer import (  # noqa: E402
    build_config,
    build_ppo_models,
    parse_args,
    unpack_sensor_observations,
)


def _cfg(num_envs: int):
    patch = json.loads(
        (_TRAINING_ROOT / "configs" / "gigaflow_v2.json").read_text()
    )
    patch["scenarios"] = []
    args, explicit = parse_args([])
    args.num_envs = int(num_envs)
    cfg = build_config(args, patch=patch, explicit=explicit)
    cfg["ppo"]["rollout_steps"] = 4
    cfg["ppo"]["env_minibatches"] = int(num_envs)
    cfg["ppo"]["epochs"] = 1
    return cfg


def _open_env(cfg, num_envs, opponent, device):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["gigaflow_force_track"] = "Oschersleben"
    env_cfg["gigaflow_force_variant"] = "width_0.8"
    env_cfg["domain_randomization"]["enabled"] = False
    if opponent:
        env_cfg["opponent_strategy"] = "policy"
        env_cfg["actor_type"] = cfg["model"]["actor_type"]
        env_cfg["actor_hidden_layers"] = list(
            cfg["model"]["actor_hidden_layers"]
        )
        env_cfg["lidar_pool_bins"] = int(cfg["model"]["lidar_pool_bins"])
        env_cfg["lidar_projection_dim"] = int(
            cfg["model"]["lidar_projection_dim"]
        )
        env_cfg["lidar_dim"] = ACTOR_LIDAR_DIM
    else:
        env_cfg["opponent_strategy"] = None
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def smoke_optimizer(device: torch.device) -> dict:
    cfg = _cfg(8)
    actor_norm = ObsNormalizer(GIGAFLOW_ACTOR_OBS_DIM, device=device)
    critic_norm = ObsNormalizer(GIGAFLOW_CRITIC_OBS_DIM, device=device)
    models, trainer = build_ppo_models(
        cfg, device, actor_norm, critic_norm, num_envs=8
    )
    if models.actor.lidar_dim != ACTOR_LIDAR_DIM:
        raise RuntimeError("actor is not lidar-split")
    if trainer.critic_obs_dim != GIGAFLOW_CRITIC_OBS_DIM:
        raise RuntimeError("critic is not privileged+C")
    env = _open_env(cfg, 8, opponent=False, device=device)
    try:
        raw, extras = env.reset(with_sensors=True)
        actor_obs, critic_obs = unpack_sensor_observations(raw)
        trainer.initialize(actor_obs, critic_obs)
        metrics = None
        for _ in range(trainer.rollout_steps):
            action = trainer.act(actor_obs, critic_obs)
            raw, reward, done, extras = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
            actor_obs, critic_obs = unpack_sensor_observations(raw)
            if not torch.isfinite(actor_obs).all():
                raise RuntimeError("NaN in actor obs")
            metrics = trainer.observe(
                actor_obs,
                critic_obs,
                reward,
                done,
                timed_out=extras["termination"]["time_out"],
                timeout_critic_obs=extras["observations"]["terminal_critic"],
            )
        if metrics is None:
            raise RuntimeError("optimizer step did not run")
        if not np.isfinite(metrics["policy_loss"]):
            raise RuntimeError(f"non-finite policy_loss {metrics['policy_loss']}")
        return {
            "policy_loss": float(metrics["policy_loss"]),
            "value_loss": float(metrics["value_loss"]),
            "actor_obs_dim": int(models.actor.obs_dim),
            "critic_obs_dim": int(trainer.critic_obs_dim),
            "lidar_dim": int(models.actor.lidar_dim),
            "value_clip": float(trainer.value_clip),
            "advantage_filter_mode": trainer.advantage_filter_mode,
            "track": extras.get("gigaflow_track"),
            "variant": extras.get("gigaflow_variant"),
        }
    finally:
        env.close()


def smoke_two_agent_rollout(device: torch.device) -> dict:
    cfg = _cfg(1)
    env = _open_env(cfg, 1, opponent=True, device=device)
    try:
        raw, extras = env.reset(with_sensors=True)
        actor, critic = unpack_sensor_observations(raw)
        env._c_ego[:, C_VEL] = 1.0 / 1.5
        env._c_opp[:, C_VEL] = 1.5
        env._apply_gigaflow_plant(torch.ones(1, dtype=torch.bool, device=device))
        env._write_conditioning_columns()
        ego_moved = False
        opp_moved = False
        for _ in range(16):
            actions = torch.zeros(1, 2, device=device)
            actions[:, 0] = 1.0
            env._policy_opponent_actions[:, 0] = 1.0
            env._policy_opponent_actions[:, 1] = 0.0
            env._policy_actions_valid = True
            raw, reward, done, extras = env.step(
                actions, n_steps=env.control_interval, with_sensors=True
            )
            actor, critic = unpack_sensor_observations(raw)
            if not torch.isfinite(actor).all() or not torch.isfinite(critic).all():
                raise RuntimeError("NaN in two-agent obs")
            if not torch.isfinite(reward).all():
                raise RuntimeError("NaN in reward")
            if float(env._ego.tensor["vx"].abs().max()) > 0.05:
                ego_moved = True
            if float(env._opponent.tensor["vx"].abs().max()) > 0.05:
                opp_moved = True
        caps = speed_cap_from_c(
            torch.stack([env._c_ego[0, C_VEL], env._c_opp[0, C_VEL]])
            .detach()
            .cpu()
            .numpy(),
            float(cfg["env"]["gigaflow_nominal_speed_mps"]),
        )
        if extras.get("gigaflow_track") != "Oschersleben":
            raise RuntimeError(f"track {extras.get('gigaflow_track')!r}")
        if extras.get("gigaflow_variant") != "width_0.8":
            raise RuntimeError(f"variant {extras.get('gigaflow_variant')!r}")
        if not ego_moved or not opp_moved:
            raise RuntimeError(
                f"agents did not step ego={ego_moved} opp={opp_moved} "
                f"vx_ego={float(env._ego.tensor['vx'][0])} "
                f"vx_opp={float(env._opponent.tensor['vx'][0])}"
            )
        return {
            "track": extras["gigaflow_track"],
            "variant": extras["gigaflow_variant"],
            "ego_c_vel": float(env._c_ego[0, C_VEL]),
            "opp_c_vel": float(env._c_opp[0, C_VEL]),
            "ego_speed_cap": float(caps[0]),
            "opp_speed_cap": float(caps[1]),
            "vx_ego": float(env._ego.tensor["vx"][0]),
            "vx_opp": float(env._opponent.tensor["vx"][0]),
            "actor_obs_dim": int(actor.shape[-1]),
            "critic_obs_dim": int(critic.shape[-1]),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    print(format_gigaflow_spec())
    opt = smoke_optimizer(device)
    print("SMOKE1_OK", json.dumps(opt, sort_keys=True))
    roll = smoke_two_agent_rollout(device)
    print("SMOKE2_OK", json.dumps(roll, sort_keys=True))


if __name__ == "__main__":
    main()

"""Shared deterministic policy rollout for training and evaluation tools."""

from __future__ import annotations

import random
from collections.abc import Callable

import torch

from f1tenth_env import runtime as rt


def _actor_obs_from_env(obs, *, with_sensors: bool) -> torch.Tensor:
    if with_sensors:
        if not isinstance(obs, dict) or "actor" not in obs:
            raise TypeError(
                "with_sensors=True requires a dict observation with an 'actor' key"
            )
        return obs["actor"].to(torch.float32)
    if isinstance(obs, dict):
        raise TypeError(
            "with_sensors=False expects a flat observation tensor, not a dict"
        )
    return obs.to(torch.float32)


def deterministic_rollout(
    env,
    actor,
    normalize: Callable[[torch.Tensor], torch.Tensor],
    *,
    num_steps: int,
    control_interval: int,
    clip_actions: float,
    seed: int = 0,
    callback: Callable | None = None,
    capture_state_before: bool = False,
    with_sensors: bool = False,
) -> dict:
    python_state = random.getstate()
    devices = (
        [env.device.index if env.device.index is not None else torch.cuda.current_device()]
        if env.device.type == "cuda"
        else []
    )
    total_reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    done_count = 0
    finite = True

    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            torch.manual_seed(seed)
            raw_obs, _ = env.reset(seed=seed, with_sensors=with_sensors)
            obs = _actor_obs_from_env(raw_obs, with_sensors=with_sensors)
            with torch.no_grad():
                for step in range(num_steps):
                    state_before = (
                        env.read_state() if capture_state_before else None
                    )
                    actions, _ = actor(
                        normalize(obs), deterministic=True, with_logprob=False
                    )
                    actions = actions.clamp(-clip_actions, clip_actions)
                    raw_obs, reward, done, extras = env.step(
                        actions.to(rt.tc_float),
                        n_steps=control_interval,
                        with_sensors=with_sensors,
                    )
                    obs = _actor_obs_from_env(raw_obs, with_sensors=with_sensors)
                    total_reward += reward.to(torch.float32)
                    done_count += int(done.sum().item())
                    finite = finite and bool(
                        torch.isfinite(obs).all() and torch.isfinite(reward).all()
                    )
                    if callback is not None:
                        callback(step, env, state_before, reward, done, extras)
    finally:
        random.setstate(python_state)
    return {
        "total_reward": total_reward,
        "done_count": done_count,
        "finite": finite,
        "final_observation": obs,
    }

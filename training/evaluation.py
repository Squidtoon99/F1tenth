"""Shared deterministic policy rollout for training and evaluation tools."""

from __future__ import annotations

import random
from collections.abc import Callable

import torch

from f1tenth_env import runtime as rt


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
            obs, _ = env.reset(seed=seed)
            obs = obs.to(torch.float32)
            with torch.no_grad():
                for step in range(num_steps):
                    state_before = (
                        env.read_state() if capture_state_before else None
                    )
                    actions, _ = actor(
                        normalize(obs), deterministic=True, with_logprob=False
                    )
                    actions = actions.clamp(-clip_actions, clip_actions)
                    obs, reward, done, extras = env.step(
                        actions.to(rt.tc_float), n_steps=control_interval
                    )
                    obs = obs.to(torch.float32)
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

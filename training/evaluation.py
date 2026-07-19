"""Shared deterministic policy rollout for training and evaluation tools."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

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


def flat_architecture_from_state_dict(
    actor_sd: dict, *, obs_dim: int, action_dim: int
) -> dict:
    """Infer a format-2 flat_mlp architecture from Linear weight shapes."""
    hidden = []
    idx = 0
    while f"net.{idx}.weight" in actor_sd:
        hidden.append(int(actor_sd[f"net.{idx}.weight"].shape[0]))
        idx += 2
    if not hidden:
        raise ValueError(
            "Cannot infer flat_mlp hidden layers from actor state_dict"
        )
    return {
        "name": "flat_mlp",
        "version": 1,
        "obs_dim": int(obs_dim),
        "hidden_layers": hidden,
        "activation": "relu",
        "action_dim": int(action_dim),
    }


def resolve_actor_architecture_from_payload(payload: dict) -> dict:
    """Return the concrete actor architecture encoded by a sensor artifact."""
    from standalone_trainer import normalize_actor_architecture

    arch = payload.get("actor_architecture")
    if isinstance(arch, dict):
        return normalize_actor_architecture(arch)
    version = int(payload.get("policy_format_version", 0))
    if version >= 3:
        raise ValueError(
            "Format-3 sensor policy artifact is missing actor_architecture metadata."
        )
    actor = payload.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("Sensor policy artifact is missing the actor state_dict.")
    if any(key.startswith("encoder.") for key in actor):
        raise ValueError(
            "CNN actor weights require format-3 actor_architecture metadata; "
            "there is no format-2 CNN compatibility path."
        )
    obs_dim = payload.get("actor_obs_dim", payload.get("obs_dim"))
    action_dim = payload.get("action_dim", 2)
    if obs_dim is None:
        raise ValueError(
            "Sensor policy artifact is missing actor_obs_dim/obs_dim for "
            "flat format-2 architecture inference."
        )
    return flat_architecture_from_state_dict(
        actor, obs_dim=int(obs_dim), action_dim=int(action_dim)
    )


def load_sensor_actor_bundle(
    ckpt: str | Path,
    device: torch.device,
    *,
    expected_actor_obs_dim: int,
    expected_action_dim: int,
    expected_layout_version: int,
    norm_eps: float = 1e-8,
    norm_clip: float = 10.0,
    expected_architecture: dict | None = None,
    expected_critic_obs_dim: int | None = None,
    require_obs_norm: bool = False,
):
    """Construct a concrete actor + 1,093-D normalizer from artifact metadata."""
    from standalone_trainer import (
        ObsNormalizer,
        architectures_match,
        normalize_actor_architecture,
        reference_actor_from_architecture,
        validate_sensor_policy_artifact,
    )

    path = Path(ckpt)
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = resolve_actor_architecture_from_payload(payload)
    if expected_architecture is not None and not architectures_match(
        architecture, expected_architecture
    ):
        raise ValueError(
            f"{path.name}: actor_architecture mismatch: "
            f"got {normalize_actor_architecture(architecture)!r}, "
            f"expected {normalize_actor_architecture(expected_architecture)!r}"
        )
    validate_sensor_policy_artifact(
        payload,
        expected_actor_obs_dim=int(expected_actor_obs_dim),
        expected_action_dim=int(expected_action_dim),
        expected_layout_version=int(expected_layout_version),
        expected_architecture=architecture,
        expected_critic_obs_dim=expected_critic_obs_dim,
    )
    actor = reference_actor_from_architecture(architecture).to(
        device=device, dtype=torch.float32
    )
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    normalizer = ObsNormalizer(
        obs_dim=int(expected_actor_obs_dim),
        device=device,
        eps=float(norm_eps),
        clip=float(norm_clip),
    )
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    elif require_obs_norm:
        raise ValueError(f"{path.name}: sensor policy artifact is missing obs_norm")
    return actor, normalizer, architecture, payload


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

"""Per-episode domain randomization for training and nominal evaluation."""

from __future__ import annotations

from typing import Any

import torch

from . import runtime as rt


def _uniform(
    low: float, high: float, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=rt.tc_float) * (high - low) + low


def init_dr_state(
    num_envs: int,
    env_cfg: dict[str, Any],
    device: torch.device,
    num_actions: int,
    *,
    base_vehicle_mass: float,
    base_tire_friction: float,
    base_action_latency: int,
) -> dict[str, Any]:
    """Allocate per-env DR buffers at nominal (disabled) values."""
    dr_cfg = env_cfg.get("domain_randomization") or {}
    max_latency = int(dr_cfg.get("action_latency_steps_max", 3))
    max_latency = max(max_latency, base_action_latency, 1)

    return {
        "cfg": dr_cfg,
        "enabled": bool(dr_cfg.get("enabled", False)),
        "base_vehicle_mass": float(base_vehicle_mass),
        "base_tire_friction": float(base_tire_friction),
        "base_action_latency": int(base_action_latency),
        "max_latency": max_latency,
        "tire_friction": torch.full(
            (num_envs,), base_tire_friction, dtype=rt.tc_float, device=device
        ),
        "vehicle_mass": torch.full(
            (num_envs,), base_vehicle_mass, dtype=rt.tc_float, device=device
        ),
        "mass_scale": torch.ones((num_envs,), dtype=rt.tc_float, device=device),
        # drive_scale multiplies the drive force (motor/gearing strength) and
        # steer_bias (rad) adds a steering-alignment offset.
        "drive_scale": torch.ones((num_envs,), dtype=rt.tc_float, device=device),
        "steer_bias": torch.zeros((num_envs,), dtype=rt.tc_float, device=device),
        "action_latency_steps": torch.full(
            (num_envs,), base_action_latency, dtype=torch.int32, device=device
        ),
        "obs_latency_steps": torch.zeros(
            (num_envs,), dtype=torch.int32, device=device
        ),
        "obs_noise_std": torch.zeros((num_envs,), dtype=rt.tc_float, device=device),
        "action_history": torch.zeros(
            (num_envs, max_latency + 1, num_actions),
            dtype=rt.tc_float,
            device=device,
        ),
        "obs_history": None,
        "num_obs": 0,
    }


def sample_dr_on_reset(
    dr: dict[str, Any],
    reset_mask: torch.Tensor,
    device: torch.device,
) -> None:
    """Resample DR parameters for env rows in ``reset_mask``."""
    if not dr["enabled"]:
        return

    cfg = dr["cfg"]
    n = int(reset_mask.sum().item())
    if n == 0:
        return

    def _range(key: str, default: tuple[float, float]) -> tuple[float, float]:
        val = cfg.get(key, default)
        return float(val[0]), float(val[1])

    tf_lo, tf_hi = _range("tire_friction_range", (0.6, 0.85))
    mass_lo, mass_hi = _range("vehicle_mass_range", (3.2, 4.2))
    scale_lo, scale_hi = _range("mass_scale_range", (0.9, 1.1))
    act_lat_lo, act_lat_hi = _range(
        "action_latency_steps_range", (0.0, float(dr["base_action_latency"]))
    )
    obs_lat_lo, obs_lat_hi = _range("obs_latency_steps_range", (0.0, 0.0))
    noise_lo, noise_hi = _range("obs_noise_std_range", (0.0, 0.02))
    drive_lo, drive_hi = _range("drive_scale_range", (1.0, 1.0))
    steer_bias_lo, steer_bias_hi = _range("steer_bias_range", (0.0, 0.0))

    act_lat_hi = max(act_lat_hi, act_lat_lo)
    obs_lat_hi = max(obs_lat_hi, obs_lat_lo)

    dr["tire_friction"][reset_mask] = _uniform(tf_lo, tf_hi, (n,), device)
    dr["vehicle_mass"][reset_mask] = _uniform(mass_lo, mass_hi, (n,), device)
    dr["mass_scale"][reset_mask] = _uniform(scale_lo, scale_hi, (n,), device)
    dr["drive_scale"][reset_mask] = _uniform(drive_lo, drive_hi, (n,), device)
    dr["steer_bias"][reset_mask] = _uniform(
        steer_bias_lo, steer_bias_hi, (n,), device
    )
    dr["action_latency_steps"][reset_mask] = _uniform(
        act_lat_lo, act_lat_hi + 1.0, (n,), device
    ).to(torch.int32)
    dr["obs_latency_steps"][reset_mask] = _uniform(
        obs_lat_lo, obs_lat_hi + 1.0, (n,), device
    ).to(torch.int32)
    dr["obs_noise_std"][reset_mask] = _uniform(noise_lo, noise_hi, (n,), device)

    dr["action_history"][reset_mask] = 0.0
    if dr["obs_history"] is not None:
        dr["obs_history"][reset_mask] = 0.0


def latency_actions(
    dr: dict[str, Any], actions: torch.Tensor
) -> torch.Tensor:
    """Return per-env delayed actions from the rolling history buffer."""
    hist = dr["action_history"]
    hist.copy_(torch.roll(hist, shifts=1, dims=1))
    hist[:, 0] = actions

    lat = dr["action_latency_steps"]
    lat = lat.clamp(0, hist.shape[1] - 1)
    batch = torch.arange(actions.shape[0], device=actions.device)
    return hist[batch, lat]


def apply_obs_dr(
    dr: dict[str, Any], obs: torch.Tensor
) -> torch.Tensor:
    """Optional observation delay and additive Gaussian noise."""
    if dr["obs_history"] is None:
        dr["obs_history"] = obs.new_zeros(
            (obs.shape[0], dr["max_latency"] + 1, obs.shape[1])
        )

    hist = dr["obs_history"]
    hist.copy_(torch.roll(hist, shifts=1, dims=1))
    hist[:, 0] = obs

    if dr["enabled"]:
        lat = dr["obs_latency_steps"].clamp(0, hist.shape[1] - 1)
        batch = torch.arange(obs.shape[0], device=obs.device)
        out = hist[batch, lat].clone()
        if dr["obs_noise_std"].any():
            noise = torch.randn_like(out) * dr["obs_noise_std"].unsqueeze(1)
            out = out + noise
        return out

    return obs


def dr_metrics(dr: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Expose current DR parameters for logging/tests."""
    return {
        "dr/tire_friction": dr["tire_friction"],
        "dr/vehicle_mass": dr["vehicle_mass"],
        "dr/mass_scale": dr["mass_scale"],
        "dr/drive_scale": dr["drive_scale"],
        "dr/steer_bias": dr["steer_bias"],
        "dr/action_latency_steps": dr["action_latency_steps"].to(rt.tc_float),
        "dr/obs_latency_steps": dr["obs_latency_steps"].to(rt.tc_float),
        "dr/obs_noise_std": dr["obs_noise_std"],
    }

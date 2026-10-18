"""Own-agent Gigaflow conditioning (Cusumano-Towner et al. 2025, arXiv 2502.03349v1).

X(a) draws Uniform(1/a, 1) or Uniform(1, a) with equal probability. C is
concatenated after the baseline 1097-d actor / 392-d critic vectors.
"""

from __future__ import annotations

import numpy as np

GIGAFLOW_C_DIM = 10
GIGAFLOW_ACTOR_OBS_DIM = 1097 + GIGAFLOW_C_DIM
GIGAFLOW_CRITIC_OBS_DIM = 392 + GIGAFLOW_C_DIM

C_LENGTH = 0
C_WIDTH = 1
C_THROTTLE = 2
C_STEER = 3
C_ACCEL = 4
C_VEL = 5
C_COLLISION = 6
C_BOUNDARY = 7
C_COMFORT = 8
C_LANE = 9

GIGAFLOW_C_NAMES = (
    "car_length",
    "car_width",
    "throttle_response",
    "steer_response",
    "accel_limit",
    "speed_limit",
    "collision",
    "boundary",
    "comfort",
    "lane_alignment",
)

GIGAFLOW_C_DYNAMICS = {
    "throttle_response": "X(1.25)",
    "steer_response": "X(1.25)",
    "accel_limit": "X(1.5)",
    "speed_limit": "X(1.5)",
}
GIGAFLOW_C_REWARD = {
    "collision": "X(2) on crash",
    "boundary": "X(2) on wall/oob",
    "comfort": "X(2) on steer change/history",
    "lane_alignment": "X(2) on progress",
}
GIGAFLOW_TABLE_A3 = {
    "value_clip": 0.0,
    "max_grad_norm": 0.5,
    "value_loss": "0.5 * MSE (unclipped)",
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "lr_schedule": "cosine from 5e-4",
    "advantage_normalization": True,
    "advantage_filter": "running_max eta=0.01",
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "orthogonal_init": True,
    "caps_spatial_coef": 0.1,
}

_THROTTLE_A = 1.25
_STEER_A = 1.25
_ACCEL_A = 1.5
_VEL_A = 1.5
_REWARD_A = 2.0


def sample_x(rng: np.random.Generator, a: float, size=None):
    a = float(a)
    if a <= 1.0 or not np.isfinite(a):
        raise ValueError(f"X(a) requires a>1, got {a!r}")
    if size is None:
        if rng.random() < 0.5:
            return float(rng.uniform(1.0 / a, 1.0))
        return float(rng.uniform(1.0, a))
    n = int(size)
    lo = rng.uniform(1.0 / a, 1.0, size=n)
    hi = rng.uniform(1.0, a, size=n)
    pick_lo = rng.random(n) < 0.5
    return np.where(pick_lo, lo, hi).astype(np.float32)


def sample_agent_conditioning(
    rng: np.random.Generator,
    n: int,
    *,
    car_length: float,
    car_width: float,
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    out = np.zeros((n, GIGAFLOW_C_DIM), dtype=np.float32)
    out[:, C_LENGTH] = np.float32(car_length)
    out[:, C_WIDTH] = np.float32(car_width)
    out[:, C_THROTTLE] = sample_x(rng, _THROTTLE_A, size=n)
    out[:, C_STEER] = sample_x(rng, _STEER_A, size=n)
    out[:, C_ACCEL] = sample_x(rng, _ACCEL_A, size=n)
    out[:, C_VEL] = sample_x(rng, _VEL_A, size=n)
    out[:, C_COLLISION] = sample_x(rng, _REWARD_A, size=n)
    out[:, C_BOUNDARY] = sample_x(rng, _REWARD_A, size=n)
    out[:, C_COMFORT] = sample_x(rng, _REWARD_A, size=n)
    out[:, C_LANE] = sample_x(rng, _REWARD_A, size=n)
    return out


def speed_cap_from_c(c_vel, nominal_speed_mps: float):
    return np.float32(nominal_speed_mps) * np.asarray(c_vel, dtype=np.float32)


def format_gigaflow_spec() -> str:
    dyn = ", ".join(f"{k}={v}" for k, v in GIGAFLOW_C_DYNAMICS.items())
    rew = ", ".join(f"{k}={v}" for k, v in GIGAFLOW_C_REWARD.items())
    knobs = ", ".join(f"{k}={v}" for k, v in GIGAFLOW_TABLE_A3.items())
    return (
        f"C_dynamics: {dyn}\n"
        f"C_reward: {rew}\n"
        f"Table A3: {knobs}"
    )

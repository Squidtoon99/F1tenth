from __future__ import annotations

import math


def integrate_steering_delta(
    current_steer_rad: float,
    normalized_delta: float,
    *,
    delta_max_rad: float,
    max_steer: float,
    clip_actions: float = 1.0,
) -> float:
    if not math.isfinite(current_steer_rad) or not math.isfinite(normalized_delta):
        return 0.0
    delta = max(-clip_actions, min(clip_actions, float(normalized_delta)))
    return max(
        -float(max_steer),
        min(
            float(max_steer),
            float(current_steer_rad) + delta * float(delta_max_rad),
        ),
    )


def steer_history_angles_and_deltas(
    steer_history: tuple[float, float, float, float] | list[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Map length-4 executed steer history to layout angles and consecutive deltas."""
    if len(steer_history) < 4:
        raise ValueError(
            f"steer_history length={len(steer_history)}; expected 4"
        )
    t0 = float(steer_history[0])
    t1 = float(steer_history[1])
    t2 = float(steer_history[2])
    t3 = float(steer_history[3])
    return (t0, t1, t2), (t0 - t1, t1 - t2, t2 - t3)

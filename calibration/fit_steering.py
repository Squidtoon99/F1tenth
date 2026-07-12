"""Steering gain / lag / effective max-steer fits."""

from __future__ import annotations

import numpy as np

from bag_io import linear_fit, resample


def fit_servo_map(ackermann: np.ndarray, servo: np.ndarray) -> dict:
    """Fit ``servo = gain * steer + offset`` from commanded steer vs servo."""
    if ackermann.shape[0] < 20 or servo.shape[0] < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = ackermann[:, 0]
    steer = ackermann[:, 2]
    servo_r = resample(servo, t)[:, 1]
    moving = np.abs(np.gradient(steer, t)) > 1e-4
    # Prefer samples spanning the steer range, not only zeros.
    use = moving | (np.abs(steer) > 0.02)
    if use.sum() < 20:
        use = np.ones_like(steer, dtype=bool)

    gain, offset, r2 = linear_fit(steer[use], servo_r[use])
    return {
        "status": "identified" if r2 > 0.95 else "weakly_identified",
        "steering_angle_to_servo_gain": gain,
        "steering_angle_to_servo_offset": offset,
        "r2": r2,
        "steer_min": float(np.min(steer)),
        "steer_max": float(np.max(steer)),
        "servo_min": float(np.min(servo_r)),
        "servo_max": float(np.max(servo_r)),
    }


def fit_max_steer_from_circles(
    odom: np.ndarray,
    ackermann: np.ndarray,
    wheelbase: float = 0.325,
    v_gate: float = 0.8,
) -> dict:
    """Estimate effective max steer from yaw = v * tan(delta) / L segments."""
    if odom.shape[0] < 50 or ackermann.shape[0] < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = odom[:, 0]
    vx = odom[:, 1]
    yaw_rate = odom[:, 2]
    cmd = resample(ackermann, t)
    steer_cmd = cmd[:, 2]
    speed = np.abs(vx)

    # Prefer steady-ish cornering: nonzero yaw, moving, relatively constant steer.
    dsteer = np.abs(np.gradient(steer_cmd, t))
    mask = (speed > v_gate) & (np.abs(yaw_rate) > 0.1) & (dsteer < 0.2)
    if mask.sum() < 30:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_steady_cornering"}

    # delta_eff = atan(L * yaw / v), using signed quantities carefully.
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.arctan(wheelbase * yaw_rate / np.where(np.abs(vx) < 1e-3, np.nan, vx))
    delta = delta[mask]
    delta = delta[np.isfinite(delta)]
    if delta.size < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "degenerate_delta"}

    max_steer = float(np.percentile(np.abs(delta), 95))
    cmd_max = float(np.percentile(np.abs(steer_cmd[mask]), 95))
    return {
        "status": "weakly_identified",
        "max_steer": max_steer,
        "commanded_steer_p95": cmd_max,
        "wheelbase_assumed": wheelbase,
        "n_samples": int(delta.size),
    }


def fit_steer_lag(
    ackermann: np.ndarray,
    odom: np.ndarray,
    control_dt: float = 0.1,
) -> dict:
    """Estimate first-order steer lag ``t_delta`` from command vs yaw response."""
    if ackermann.shape[0] < 50 or odom.shape[0] < 50:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = odom[:, 0]
    yaw_rate = odom[:, 2]
    cmd = resample(ackermann, t)
    steer = cmd[:, 2]
    # Normalize both to unit scale for lag identification on rising edges.
    if np.std(steer) < 1e-4 or np.std(yaw_rate) < 1e-4:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_excitation"}

    steer_n = steer / (np.max(np.abs(steer)) + 1e-9)
    yaw_n = yaw_rate / (np.max(np.abs(yaw_rate)) + 1e-9)

    best = None
    for t_delta in np.linspace(0.02, 0.4, 40):
        alpha = control_dt / (t_delta + control_dt)
        filt = np.zeros_like(steer_n)
        for i in range(1, len(steer_n)):
            filt[i] = filt[i - 1] + alpha * (steer_n[i] - filt[i - 1])
        err = float(np.mean((filt - yaw_n) ** 2))
        if best is None or err < best[0]:
            best = (err, float(t_delta), float(alpha))

    assert best is not None
    return {
        "status": "weakly_identified",
        "t_delta": best[1],
        "lag_alpha": best[2],
        "mse": best[0],
        "control_dt": control_dt,
    }

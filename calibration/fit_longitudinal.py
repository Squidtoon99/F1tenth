"""Longitudinal / ERPM / friction / rolling-resistance fits."""

from __future__ import annotations

import numpy as np

from bag_io import G, detect_accel_scale, linear_fit, resample


def not_identifiable(reason: str) -> dict:
    return {"status": "NOT_IDENTIFIABLE", "reason": reason}


def fit_speed_to_erpm(ackermann: np.ndarray, motor_speed: np.ndarray) -> dict:
    """Fit ``erpm = gain * speed + offset`` on near-steady samples."""
    if ackermann.shape[0] < 20 or motor_speed.shape[0] < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = ackermann[:, 0]
    speed = ackermann[:, 1]
    erpm = resample(motor_speed, t)[:, 1]
    accel = np.gradient(speed, t)
    steady = np.abs(accel) < 0.5
    if steady.sum() < 20:
        steady = np.ones_like(speed, dtype=bool)

    gain, offset, r2 = linear_fit(speed[steady], erpm[steady])
    return {
        "status": "identified" if r2 > 0.99 else "weakly_identified",
        "speed_to_erpm_gain": gain,
        "speed_to_erpm_offset": offset,
        "r2": r2,
        "n_samples": int(steady.sum()),
    }


def fit_odom_sign_and_scale(odom: np.ndarray, ackermann: np.ndarray) -> dict:
    """Compare odom vx sign/magnitude against commanded speed."""
    if odom.shape[0] < 20 or ackermann.shape[0] < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = odom[:, 0]
    vx = odom[:, 1]
    cmd = resample(ackermann, t)[:, 1]
    moving = np.abs(cmd) > 0.5
    if moving.sum() < 20:
        return {
            "status": "weakly_identified",
            "odom_vx_min": float(np.min(vx)),
            "odom_vx_max": float(np.max(vx)),
            "odom_vx_mean": float(np.mean(vx)),
            "note": "little commanded motion",
        }

    # Correlation of signs: positive correlation => same sign convention.
    corr = float(np.corrcoef(cmd[moving], vx[moving])[0, 1])
    scale, _, r2 = linear_fit(cmd[moving], vx[moving])
    return {
        "status": "identified" if abs(corr) > 0.7 else "weakly_identified",
        "cmd_odom_corr": corr,
        "odom_vs_cmd_scale": float(scale),
        "r2": float(r2),
        "odom_vx_min": float(np.min(vx)),
        "odom_vx_max": float(np.max(vx)),
        "sign_matches_command": bool(corr > 0.0),
    }


def fit_accel_limit(odom: np.ndarray, ackermann: np.ndarray) -> dict:
    """Summarize the observed longitudinal acceleration envelope from odom."""
    if odom.shape[0] < 50:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = odom[:, 0]
    vx = odom[:, 1]
    ax = np.gradient(vx, t)
    cmd = resample(ackermann, t)[:, 1] if ackermann.shape[0] else np.zeros_like(vx)

    # Gate on nontrivial motion / command.
    mask = (np.abs(vx) > 0.2) | (np.abs(cmd) > 0.2)
    if mask.sum() < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_motion"}

    ax_sel = ax[mask]
    return {
        "status": "weakly_identified",
        "ax_p95": float(np.percentile(np.abs(ax_sel), 95)),
        "ax_p99": float(np.percentile(np.abs(ax_sel), 99)),
        "ax_max": float(np.max(np.abs(ax_sel))),
        "ax_min": float(np.min(ax_sel)),
        "ax_max_signed": float(np.max(ax_sel)),
    }


def fit_f_drive_equivalent(odom: np.ndarray, mass: float = 3.74) -> dict:
    """Low-speed plateau accel -> equivalent drive force f ≈ m a."""
    if odom.shape[0] < 50:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = odom[:, 0]
    vx = odom[:, 1]
    ax = np.gradient(vx, t)
    # Low-speed positive accel plateaus (launch).
    mask = (np.abs(vx) < 2.0) & (ax > 0.5)
    if mask.sum() < 10:
        # Fall back to top positive accel overall.
        pos = ax[ax > 0.2]
        if pos.size < 5:
            return {"status": "NOT_IDENTIFIABLE", "reason": "no_launch_segment"}
        a_plat = float(np.percentile(pos, 90))
    else:
        a_plat = float(np.percentile(ax[mask], 90))

    return {
        "status": "weakly_identified",
        "a_plateau": a_plat,
        "f_drive_max": float(mass * a_plat),
        "mass": mass,
        "note": "speed-loop equivalent, not open-loop force",
    }


def fit_c_roll(odom: np.ndarray, ackermann: np.ndarray, mass: float = 3.74) -> dict:
    """Coast segments where command ~ 0 and speed decays."""
    if odom.shape[0] < 50 or ackermann.shape[0] < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    t = odom[:, 0]
    vx = odom[:, 1]
    ax = np.gradient(vx, t)
    cmd = resample(ackermann, t)[:, 1]
    coast = (np.abs(cmd) < 0.15) & (np.abs(vx) > 0.5) & (ax < -0.05)
    if coast.sum() < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_coast_segment"}

    # Rolling resistance force ~ m * |ax| while coasting.
    f = mass * np.abs(ax[coast])
    return {
        "status": "weakly_identified",
        "c_roll": float(np.median(f)),
        "c_roll_p90": float(np.percentile(f, 90)),
        "n_samples": int(coast.sum()),
    }


def fit_tire_mu_lower_bound(
    imu: np.ndarray,
    odom: np.ndarray,
    static_imu: np.ndarray | None = None,
    v_gate: float = 1.0,
    smooth_s: float = 0.5,
) -> dict:
    """Sustained horizontal acceleration / g while moving — lower bound on mu."""
    if imu.shape[0] < 20 or odom.shape[0] < 20:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_samples"}

    v = np.abs(odom[:, 1])
    v_at_imu = np.interp(imu[:, 0], odom[:, 0], v)
    calibration_imu = (
        static_imu if static_imu is not None and static_imu.shape[0] >= 10 else imu
    )
    az = float(np.median(calibration_imu[:, 3]))
    scale, _ = detect_accel_scale(az)
    bias_xy = np.median(calibration_imu[:, 1:3], axis=0)
    accel_xy = (imu[:, 1:3] - bias_xy) * scale

    dt = float(np.median(np.diff(imu[:, 0])))
    window = max(1, int(round(smooth_s / max(dt, 1e-6))))
    if window % 2 == 0:
        window += 1
    half = window // 2
    sustained_xy = np.column_stack(
        [
            np.median(
                np.lib.stride_tricks.sliding_window_view(
                    np.pad(accel_xy[:, axis], half, mode="edge"),
                    window,
                ),
                axis=1,
            )
            for axis in range(2)
        ]
    )
    a_horiz = np.linalg.norm(sustained_xy, axis=1)
    fast = v_at_imu > v_gate
    if half:
        fast[:half] = False
        fast[-half:] = False
    if fast.sum() < 10:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_moving_imu"}

    peak = float(np.percentile(a_horiz[fast], 99))
    raw_peak = float(np.percentile(np.linalg.norm(accel_xy[fast], axis=1), 99))
    return {
        "status": "weakly_identified",
        "tire_friction_lower_bound": peak / G,
        "a_horiz_p99": peak,
        "a_horiz_raw_p99": raw_peak,
        "accel_scale_used": scale,
        "accel_bias_raw_xy": bias_xy.tolist(),
        "smooth_window_s": float(window * dt),
    }


def fit_current_to_force(
    motor_current: np.ndarray,
    odom: np.ndarray,
    mass: float = 3.74,
) -> dict:
    """Fit open-loop current→accel gain from boxed-wheel / sensors-free odom.

    Uses commanded or measured current vs odom ax. Prefer bags that also have
    /sensors/core for measured current; commanded current is a fallback.
    """
    if motor_current.shape[0] < 20 or odom.shape[0] < 50:
        return not_identifiable("insufficient_current_or_odom")

    t = odom[:, 0]
    vx = odom[:, 1]
    ax = np.gradient(vx, t)
    i = resample(motor_current, t)[:, 1]
    mask = np.abs(i) > 0.2
    if mask.sum() < 20:
        return not_identifiable("no_nonzero_current")

    # a ≈ k * I  =>  f ≈ m k I  => f_drive_max ≈ m k i_max_used
    k, _, r2 = linear_fit(i[mask], ax[mask])
    i_abs_max = float(np.max(np.abs(i)))
    return {
        "status": "identified" if r2 > 0.5 else "weakly_identified",
        "accel_per_amp": float(k),
        "force_per_amp": float(mass * k),
        "f_drive_max_at_iabs": float(mass * abs(k) * i_abs_max),
        "i_abs_max_in_bag": i_abs_max,
        "r2": float(r2),
        "mass": mass,
    }


def fit_brake_current(
    motor_brake: np.ndarray,
    odom: np.ndarray,
    mass: float = 3.74,
) -> dict:
    """Fit brake current → deceleration from no-load / low-speed bags."""
    if motor_brake.shape[0] < 20 or odom.shape[0] < 50:
        return not_identifiable("insufficient_brake_or_odom")

    t = odom[:, 0]
    vx = odom[:, 1]
    ax = np.gradient(vx, t)
    b = resample(motor_brake, t)[:, 1]
    mask = (b > 0.2) & (np.abs(vx) > 0.3)
    if mask.sum() < 15:
        return not_identifiable("no_brake_segment")

    # Expect ax negative while braking; fit |ax| ≈ k * brake.
    k, _, r2 = linear_fit(b[mask], -ax[mask])
    b_max = float(np.max(b))
    return {
        "status": "identified" if r2 > 0.5 else "weakly_identified",
        "decel_per_amp": float(k),
        "f_brake_max_at_bmax": float(mass * abs(k) * b_max),
        "brake_max_in_bag": b_max,
        "r2": float(r2),
        "mass": mass,
    }

"""Planar vehicle dynamics: tyre forces at 4 patches -> body + wheel-spin ODEs.

Everything is vectorized over the env batch ``N`` and the 4 wheels ``[LR, RR, LF, RF]``.
The integrator is semi-implicit Euler (update velocities from forces at the current
state, then advance positions with the new velocities) which is stable for the
stiff tyre/wheel-spin coupling at ``sim_dt`` without the ringing of explicit Euler.
"""

from __future__ import annotations

import torch

from .drivetrain import wheel_axle_torques
from .suspension import quasi_static_loads


def ackermann_wheel_angles(params, delta_center: torch.Tensor) -> torch.Tensor:
    """Left/right front wheel angles (N,2) from a centre steer angle (N,)."""
    small = delta_center.abs() < 1e-6
    tan = torch.tan(delta_center)
    tan = torch.where(small, torch.ones_like(tan), tan)
    r = params.wheelbase / tan
    half = params.track_width / 2.0
    delta_left = torch.atan(params.wheelbase / (r - half))
    delta_right = torch.atan(params.wheelbase / (r + half))
    zero = torch.zeros_like(delta_center)
    delta_left = torch.where(small, zero, delta_left)
    delta_right = torch.where(small, zero, delta_right)
    return torch.stack([delta_left, delta_right], dim=-1)


def wheel_offsets(params, device, dtype) -> torch.Tensor:
    return torch.tensor(params.wheel_xy, device=device, dtype=dtype)


def wheel_frame_velocities(
    params,
    vx: torch.Tensor,
    vy: torch.Tensor,
    r: torch.Tensor,
    delta_wheel: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-wheel longitudinal/lateral contact velocity (N,4) in each wheel frame."""
    x_i = offsets[:, 0]
    y_i = offsets[:, 1]
    vx_i = vx.unsqueeze(1) - r.unsqueeze(1) * y_i
    vy_i = vy.unsqueeze(1) + r.unsqueeze(1) * x_i
    cd = torch.cos(delta_wheel)
    sd = torch.sin(delta_wheel)
    v_long = cd * vx_i + sd * vy_i
    v_lat = -sd * vx_i + cd * vy_i
    return v_long, v_lat


def step_dynamic(state, params, tire, dt, susp_filter=None):
    """Advance the dynamic model one ``dt``. Mutates and returns ``state``.

    ``state`` is a dict of (N,)/(N,4) tensors: X, Y, yaw, vx, vy, r, omega, steer,
    throttle, mass, mu. Returns diagnostics used for slip readback and obs accel.
    """
    device, dtype = state["vx"].device, state["vx"].dtype
    offsets = wheel_offsets(params, device, dtype)
    r_wheel = params.wheel_radius
    mass = state["mass"]
    mu = state["mu"]

    delta_lr = ackermann_wheel_angles(params, state["steer"])
    delta_wheel = torch.zeros_like(state["omega"])
    delta_wheel[:, 2] = delta_lr[:, 0]
    delta_wheel[:, 3] = delta_lr[:, 1]

    v_long, v_lat = wheel_frame_velocities(
        params, state["vx"], state["vy"], state["r"], delta_wheel, offsets
    )

    eps = params.v_eps
    v_blend = max(params.low_speed_blend, eps)
    # Modern-PhysX slip (VhTireFunctions.cpp): normalize by |v_long| + a fixed
    # offset rather than the legacy max(|wheel_speed|, |v_long|). The longitudinal
    # offset switches between an active value (drive or brake torque applied) and a
    # larger passive value (coasting). Keep the -v_lat sign for the lateral force
    # (the tyre force opposes the contact-patch lateral velocity); the observation
    # slip angle uses the opposite (geometric) sign, see slip_angle_obs below.
    v_long_abs = v_long.abs()
    active = (state["throttle"].abs() > eps).unsqueeze(1)
    min_long = torch.where(
        active,
        v_long.new_full((), params.slip_min_active_long),
        v_long.new_full((), params.slip_min_passive_long),
    )
    denom = v_long_abs + min_long
    alpha = torch.atan2(-v_lat, v_long_abs + params.slip_min_lat)
    wheel_speed = r_wheel * state["omega"]
    kappa = (wheel_speed - v_long) / denom

    # Body accelerations from the previous substep drive the (quasi-static) load
    # transfer; on the first call ax/ay default to 0 (static split).
    ax_prev = state.get("ax")
    ay_prev = state.get("ay")
    if ax_prev is None:
        ax_prev = torch.zeros_like(state["vx"])
        ay_prev = torch.zeros_like(state["vx"])
    if susp_filter is not None:
        Fz = susp_filter.step(ax_prev, ay_prev, dt, mass)
    else:
        Fz = quasi_static_loads(params, ax_prev, ay_prev, mass)

    mu_w = mu.unsqueeze(1).expand_as(Fz)
    peak = tire.load_scaled_mu(Fz, mu_w).clamp_min(1e-4) * Fz
    fx_w, fy_w = tire.forces(kappa, alpha, Fz, mu_w)

    # Optional tyre relaxation: first-order lag of the contact forces with a
    # speed-dependent time constant tau = relax_len / |v|, modelling the finite
    # distance a tyre must roll to build up force.
    if params.tire_relax_len > 0.0 and "fx_lag" in state:
        tau = params.tire_relax_len / v_long.abs().clamp_min(v_blend)
        beta = dt / (tau + dt)
        fx_w = state["fx_lag"] + beta * (fx_w - state["fx_lag"])
        fy_w = state["fy_lag"] + beta * (fy_w - state["fy_lag"])
        state["fx_lag"] = fx_w
        state["fy_lag"] = fy_w

    cd = torch.cos(delta_wheel)
    sd = torch.sin(delta_wheel)
    fx_b = cd * fx_w - sd * fy_w
    fy_b = sd * fx_w + cd * fy_w

    fx_total = fx_b.sum(dim=1)
    fy_total = fy_b.sum(dim=1)
    mz = (offsets[:, 0] * fy_b - offsets[:, 1] * fx_b).sum(dim=1)

    if params.enable_aero_drag and params.dragcoeff > 0.0:
        speed = torch.sqrt(state["vx"] ** 2 + state["vy"] ** 2).clamp_min(1e-6)
        fx_total = fx_total - params.dragcoeff * speed * state["vx"]
        fy_total = fy_total - params.dragcoeff * speed * state["vy"]

    ax = fx_total / mass
    ay = fy_total / mass
    dr = mz / params.izz

    tau_axle = wheel_axle_torques(
        params, state["throttle"], state["vx"], state["omega"], mu, mass,
        drive_scale=state.get("drive_scale"),
    )
    # Wheel-spin ODE is stiff (tiny wheel inertia + steep tyre-slip slope). Use a
    # linearized-implicit (backward-Euler) update so it is unconditionally stable
    # at sim_dt instead of requiring a tiny explicit step. The tyre longitudinal
    # slope d(Fx)/d(omega) ~ peak * B * C * R / denom damps the update.
    iw = params.wheel_inertia
    g_tau = tau_axle - r_wheel * fx_w
    k_tau = peak * params.tire_B_long * params.tire_C_long * (r_wheel ** 2) / denom
    omega = state["omega"] + (dt * g_tau / iw) / (1.0 + dt * k_tau / iw)

    vx = state["vx"] + dt * (ax + state["r"] * state["vy"])
    vy = state["vy"] + dt * (ay - state["r"] * state["vx"])
    r = state["r"] + dt * dr

    yaw = state["yaw"] + dt * r
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    state["X"] = state["X"] + dt * (vx * cos_y - vy * sin_y)
    state["Y"] = state["Y"] + dt * (vx * sin_y + vy * cos_y)
    state["yaw"] = yaw
    state["vx"] = vx
    state["vy"] = vy
    state["r"] = r
    state["omega"] = omega
    state["ax"] = ax
    state["ay"] = ay

    return {
        "v_long": v_long,
        "v_lat": v_lat,
        "kappa": kappa,
        "alpha": alpha,
        # Geometric slip angle for the observation (positive when the contact patch
        # slides toward +y), i.e. the sign convention of car.compute_tyre_slip. This
        # is -alpha since alpha carries the force-opposing sign.
        "slip_angle_obs": torch.atan2(v_lat, v_long_abs + params.slip_min_lat),
        "Fz": Fz,
        "fx_w": fx_w,
        "fy_w": fy_w,
    }


def step_kinematic(state, params, dt):
    """Advance the Tier-0 kinematic single-track model one ``dt``."""
    # Force/brake effort → accel with a soft clamp (mass from state when present).
    mass = state.get("mass")
    if mass is None:
        mass = torch.full_like(state["vx"], params.mass)
    throttle = state["throttle"].clamp_min(0.0)
    brake = (-state["throttle"]).clamp_min(0.0)
    f = throttle * params.f_drive_max - brake * params.f_brake_max
    accel = (f / mass.clamp_min(1e-3)).clamp(
        -params.kinematic_accel_limit, params.kinematic_accel_limit
    )
    vx = state["vx"] + dt * accel
    delta = state["steer"]
    r = vx * torch.tan(delta) / params.wheelbase
    yaw = state["yaw"] + dt * r
    state["X"] = state["X"] + dt * vx * torch.cos(yaw)
    state["Y"] = state["Y"] + dt * vx * torch.sin(yaw)
    state["yaw"] = yaw
    state["vx"] = vx
    state["vy"] = torch.zeros_like(vx)
    state["r"] = r
    state["omega"] = (vx / params.wheel_radius).unsqueeze(1).expand_as(state["omega"])
    state["ax"] = accel
    state["ay"] = vx * r

    delta_lr = ackermann_wheel_angles(params, delta)
    delta_wheel = torch.zeros_like(state["omega"])
    delta_wheel[:, 2] = delta_lr[:, 0]
    delta_wheel[:, 3] = delta_lr[:, 1]
    offsets = wheel_offsets(params, vx.device, vx.dtype)
    v_long, v_lat = wheel_frame_velocities(
        params, state["vx"], state["vy"], state["r"], delta_wheel, offsets
    )
    # Rigid rolling (omega = vx / r): longitudinal slip is ~0. Report the geometric
    # slip angle so the observation slip block is consistent with the dynamic model.
    return {
        "v_long": v_long,
        "v_lat": v_lat,
        "kappa": torch.zeros_like(v_long),
        "slip_angle_obs": torch.atan2(v_lat, v_long.abs() + params.slip_min_lat),
    }

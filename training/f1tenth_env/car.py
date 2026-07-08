"""
Vehicle geometry, control setup, and force-based drive/brake model.
"""

import os
from typing import Any

import numpy as np
import torch
import genesis as gs
import genesis.utils.geom as gu

URDF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "F110.export.urdf")
)

WHEEL_JOINTS = [
    "left_rear_wheel_joint",
    "right_rear_wheel_joint",
    "left_front_wheel_joint",
    "right_front_wheel_joint",
]

STEER_JOINTS = [
    "left_steering_hinge_joint",
    "right_steering_hinge_joint",
]

WHEEL_RADIUS = 0.05
WHEELBASE = 0.325
TRACK_WIDTH = 0.20

KP_WHEELS = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
KV_WHEELS = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

KP_STEER = np.array([8.0, 8.0], dtype=np.float32)
KV_STEER = np.array([0.8, 0.8], dtype=np.float32)
EFF_STEER = np.array([10.0, 10.0], dtype=np.float32)

CHASSIS_FRICTION = 0.05


def ackermann_left_right(
    delta_center: torch.Tensor, L: float, W: float
) -> torch.Tensor:
    """Convert center steering angle to left/right Ackermann hinge angles."""
    small_angle_mask = torch.abs(delta_center) < 1e-6

    R = L / torch.tan(delta_center)
    R_left = R - (W / 2.0)
    R_right = R + (W / 2.0)

    delta_left = torch.atan(L / R_left)
    delta_right = torch.atan(L / R_right)

    delta_left = torch.where(small_angle_mask, torch.zeros_like(delta_left), delta_left)
    delta_right = torch.where(
        small_angle_mask, torch.zeros_like(delta_right), delta_right
    )

    return torch.stack([delta_left, delta_right], dim=1)


def compute_tyre_slip(
    wheel_state: dict[str, torch.Tensor],
    wheel_radius: float,
    slip_eps: float = 0.1,
) -> torch.Tensor:
    """
    Per-wheel slip ratio and slip angle for observation / reward.

    Wheel order: [left_rear, right_rear, left_front, right_front].
    Returns (N, 8): [slip_ratio x4, slip_angle x4].
    """
    # motion_link_vel is the wheel link-COM velocity in the WORLD frame. Slip is a
    # wheel-frame quantity (column 0 = forward, column 1 = lateral), so rotate the
    # velocity into each wheel's frame using frame_quat (base_link for the rear
    # wheels, the steering hinge for the front wheels) before splitting it.
    lin_vel = wheel_state["motion_link_vel"]
    frame_quat = wheel_state.get("frame_quat")
    if frame_quat is not None:
        lin_vel_local = gu.inv_transform_by_quat(lin_vel, frame_quat)
    else:
        # No frame given: treat the velocity as already wheel-frame (unit tests).
        lin_vel_local = lin_vel
    spin_rate = wheel_state["dof_vel"]

    v_fwd = lin_vel_local[:, :, 0]
    v_lat = lin_vel_local[:, :, 1]

    slip_angle = torch.atan2(v_lat, torch.abs(v_fwd).clamp_min(slip_eps))

    wheel_speed = wheel_radius * spin_rate
    denom = torch.maximum(torch.abs(wheel_speed), torch.abs(v_fwd)).clamp_min(slip_eps)
    slip_ratio = (wheel_speed - v_fwd) / denom

    return torch.cat([slip_ratio, slip_angle], dim=-1)


def compute_wheel_torques(
    throttle_cmd: torch.Tensor,
    base_lin_vel_body: torch.Tensor,
    wheel_dof_vel: torch.Tensor,
    env_cfg: dict[str, Any],
    vehicle_mass: float | torch.Tensor = 3.74,
    tire_friction: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-wheel drive/brake torques (Nm) from force limits.

    Drive torque is distributed AWD; brake opposes wheel spin; coast applies
    light rolling resistance only. Wheel order: [LR, RR, LF, RF].
    """
    f_drive_max = float(env_cfg.get("f_drive_max", 23.0))
    f_brake_max = float(env_cfg.get("f_brake_max", 23.0))
    power_max = float(env_cfg.get("power_max", 255.0))
    wheel_radius = float(env_cfg.get("wheel_radius", WHEEL_RADIUS))
    k_front = float(env_cfg.get("k_drive_front", 0.5))
    v_eps = float(env_cfg.get("v_eps", 0.1))
    if tire_friction is None:
        mue = torch.full(
            (throttle_cmd.shape[0],),
            float(env_cfg.get("tire_friction", 0.7)),
            dtype=gs.tc_float,
            device=throttle_cmd.device,
        )
    else:
        mue = tire_friction.reshape(-1).to(dtype=gs.tc_float, device=throttle_cmd.device)
    c_roll = float(env_cfg.get("c_roll", 0.0))
    drive_sign = float(env_cfg.get("drive_torque_sign", 1.0))

    throttle = torch.clamp(throttle_cmd, min=0.0)
    brake = torch.clamp(-throttle_cmd, min=0.0)

    v_mag = torch.linalg.norm(base_lin_vel_body[:, :2], dim=-1)

    if isinstance(vehicle_mass, torch.Tensor):
        mass = vehicle_mass.reshape(-1).to(dtype=gs.tc_float, device=throttle_cmd.device)
    else:
        mass = torch.full(
            (throttle_cmd.shape[0],),
            float(vehicle_mass),
            dtype=gs.tc_float,
            device=throttle_cmd.device,
        )
    traction_cap = mue * mass * 9.81
    f_drive = throttle * f_drive_max
    f_drive = torch.minimum(f_drive, power_max / torch.clamp(v_mag, min=v_eps))
    f_drive = torch.minimum(f_drive, traction_cap)

    f_front = k_front * f_drive
    f_rear = (1.0 - k_front) * f_drive

    tau_lr = drive_sign * f_rear * 0.5 * wheel_radius
    tau_rr = drive_sign * f_rear * 0.5 * wheel_radius
    tau_lf = drive_sign * f_front * 0.5 * wheel_radius
    tau_rf = drive_sign * f_front * 0.5 * wheel_radius
    tau_drive = torch.stack([tau_lr, tau_rr, tau_lf, tau_rf], dim=1)

    f_brake = brake * f_brake_max
    brake_sign = torch.sign(wheel_dof_vel)
    brake_sign = torch.where(
        brake_sign == 0, torch.ones_like(brake_sign), brake_sign
    )
    tau_brake = -brake_sign * (f_brake * 0.25 * wheel_radius).unsqueeze(1)

    tau = torch.zeros_like(tau_drive)
    drive_mask = throttle > 1e-3
    brake_mask = brake > 1e-3
    coast_mask = (~drive_mask) & (~brake_mask)

    if drive_mask.any():
        tau = torch.where(drive_mask.unsqueeze(1), tau_drive, tau)
    if brake_mask.any():
        tau = torch.where(brake_mask.unsqueeze(1), tau_brake, tau)

    if c_roll > 0.0 and coast_mask.any():
        roll_sign = torch.sign(wheel_dof_vel)
        roll_sign = torch.where(
            roll_sign == 0, torch.ones_like(roll_sign), roll_sign
        )
        tau = torch.where(
            coast_mask.unsqueeze(1), -roll_sign * c_roll, tau
        )

    return tau


def compute_dissipative_force_world(
    lin_vel_world: torch.Tensor,
    env_cfg: dict[str, Any],
) -> torch.Tensor:
    """
    Dissipative forces on the chassis in world frame (N).

    Aero drag uses F = -dragcoeff * |v| * v so |F| = dragcoeff * v^2.
    """
    force = torch.zeros_like(lin_vel_world)
    vel_xy = lin_vel_world[:, :2]
    speed = torch.linalg.norm(vel_xy, dim=-1, keepdim=True)

    if bool(env_cfg.get("enable_aero_drag", False)):
        dragcoeff = float(env_cfg.get("dragcoeff", 0.075))
        force[:, :2] -= dragcoeff * speed * vel_xy

    c_roll = float(env_cfg.get("c_roll", 0.0))
    if c_roll > 0.0:
        v_fwd = vel_xy[:, 0]
        roll_sign = torch.sign(v_fwd)
        roll_sign = torch.where(roll_sign == 0, torch.ones_like(roll_sign), roll_sign)
        force[:, 0] -= c_roll * roll_sign

    return force


def compute_chassis_longitudinal_force(
    throttle_cmd: torch.Tensor,
    base_lin_vel_body: torch.Tensor,
    env_cfg: dict[str, Any],
    vehicle_mass: float = 3.74,
) -> torch.Tensor:
    """Longitudinal drive/brake force (N) with power and traction caps."""
    f_drive_max = float(env_cfg.get("f_drive_max", 23.0))
    f_brake_max = float(env_cfg.get("f_brake_max", 23.0))
    power_max = float(env_cfg.get("power_max", 255.0))
    v_eps = float(env_cfg.get("v_eps", 0.1))
    mue = float(env_cfg.get("tire_friction", 0.7))
    dragcoeff = float(env_cfg.get("dragcoeff", 0.075))
    enable_drag = bool(env_cfg.get("enable_aero_drag", False))

    throttle = torch.clamp(throttle_cmd, min=0.0)
    brake = torch.clamp(-throttle_cmd, min=0.0)

    v_mag = torch.linalg.norm(base_lin_vel_body[:, :2], dim=-1)
    v_fwd = base_lin_vel_body[:, 0]

    traction_cap = mue * vehicle_mass * 9.81

    f_drive = throttle * f_drive_max
    f_drive = torch.minimum(f_drive, power_max / torch.clamp(v_mag, min=v_eps))
    f_drive = torch.minimum(
        f_drive,
        torch.tensor(traction_cap, dtype=gs.tc_float, device=throttle_cmd.device),
    )

    f_brake = brake * f_brake_max
    f_brake = torch.minimum(
        f_brake,
        torch.tensor(traction_cap, dtype=gs.tc_float, device=throttle_cmd.device),
    )

    f_long = torch.zeros_like(f_drive)
    f_long = torch.where(throttle > 1e-3, f_drive, f_long)
    f_long = torch.where(brake > 1e-3, -f_brake, f_long)

    if enable_drag and dragcoeff > 0.0:
        coast = (throttle <= 1e-3) & (brake <= 1e-3)
        f_drag = -dragcoeff * v_fwd * torch.abs(v_fwd)
        f_long = torch.where(coast, f_drag, f_long)

    return f_long


def chassis_force_to_root_world(
    f_long_body: torch.Tensor,
    base_quat: torch.Tensor,
) -> torch.Tensor:
    """Map body-frame longitudinal force to world-frame root linear force."""
    from genesis.utils.geom import quat_to_xyz

    yaw = quat_to_xyz(base_quat, rpy=True, degrees=False)[:, 2]
    fx = f_long_body * torch.cos(yaw)
    fy = f_long_body * torch.sin(yaw)
    return torch.stack([fx, fy, torch.zeros_like(fx)], dim=-1)


def setup_entity_controls(
    car,
    env_cfg: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Configure wheel torque mode, steering PD, and tire friction from config."""
    env_cfg = env_cfg or {}
    wheel_dofs = []
    for name in WHEEL_JOINTS:
        wheel_dofs.extend(car.get_joint(name).dofs_idx_local)
    steer_dofs = []
    for name in STEER_JOINTS:
        steer_dofs.extend(car.get_joint(name).dofs_idx_local)

    f_drive_max = float(env_cfg.get("f_drive_max", 23.0))
    f_brake_max = float(env_cfg.get("f_brake_max", 23.0))
    wheel_radius = float(env_cfg.get("wheel_radius", WHEEL_RADIUS))
    # Per-wheel torque envelope: half-axle force * radius with margin.
    tau_max = max(f_drive_max, f_brake_max) * wheel_radius * 0.55 * 1.5
    eff_wheels = np.full(4, tau_max, dtype=np.float32)

    car.set_dofs_kp(KP_WHEELS, wheel_dofs)
    car.set_dofs_kv(KV_WHEELS, wheel_dofs)
    car.set_dofs_force_range(-eff_wheels, eff_wheels, wheel_dofs)

    car.set_dofs_kp(KP_STEER, steer_dofs)
    car.set_dofs_kv(KV_STEER, steer_dofs)
    car.set_dofs_force_range(-EFF_STEER, EFF_STEER, steer_dofs)

    c_roll = float(env_cfg.get("c_roll", 0.0))
    car.set_dofs_damping(np.zeros(4, dtype=np.float32), wheel_dofs)
    car.set_dofs_frictionloss(np.zeros(4, dtype=np.float32), wheel_dofs)
    if c_roll > 0.0:
        car.set_dofs_damping(np.full(4, c_roll, dtype=np.float32), wheel_dofs)

    tire_friction = float(env_cfg.get("tire_friction", 0.7))
    car.set_friction(CHASSIS_FRICTION)

    link_names = [
        "base_link",
        "left_rear_wheel",
        "right_rear_wheel",
        "left_front_wheel",
        "right_front_wheel",
    ]
    link_ids = [car.get_link(name).idx_local for name in link_names]

    ratios = np.array(
        [
            1.0,
            tire_friction / CHASSIS_FRICTION,
            tire_friction / CHASSIS_FRICTION,
            tire_friction / CHASSIS_FRICTION,
            tire_friction / CHASSIS_FRICTION,
        ],
        dtype=np.float32,
    )
    ratios_t = torch.from_numpy(ratios[None, :]).to(device=gs.device)
    car.set_friction_ratio(ratios_t, links_idx_local=link_ids)
    return (wheel_dofs, steer_dofs)

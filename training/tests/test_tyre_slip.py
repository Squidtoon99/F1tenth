"""Tyre-slip correctness tests for `f1tenth_env/car.py::compute_tyre_slip`.

Two groups:

1. Formula tests - feed wheel-FRAME velocities (the contract the function assumes:
   column 0 = forward, column 1 = lateral) and assert standard slip ratio/angle.
2. Frame-sensitivity demonstration - show that if the wheel velocity is left in the
   WORLD frame (as the env appears to do: `wheel_state["frame_quat"]` is collected
   but never applied), a wheel that is physically rolling straight reports a slip
   angle equal to the car yaw instead of ~0. This pins the suspected bug.

Wheel order: [left_rear, right_rear, left_front, right_front].
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

WHEEL_RADIUS = 0.05
SLIP_EPS = 0.1


def _wheel_state(motion_link_vel, dof_vel, frame_quat=None):
    state = {
        "motion_link_vel": torch.tensor(motion_link_vel, dtype=torch.float32),
        "dof_vel": torch.tensor(dof_vel, dtype=torch.float32),
    }
    if frame_quat is not None:
        state["frame_quat"] = torch.tensor(frame_quat, dtype=torch.float32)
    return state


def _quat_wxyz_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], np.float32)


def _rotate_world_to_local(vec_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body/local frame (inverse rotation)."""
    w, x, y, z = quat_wxyz
    u = np.array([x, y, z], dtype=np.float64)
    v = vec_world.astype(np.float64)
    # inverse rotation = rotate by conjugate (negate vector part)
    u = -u
    t = 2.0 * np.cross(u, v)
    return (v + w * t + np.cross(u, t)).astype(np.float32)


# --- formula tests (wheel-frame inputs) --------------------------------------
def test_slip_free_roll_is_zero(real_modules):
    """Wheel rolling straight at v with omega = v/r -> slip ratio and angle ~ 0."""
    v = 3.0
    omega = v / WHEEL_RADIUS
    mlv = np.tile([v, 0.0, 0.0], (1, 4, 1)).astype(np.float32)
    dof = np.full((1, 4), omega, dtype=np.float32)
    slip = real_modules.car.compute_tyre_slip(_wheel_state(mlv, dof), WHEEL_RADIUS, SLIP_EPS)
    ratio, angle = slip[0, :4], slip[0, 4:]
    assert torch.allclose(ratio, torch.zeros(4), atol=1e-5)
    assert torch.allclose(angle, torch.zeros(4), atol=1e-6)


def test_slip_ratio_positive_under_wheelspin(real_modules):
    """Wheel spinning faster than ground (throttle) -> positive slip ratio."""
    v = 2.0
    omega = 4.0 / WHEEL_RADIUS  # surface speed 4 > ground 2
    mlv = np.tile([v, 0.0, 0.0], (1, 4, 1)).astype(np.float32)
    dof = np.full((1, 4), omega, dtype=np.float32)
    slip = real_modules.car.compute_tyre_slip(_wheel_state(mlv, dof), WHEEL_RADIUS, SLIP_EPS)
    ratio = slip[0, :4]
    assert torch.all(ratio > 0.0)
    assert ratio[0].item() == pytest.approx((4.0 - 2.0) / 4.0, abs=1e-5)


def test_slip_ratio_negative_under_lockup(real_modules):
    """Locked wheel (omega=0) while moving -> slip ratio ~ -1."""
    v = 2.0
    mlv = np.tile([v, 0.0, 0.0], (1, 4, 1)).astype(np.float32)
    dof = np.zeros((1, 4), dtype=np.float32)
    slip = real_modules.car.compute_tyre_slip(_wheel_state(mlv, dof), WHEEL_RADIUS, SLIP_EPS)
    ratio = slip[0, :4]
    assert torch.allclose(ratio, torch.full((4,), -1.0), atol=1e-5)


@pytest.mark.parametrize("v_lat", [0.5, -0.8, 1.2])
def test_slip_angle_matches_lateral(real_modules, v_lat):
    """Slip angle equals atan2(v_lat, |v_fwd|) for wheel-frame inputs."""
    v_fwd = 3.0
    omega = v_fwd / WHEEL_RADIUS
    mlv = np.tile([v_fwd, v_lat, 0.0], (1, 4, 1)).astype(np.float32)
    dof = np.full((1, 4), omega, dtype=np.float32)
    slip = real_modules.car.compute_tyre_slip(_wheel_state(mlv, dof), WHEEL_RADIUS, SLIP_EPS)
    angle = slip[0, 4:]
    expected = math.atan2(v_lat, abs(v_fwd))
    assert torch.allclose(angle, torch.full((4,), expected), atol=1e-5)


# --- frame-sensitivity demonstration (the suspected bug) ---------------------
@pytest.mark.parametrize("yaw", [0.4, -0.7, 1.0])
def test_world_frame_velocity_corrupts_slip_angle(real_modules, yaw):
    """A wheel physically rolling straight (wheel-frame v=[v,0,0]) but observed in
    the WORLD frame yields slip_angle ~ yaw instead of ~ 0. This is exactly what
    happens if `frame_quat` is never applied before `compute_tyre_slip`.

    Confirms: (a) feeding world-frame velocity is wrong, and (b) rotating by the
    frame quaternion first recovers the correct ~0 slip angle.
    """
    v = 3.0
    omega = v / WHEEL_RADIUS
    quat = _quat_wxyz_yaw(yaw)
    # true wheel-frame velocity is purely forward; world frame is yaw-rotated
    world_vel = _rotate_local_to_world(np.array([v, 0.0, 0.0], np.float32), quat)

    # BUGGY path: compute slip directly on the world-frame velocity
    mlv_world = np.tile(world_vel, (1, 4, 1)).astype(np.float32)
    dof = np.full((1, 4), omega, dtype=np.float32)
    buggy = real_modules.car.compute_tyre_slip(
        _wheel_state(mlv_world, dof), WHEEL_RADIUS, SLIP_EPS
    )
    buggy_angle = buggy[0, 4:]
    assert torch.allclose(buggy_angle, torch.full((4,), float(yaw)), atol=1e-3), (
        "world-frame velocity should make slip_angle track yaw"
    )

    # CORRECT path: rotate world->wheel frame first, then compute
    wheel_vel = _rotate_world_to_local(world_vel, quat)
    mlv_wheel = np.tile(wheel_vel, (1, 4, 1)).astype(np.float32)
    fixed = real_modules.car.compute_tyre_slip(
        _wheel_state(mlv_wheel, dof), WHEEL_RADIUS, SLIP_EPS
    )
    assert torch.allclose(fixed[0, 4:], torch.zeros(4), atol=1e-3)


def _rotate_local_to_world(vec_local: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    u = np.array([x, y, z], dtype=np.float64)
    v = vec_local.astype(np.float64)
    t = 2.0 * np.cross(u, v)
    return (v + w * t + np.cross(u, t)).astype(np.float32)


# --- post-fix correctness: frame_quat rotation inside compute_tyre_slip --------
@pytest.mark.parametrize("yaw", [0.0, 0.4, -0.7, 1.0])
def test_frame_quat_rotation_recovers_zero_slip(real_modules, yaw):
    """With the fix, `compute_tyre_slip` rotates the WORLD-frame wheel velocity by
    `frame_quat` into the wheel frame. A wheel rolling straight (wheel-frame
    [v,0,0]) observed in the world frame, plus its frame quaternion, must yield
    slip_angle ~ 0 and slip_ratio ~ 0 for all yaws."""
    v = 3.0
    omega = v / WHEEL_RADIUS
    quat = _quat_wxyz_yaw(yaw)
    world_vel = _rotate_local_to_world(np.array([v, 0.0, 0.0], np.float32), quat)

    mlv = np.tile(world_vel, (1, 4, 1)).astype(np.float32)
    dof = np.full((1, 4), omega, dtype=np.float32)
    fquat = np.tile(quat, (1, 4, 1)).astype(np.float32)

    slip = real_modules.car.compute_tyre_slip(
        _wheel_state(mlv, dof, frame_quat=fquat), WHEEL_RADIUS, SLIP_EPS
    )
    ratio, angle = slip[0, :4], slip[0, 4:]
    assert torch.allclose(angle, torch.zeros(4), atol=1e-3), f"yaw={yaw}, angle={angle}"
    assert torch.allclose(ratio, torch.zeros(4), atol=1e-3), f"yaw={yaw}, ratio={ratio}"


def test_frame_quat_rotation_preserves_lateral_slip(real_modules):
    """A genuine lateral slide in the wheel frame must survive the rotation: with a
    non-zero yaw frame_quat and a wheel-frame velocity of [v, v_lat, 0] (expressed
    in world), the recovered slip_angle equals atan2(v_lat, v)."""
    v, v_lat, yaw = 3.0, 0.8, 0.6
    omega = v / WHEEL_RADIUS
    quat = _quat_wxyz_yaw(yaw)
    world_vel = _rotate_local_to_world(np.array([v, v_lat, 0.0], np.float32), quat)

    mlv = np.tile(world_vel, (1, 4, 1)).astype(np.float32)
    dof = np.full((1, 4), omega, dtype=np.float32)
    fquat = np.tile(quat, (1, 4, 1)).astype(np.float32)

    slip = real_modules.car.compute_tyre_slip(
        _wheel_state(mlv, dof, frame_quat=fquat), WHEEL_RADIUS, SLIP_EPS
    )
    expected = math.atan2(v_lat, abs(v))
    assert torch.allclose(slip[0, 4:], torch.full((4,), expected), atol=2e-3)

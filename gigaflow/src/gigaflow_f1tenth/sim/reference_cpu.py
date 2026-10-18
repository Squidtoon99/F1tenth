"""CPU reference helpers for focused correctness tests (not a production path)."""

from __future__ import annotations

import math

import numpy as np

from gigaflow_f1tenth.sim.contact import resolve_pair_contact_numpy
from gigaflow_f1tenth.sim.sensors import ray_obb_distance_numpy
from gigaflow_f1tenth.sim.vehicle import VehicleParams


def integrate_point_mass_bicycle_step(
    x: float,
    y: float,
    yaw: float,
    v: float,
    steer: float,
    accel: float,
    dt: float,
    wheelbase: float,
) -> tuple[float, float, float, float]:
    """Tiny kinematic reference used only to sanity-check pose integration signs."""
    yaw_rate = v * math.tan(steer) / max(wheelbase, 1.0e-6)
    yaw2 = yaw + yaw_rate * dt
    v2 = v + accel * dt
    x2 = x + v2 * math.cos(yaw2) * dt
    y2 = y + v2 * math.sin(yaw2) * dt
    return x2, y2, yaw2, v2


def overlapping_boxes_should_contact(
    separation: float,
    car_length: float,
    car_width: float,
) -> bool:
    params = VehicleParams()
    del params
    hit = resolve_pair_contact_numpy(
        0.0,
        0.0,
        0.0,
        separation,
        0.0,
        0.0,
        car_length,
        car_width,
    )
    return bool(hit["contact"])


def ray_hits_box_ahead() -> float:
    origin = np.array([0.0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    return ray_obb_distance_numpy(
        origin,
        direction,
        np.array([2.0, 0.0], dtype=np.float64),
        0.0,
        0.25,
        0.15,
    )

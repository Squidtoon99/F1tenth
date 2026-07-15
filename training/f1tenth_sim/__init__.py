"""Warp, PhysX-grounded vehicle simulator for F1tenth RL.

This package has no ROS dependencies and implements a batched vehicle model:

- ``params``     -- VehicleParams (seeded from the URDF + config).
- ``tire``       -- Pacejka Magic-Formula tire with a combined-slip friction ellipse.
- ``suspension`` -- quasi-static and spring-damper load transfer (dynamic Fz).
- ``drivetrain`` -- throttle/brake -> per-wheel torque (force envelope + VESC option).
- ``dynamics``   -- assembles tire forces and integrates the body + wheel-spin ODEs.
- ``sim_warp``   -- WarpVehicleSim state, stepping, reset, and readback.

The training environment uses the Warp structs and functions directly.
"""

import warp as wp

from .params import VehicleParams
from .sim_warp import WarpVehicleSim

wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN

__all__ = ["VehicleParams", "WarpVehicleSim"]

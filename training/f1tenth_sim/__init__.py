"""Reduced bicycle vehicle simulator for F1tenth RL.

This package has no ROS dependencies:

- ``params``   -- VehicleParams (URDF geometry + identified/prior scalars).
- ``dynamics`` -- gate-matched current channels and the axle bicycle plant.
- ``sim_warp`` -- WarpVehicleSim state, stepping, reset, and readback.

The training environment uses the Warp structs and functions directly.
"""

import warp as wp

from .params import VehicleParams
from .sim_warp import WarpVehicleSim

wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN

__all__ = ["VehicleParams", "WarpVehicleSim"]

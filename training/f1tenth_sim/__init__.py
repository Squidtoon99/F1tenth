"""Pure-Torch, PhysX-style vehicle simulator for F1tenth RL.

This package is intentionally free of Genesis and ROS dependencies so it can be
imported and unit-tested anywhere PyTorch is available (CPU, CUDA, or MPS). It
implements a fully-vectorized (``num_envs`` batched) vehicle model:

- ``params``     -- VehicleParams (seeded from the URDF + config).
- ``tire``       -- Pacejka Magic-Formula tire with a combined-slip friction ellipse.
- ``suspension`` -- quasi-static and spring-damper load transfer (dynamic Fz).
- ``drivetrain`` -- throttle/brake -> per-wheel torque (force envelope + VESC option).
- ``dynamics``   -- assembles tire forces and integrates the body + wheel-spin ODEs.
- ``sim``        -- TorchVehicleSim: owns the state tensors, ``step``/``reset``/``read_state``.

The public surface used by the training env is ``TorchVehicleSim`` and
``VehicleParams``.
"""

from .params import VehicleParams
from .sim import TorchVehicleSim

__all__ = ["VehicleParams", "TorchVehicleSim"]

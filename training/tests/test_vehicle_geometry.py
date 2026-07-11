"""Parity guards tying the deploy/ROS vehicle geometry to the URDF source of truth.

TorchSim reads chassis geometry from ``training/F110.export.urdf`` via
``VehicleParams.from_urdf`` (used by ``from_config``). These tests pin the parsed
values and assert the per-car deploy overlay (``deploy/cars/car01/params.yaml``)
and the training body envelope stay consistent with it, so the sim and the car
never silently drift apart.
"""

from __future__ import annotations

import os

import yaml

from f1tenth_sim import VehicleParams

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_URDF = os.path.join(_REPO_ROOT, "training", "F110.export.urdf")
_CAR01 = os.path.join(_REPO_ROOT, "deploy", "cars", "car01", "params.yaml")


def _urdf_params() -> VehicleParams:
    return VehicleParams.from_urdf(_URDF)


def test_urdf_geometry_matches_nominal():
    """The URDF still parses to the standard F1TENTH chassis geometry."""
    p = _urdf_params()
    assert abs(p.wheelbase - 0.325) < 1e-3
    assert abs(p.track_width - 0.20) < 1e-3
    assert abs(p.wheel_radius - 0.05) < 1e-3
    # CoG split sums to the wheelbase.
    assert abs((p.lf + p.lr) - p.wheelbase) < 1e-6
    assert abs(p.lf - 0.1584) < 2e-3
    assert abs(p.lr - 0.1666) < 2e-3


def test_car01_overlay_geometry_matches_urdf():
    """The per-car observation overlay carries the URDF-derived geometry so on-car
    load transfer matches TorchSim."""
    p = _urdf_params()
    with open(_CAR01) as f:
        cfg = yaml.safe_load(f)

    obs = cfg["vehicle_obs"]["ros__parameters"]
    assert abs(float(obs["wheel_radius_m"]) - p.wheel_radius) < 1e-3
    assert abs(float(obs["track_width_m"]) - p.track_width) < 1e-3
    assert abs(float(obs["lf_m"]) - p.lf) < 2e-3
    assert abs(float(obs["lr_m"]) - p.lr) < 2e-3


def test_car01_odom_wheelbase_matches_urdf():
    """The VESC odometry wheelbase overlay matches the URDF wheelbase (overriding
    the stale vendored vesc_ackermann default)."""
    p = _urdf_params()
    with open(_CAR01) as f:
        cfg = yaml.safe_load(f)
    wb = float(cfg["vesc_to_odom_node"]["ros__parameters"]["wheelbase"])
    assert abs(wb - p.wheelbase) < 1e-3


def test_training_body_envelope_is_slash_spec():
    """The training body envelope matches the provisional Traxxas Slash 4x4 spec."""
    from config import DEFAULT_CONFIG

    env = DEFAULT_CONFIG["env"]
    assert abs(float(env["car_length"]) - 0.568) < 1e-6
    assert abs(float(env["car_width"]) - 0.296) < 1e-6

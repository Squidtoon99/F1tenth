"""Parity guards tying calibrated training, URDF, and deploy vehicle geometry.

The URDF supplies inertial priors, while explicit calibrated config values define
known geometry. These tests keep the backend asset, Warp sim, and car overlay from
silently drifting apart.
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
_VEHICLE_DEFAULTS = os.path.join(
    _REPO_ROOT, "src", "racing_rl", "f1tenth_rl_vehicle", "config", "vehicle.yaml"
)
_VESC_DEFAULTS = os.path.join(
    _REPO_ROOT, "src", "vehicle", "f1tenth_stack", "config", "vesc.yaml"
)


def _urdf_params() -> VehicleParams:
    return VehicleParams.from_urdf(_URDF)


def test_urdf_geometry_matches_nominal():
    """The URDF still parses to the standard F1TENTH chassis geometry."""
    p = _urdf_params()
    assert abs(p.wheelbase - 0.325) < 1e-3
    assert abs(p.track_width - 0.253) < 1e-3
    assert abs(p.wheel_radius - 0.053) < 1e-3
    # CoG split sums to the wheelbase.
    assert abs((p.lf + p.lr) - p.wheelbase) < 1e-6
    assert abs(p.lf - 0.1584) < 2e-3
    assert abs(p.lr - 0.1666) < 2e-3


def test_config_geometry_overrides_urdf_and_updates_wheel_offsets():
    p = VehicleParams.from_config(
        {"wheelbase": 0.325, "track_width": 0.253, "wheel_radius": 0.053}
    )
    assert abs(p.track_width - 0.253) < 1e-6
    assert abs(p.wheel_xy[0][0] + p.lr) < 1e-6
    assert abs(p.wheel_xy[2][0] - p.lf) < 1e-6
    assert abs(p.wheel_xy[0][1] - 0.1265) < 1e-6
    assert abs(p.wheel_xy[1][1] + 0.1265) < 1e-6


def test_training_config_uses_calibrated_geometry_and_roll_split():
    from config import DEFAULT_CONFIG

    env = DEFAULT_CONFIG["env"]
    params = VehicleParams.from_config(env)
    assert abs(params.wheelbase - 0.325) < 1e-6
    assert abs(params.track_width - 0.253) < 1e-6
    assert abs(params.wheel_radius - 0.053) < 1e-6
    assert abs(params.roll_stiffness_front - 0.47) < 1e-6
    assert not hasattr(params, "susp_damping")
    assert not hasattr(params, "anti_roll")
    assert not hasattr(params, "max_speed")


def test_deploy_defaults_match_training():
    from config import DEFAULT_CONFIG

    p = _urdf_params()
    with open(_CAR01) as f:
        car = yaml.safe_load(f)
    with open(_VEHICLE_DEFAULTS) as f:
        vehicle = yaml.safe_load(f)["vehicle_obs"]["ros__parameters"]
    with open(_VESC_DEFAULTS) as f:
        vesc = yaml.safe_load(f)

    obs = car["vehicle_obs"]["ros__parameters"]
    assert abs(float(obs["wheel_radius_m"]) - p.wheel_radius) < 1e-3
    assert abs(float(obs["track_width_m"]) - p.track_width) < 1e-3
    assert abs(float(obs["lf_m"]) - p.lf) < 2e-3
    assert abs(float(obs["lr_m"]) - p.lr) < 2e-3
    assert abs(float(obs["roll_stiffness_front"]) - 0.47) < 1e-6
    wb = float(car["vesc_to_odom_node"]["ros__parameters"]["wheelbase"])
    assert abs(wb - p.wheelbase) < 1e-3
    steer_scale = car["joy_teleop"]["ros__parameters"]["human_control"][
        "axis_mappings"
    ]["drive-steering_angle"]["scale"]
    assert abs(float(steer_scale) - float(DEFAULT_CONFIG["env"]["delta_max"])) < 1e-6
    assert abs(float(vehicle["wheel_radius_m"]) - 0.053) < 1e-6
    assert abs(float(vehicle["track_width_m"]) - 0.253) < 1e-6
    assert abs(float(vehicle["roll_stiffness_front"]) - 0.47) < 1e-6
    assert abs(float(vesc["/**"]["ros__parameters"]["speed_to_erpm_gain"]) - 4300.0) < 1e-6
    assert abs(
        float(vesc["vesc_to_odom_node"]["ros__parameters"]["wheelbase"]) - 0.325
    ) < 1e-6


def test_training_body_envelope_is_slash_spec():
    """The training body envelope matches the provisional Traxxas Slash 4x4 spec."""
    from config import DEFAULT_CONFIG

    env = DEFAULT_CONFIG["env"]
    assert abs(float(env["car_length"]) - 0.568) < 1e-6
    assert abs(float(env["car_width"]) - 0.296) < 1e-6

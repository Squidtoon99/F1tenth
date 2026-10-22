"""Frozen IRL embodied identity for Gigaflow deploy / offline fine-tune.

Inference uses this vector as-is. Training may still sample `vmax_scale` and
the other estimable terms; exported artifacts and eval/deploy do not.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gigaflow_f1tenth.rewards import (
    CONDITION_FIELD_NAMES,
    PrivateStyle,
    deployment_style,
    normalize_condition_vector,
)
from gigaflow_f1tenth.sim.vehicle import F_BRAKE_MAX, F_DRIVE_MAX, VehicleParams

_PARAMS = VehicleParams()

# Live Gigaflow plant on experiments/self-play-rl: IRL mass / μ / 80-40 envelope
# plus the frozen SIM_READY_V2 bag-fit forces (vehicle.py: "do not retune").
# Sep 6 champion 26.5 N / 23.1 N is recorded but not the live plant.
IRL_MASS_KG = float(_PARAMS.mass)
IRL_TIRE_MU = float(_PARAMS.tire_mu)
IRL_F_DRIVE_N = float(F_DRIVE_MAX)
IRL_F_BRAKE_N = float(F_BRAKE_MAX)
IRL_I_DRIVE_MAX_A = float(_PARAMS.i_drive_max_a)
IRL_I_BRAKE_MAX_A = float(_PARAMS.i_brake_max_a)
IRL_I_SLEW_A_PER_S = float(_PARAMS.i_slew_a_per_s)
IRL_DRIVE_GAIN_MPS2 = IRL_F_DRIVE_N / IRL_MASS_KG
IRL_BRAKE_GAIN_MPS2 = IRL_F_BRAKE_N / IRL_MASS_KG
IRL_EFFORT_SLEW_PER_S = IRL_I_SLEW_A_PER_S / IRL_I_DRIVE_MAX_A
IRL_MAX_STEER_RAD = float(_PARAMS.max_steer)
IRL_STEERING_DELTA_MAX_RAD = float(_PARAMS.steering_delta_max)

IRL_DYNAMICS = {
    "drive_scale": 1.0,
    "steer_scale": 1.0,
    "accel_scale": 1.0,
    "vmax_scale": 1.0,
    "mass_kg": IRL_MASS_KG,
}

IRL_SOURCE = "sep6-identity-mass-mu-envelope+sim-ready-v2-forces"


def baked_raw_style(base_name: str = "centered_high_collision") -> PrivateStyle:
    raw = deployment_style(base_name).as_dict()
    raw.update(IRL_DYNAMICS)
    return PrivateStyle(**{name: float(raw[name]) for name in CONDITION_FIELD_NAMES})


def baked_normalized_condition(
    base_name: str = "centered_high_collision",
) -> list[float]:
    vec = normalize_condition_vector(baked_raw_style(base_name).raw_vector())
    return np.asarray(vec, dtype=np.float32).tolist()


def irl_identity_payload() -> dict[str, float | str]:
    return {
        "source": IRL_SOURCE,
        "mass_kg": IRL_MASS_KG,
        "tire_mu": IRL_TIRE_MU,
        "f_drive_max": IRL_F_DRIVE_N,
        "f_brake_max": IRL_F_BRAKE_N,
        "i_drive_max_a": IRL_I_DRIVE_MAX_A,
        "i_brake_max_a": IRL_I_BRAKE_MAX_A,
        "i_slew_a_per_s": IRL_I_SLEW_A_PER_S,
        "drive_gain_mps2": IRL_DRIVE_GAIN_MPS2,
        "brake_gain_mps2": IRL_BRAKE_GAIN_MPS2,
        "effort_slew_per_s": IRL_EFFORT_SLEW_PER_S,
        "max_steer_rad": IRL_MAX_STEER_RAD,
        "steering_delta_max_rad": IRL_STEERING_DELTA_MAX_RAD,
        **IRL_DYNAMICS,
    }


def baked_condition_tensor(device: str | None = None):
    import torch

    vec = torch.as_tensor(baked_normalized_condition(), dtype=torch.float32)
    return vec if device is None else vec.to(device)


def attach_baked_condition(actor, condition=None) -> None:
    """Pin ``c_irl`` on the actor so ``actor(sensor_obs)`` needs no condition."""
    vec = baked_normalized_condition() if condition is None else condition
    actor.set_baked_condition(vec)


def fold_gru_layer0(actor, condition=None):
    """Structural GRU Layer-0 fold: ``W_ih[:, :k]``, ``b_ih + W_ih[:, k:] @ e_c``."""
    return actor.fold_gru_layer0(condition)


def folded_gru_module(actor, condition=None):
    """GRU with Layer-0 input size ``cnn+proprio`` and folded bias."""
    import torch
    import torch.nn as nn

    e_c, w_folded, b_folded = actor.fold_gru_layer0(condition)
    split = actor.gru_condition_slice()
    hidden_size = int(actor.shapes().gru_hidden_dim)
    gru = nn.GRU(input_size=split, hidden_size=hidden_size, batch_first=True)
    with torch.no_grad():
        gru.weight_ih_l0.copy_(w_folded)
        gru.bias_ih_l0.copy_(b_folded)
        gru.weight_hh_l0.copy_(actor.gru.weight_hh_l0)
        gru.bias_hh_l0.copy_(actor.gru.bias_hh_l0)
    gru.to(device=actor.gru.weight_ih_l0.device)
    return gru, e_c


def folded_gru_step(actor, sensor_obs, hidden, condition=None):
    """One GRU step using folded Layer-0 weights; bitwise match at ``c_irl``."""
    import torch

    gru, e_c = folded_gru_module(actor, condition)
    lidar, proprio = actor._split_obs(sensor_obs)
    features = actor.encoder(lidar)
    trunk = torch.cat([features, proprio], dim=-1)
    out, h_n = gru(trunk.unsqueeze(1), hidden.unsqueeze(0).contiguous())
    return h_n.squeeze(0), e_c, gru


def bake_deployment_condition(
    payload: Mapping[str, Any],
    *,
    base_name: str | None = None,
) -> dict[str, Any]:
    """Rewrite a format-1 artifact so inference cannot sample embodied terms."""
    out = dict(payload)
    if "critic" in out or "central_critic" in out:
        raise ValueError("deployable artifacts must not contain critic weights")
    style_name = str(
        base_name
        or out.get("deployment_style")
        or "centered_high_collision"
    )
    schema = out.get("condition_schema")
    raw = {}
    if isinstance(schema, Mapping):
        conservative = schema.get("conservative_raw")
        if isinstance(conservative, Mapping):
            raw = {str(k): float(v) for k, v in conservative.items()}
    if not raw:
        raw = deployment_style(style_name).as_dict()
    raw.update(IRL_DYNAMICS)
    style = PrivateStyle(**{name: float(raw[name]) for name in CONDITION_FIELD_NAMES})
    out["format_version"] = 1
    out["deployment_style"] = style_name
    out["deployment_condition_normalized"] = np.asarray(
        normalize_condition_vector(style.raw_vector()), dtype=np.float32
    ).tolist()
    out["irl_baked"] = True
    out["irl_identity"] = irl_identity_payload()
    out.pop("adaptation", None)
    if isinstance(schema, Mapping):
        schema_out = dict(schema)
        schema_out["conservative_raw"] = style.as_dict()
        schema_out["irl_baked"] = True
        out["condition_schema"] = schema_out
    return out

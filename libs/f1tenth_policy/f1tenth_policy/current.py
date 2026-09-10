"""Shared applied-current observation helpers for training and deploy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Training/sim physical scale for normalized effort=1 (command envelope).
# Deploy must match these for physical actuation parity; observation v3 still
# divides by whatever directional limits are configured on the node.
TRAINING_I_DRIVE_MAX_A = 80.0
TRAINING_I_BRAKE_MAX_A = 40.0
TRAINING_I_SLEW_A_PER_S = 200.0


def applied_current_fraction(
    signed_applied_current_a: float,
    i_drive_max_a: float,
    i_brake_max_a: float,
) -> float:
    """Map signed applied amperes to a directional limit fraction.

    Drive uses ``i_drive_max_a`` and brake uses ``i_brake_max_a``. Sign is
    preserved. Invalid or non-positive limits fail closed with ``ValueError``.
    Non-finite amperes map to ``0.0``.
    """
    signed = float(signed_applied_current_a)
    if not math.isfinite(signed):
        return 0.0
    if signed >= 0.0:
        limit = float(i_drive_max_a)
    else:
        limit = float(i_brake_max_a)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            "configured directional current limit must be finite and > 0 "
            f"(signed_a={signed!r}, i_drive_max_a={i_drive_max_a!r}, "
            f"i_brake_max_a={i_brake_max_a!r})"
        )
    return signed / limit


def assert_artifact_current_limits_match(
    payload: Mapping[str, Any],
    i_drive_max_a: float,
    i_brake_max_a: float,
    i_slew_a_per_s: float,
    *,
    atol: float = 1e-6,
) -> None:
    """Reject deploy limits that diverge from the training scale."""
    deploy = {
        "i_drive_max_a": float(i_drive_max_a),
        "i_brake_max_a": float(i_brake_max_a),
        "i_slew_a_per_s": float(i_slew_a_per_s),
    }
    for name, deploy_value in deploy.items():
        artifact_value = payload.get(name)
        if artifact_value is None:
            raise ValueError(f"Sensor policy artifact is missing {name}.")
        artifact_value = float(artifact_value)
        if not math.isfinite(artifact_value) or artifact_value <= 0.0:
            raise ValueError(f"Sensor policy artifact {name}={artifact_value!r} is invalid.")
        if not math.isfinite(deploy_value) or deploy_value <= 0.0:
            raise ValueError(f"Deploy {name}={deploy_value!r} is invalid.")
        if abs(artifact_value - deploy_value) > atol:
            raise ValueError(
                f"Deploy {name}={deploy_value!r} does not match artifact training "
                f"scale {name}={artifact_value!r}."
            )

"""Shared applied-current observation helpers for training and deploy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Training/sim physical scale for normalized effort=1 (command envelope).
# Deploy must match these for physical actuation parity; observation v3 still
# divides by whatever directional limits are configured on the node.
TRAINING_I_DRIVE_MAX_A = 80.0
TRAINING_I_BRAKE_MAX_A = 20.0
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
    *,
    atol: float = 1e-6,
) -> None:
    """Reject deploy limits that silently diverge from the training scale."""
    art_drive = payload.get("i_drive_max_a")
    art_brake = payload.get("i_brake_max_a")
    if art_drive is None or art_brake is None:
        raise ValueError(
            "Sensor policy artifact is missing i_drive_max_a/i_brake_max_a; "
            "cannot claim physical current-scale parity with deploy limits."
        )
    drive = float(art_drive)
    brake = float(art_brake)
    deploy_drive = float(i_drive_max_a)
    deploy_brake = float(i_brake_max_a)
    if not math.isfinite(drive) or drive <= 0.0:
        raise ValueError(f"Sensor policy artifact i_drive_max_a={art_drive!r} is invalid.")
    if not math.isfinite(brake) or brake <= 0.0:
        raise ValueError(f"Sensor policy artifact i_brake_max_a={art_brake!r} is invalid.")
    if not math.isfinite(deploy_drive) or deploy_drive <= 0.0:
        raise ValueError(f"Deploy i_drive_max_a={i_drive_max_a!r} is invalid.")
    if not math.isfinite(deploy_brake) or deploy_brake <= 0.0:
        raise ValueError(f"Deploy i_brake_max_a={i_brake_max_a!r} is invalid.")
    if abs(drive - deploy_drive) > atol:
        raise ValueError(
            f"Deploy i_drive_max_a={deploy_drive!r} does not match "
            f"artifact training scale i_drive_max_a={drive!r}; refusing to "
            "claim physical current/force parity."
        )
    if abs(brake - deploy_brake) > atol:
        raise ValueError(
            f"Deploy i_brake_max_a={deploy_brake!r} does not match "
            f"artifact training scale i_brake_max_a={brake!r}; refusing to "
            "claim physical current/force parity."
        )

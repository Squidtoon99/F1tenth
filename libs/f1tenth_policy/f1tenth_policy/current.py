"""Shared applied-current observation helpers for training and deploy."""

from __future__ import annotations

import math


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

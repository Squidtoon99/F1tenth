"""Action layout — single source of truth.

Scaffold placeholder. The policy emits a normalized action; the drive layer maps it
to a concrete AckermannDriveStamped command.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    """Normalized action layout: (throttle, steer), each in [-1, 1]."""

    names: tuple[str, ...] = ("throttle", "steer")
    low: float = -1.0
    high: float = 1.0

    @property
    def dim(self) -> int:
        return len(self.names)


ACTION = ActionSpec()

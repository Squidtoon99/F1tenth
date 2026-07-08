"""Action layout — the shared action-space format.

The policy emits a normalized action ``(throttle, steer)``, each in [-1, 1]. The
drive layer (``src/control/f1tenth_control``) maps it to a concrete
``AckermannDriveStamped``. The mapping constants below mirror the deployed
``interfaces.py`` so both sides agree on the normalized-action -> command scale; a
parity test guards against drift.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Action -> drive mapping constants (mirror of deploy interfaces.py) --------
MAX_SPEED = 15.0
MAX_STEER = 0.44  # radians at |steer| == 1.0
CLIP_ACTIONS = 1.0
ACT_LIMIT = 1.0
CONTROL_HZ = 10.0


@dataclass(frozen=True)
class ActionSpec:
    """Normalized action layout: (throttle, steer), each in [low, high]."""

    names: tuple[str, ...] = ("throttle", "steer")
    low: float = -1.0
    high: float = 1.0

    @property
    def dim(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        return self.names.index(name)


ACTION = ActionSpec()

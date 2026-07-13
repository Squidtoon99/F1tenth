"""Action layout — the shared action-space format.

The policy emits a normalized action ``(throttle, steer)``, each in [-1, 1]:

* ``throttle > 0`` — drive effort (sim force / VESC motor current)
* ``throttle = 0`` — coast
* ``throttle < 0`` — brake effort (sim brake force / VESC brake current)
* ``steer`` — steering, mapped to ``steering_angle = steer * MAX_STEER``

The drive layer (``src/control/f1tenth_control``) maps throttle to
``AckermannDrive.acceleration`` and then to motor/brake current. Mapping
constants below mirror the deployed ``interfaces.py``; a parity test guards
against drift.
"""

from __future__ import annotations

from dataclasses import dataclass

# Longitudinal action is force/current effort, not a speed setpoint. MAX_SPEED
# remains as an observation / episode-horizon reference only.
MAX_SPEED = 15.0
MAX_STEER = 0.33  # radians at |steer| == 1.0 (real servo hard-clamp; matches training)
CLIP_ACTIONS = 1.0
ACT_LIMIT = 1.0
# Synchronized training + deploy control rate (policy decision period).
CONTROL_HZ = 20.0
# Checkpoint format for current/force longitudinal semantics. Speed-trained
# checkpoints (policy_format_version < 2) must be rejected at deploy.
POLICY_FORMAT_VERSION = 2
LONGITUDINAL_MODE = "force"


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

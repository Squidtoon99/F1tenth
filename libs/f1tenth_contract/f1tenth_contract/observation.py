"""Observation layout — the shared observation-space format.

This module documents the *format* of the flat observation vector: its dimensions
and the (start, stop) slice of every field. It is the single reference that both
sides target:

- training: ``training/f1tenth_env/observations.py`` builds this layout.
- on-car inference: ``src/racing_rl/f1tenth_rl_agent`` (``interfaces.py`` /
  ``obs_core.py``) builds and consumes it.
- the C++ mirror in ``src/common/f1tenth_common`` is checked against these values.

These are format constants only; the field-building math lives with each consumer.
A parity test in ``f1tenth_rl_agent`` asserts that the deployed ``interfaces.py``
matches the values here so the contract cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Dimensions ---------------------------------------------------------------
NUM_OBS_BASE = 380
OPPONENT_OBS_DIM = 7
NUM_OBS_1V1 = NUM_OBS_BASE + OPPONENT_OBS_DIM  # 387
NUM_ACTIONS = 2
NUM_TYRE_SLIP = 8  # [slip_ratio x4, slip_angle x4]

# --- Field slices (start, stop) within the base 380-dim vector ----------------
OBS_LIN_VEL = (0, 2)
OBS_ANG_VEL = (2, 3)
OBS_LIN_ACC = (3, 5)
OBS_LAST_ACTION = (5, 7)
OBS_TRACK_PROGRESS = (7, 9)
OBS_CENTERLINE_ANGLE = (9, 10)
OBS_CENTERLINE_DISTANCE = (10, 11)
OBS_CONTACT_FLAG = (11, 12)
OBS_FUTURE_POINTS = (12, 372)
OBS_TYRE_SLIP = (372, 380)
# Opponent-relative block, appended only when opponent observations are enabled.
OBS_OPPONENT = (380, 387)

# Ordered (name, start, stop) table for the base observation. Kept in field order
# and contiguous from 0 to NUM_OBS_BASE.
OBS_FIELDS_BASE: tuple[tuple[str, int, int], ...] = (
    ("lin_vel", *OBS_LIN_VEL),
    ("ang_vel", *OBS_ANG_VEL),
    ("lin_acc", *OBS_LIN_ACC),
    ("last_action", *OBS_LAST_ACTION),
    ("track_progress", *OBS_TRACK_PROGRESS),
    ("centerline_angle", *OBS_CENTERLINE_ANGLE),
    ("centerline_distance", *OBS_CENTERLINE_DISTANCE),
    ("contact_flag", *OBS_CONTACT_FLAG),
    ("future_points", *OBS_FUTURE_POINTS),
    ("tyre_slip", *OBS_TYRE_SLIP),
)

# The opponent block, appended for the 1v1 (387-dim) observation.
OBS_FIELDS_OPPONENT: tuple[tuple[str, int, int], ...] = (
    ("opponent", *OBS_OPPONENT),
)


def expected_num_obs(enable_opponent_obs: bool) -> int:
    """Policy observation dimension (380 solo, 387 with the opponent block)."""
    return NUM_OBS_1V1 if enable_opponent_obs else NUM_OBS_BASE


@dataclass(frozen=True)
class ObservationSpec:
    """Describes the flat observation vector via ordered (name, start, stop) fields."""

    fields: tuple[tuple[str, int, int], ...]

    @property
    def dim(self) -> int:
        return self.fields[-1][2] if self.fields else 0

    def index_of(self, name: str) -> int:
        """Return the start index of a named field in the flat vector."""
        for field_name, start, _stop in self.fields:
            if field_name == name:
                return start
        raise KeyError(name)

    def slice_of(self, name: str) -> tuple[int, int]:
        """Return the (start, stop) slice of a named field."""
        for field_name, start, stop in self.fields:
            if field_name == name:
                return (start, stop)
        raise KeyError(name)


# Canonical instances imported elsewhere.
OBSERVATION = ObservationSpec(fields=OBS_FIELDS_BASE)
OBSERVATION_1V1 = ObservationSpec(fields=OBS_FIELDS_BASE + OBS_FIELDS_OPPONENT)

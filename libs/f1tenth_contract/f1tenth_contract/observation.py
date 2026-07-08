"""Observation layout — single source of truth.

Scaffold placeholder. Define the observation fields, their order, and dimensions
here. Both training and the on-car inference node import this, so changing the
observation space is a one-file change (no rebuild during training thanks to the
editable install).

Keep this in sync with the C++ mirror in
``src/common/f1tenth_common/include/f1tenth_common/observation_layout.hpp`` — the
parity test guards it.

Migration note: consolidate the observation math currently duplicated in
``F1tenth-Genesis/f1tenth_env/observations.py`` and
``ros2_deploy/.../obs_core.py`` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationSpec:
    """Describes the flat observation vector.

    Extend ``fields`` with (name, size) entries; ``dim`` is their sum.
    """

    # TODO: replace with the real layout, e.g.
    #   ("frenet_progress", 1), ("lateral_error", 1), ("lidar", 108), ...
    fields: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def dim(self) -> int:
        return sum(size for _name, size in self.fields)

    def index_of(self, name: str) -> int:
        """Return the start index of a named field in the flat vector."""
        offset = 0
        for field_name, size in self.fields:
            if field_name == name:
                return offset
            offset += size
        raise KeyError(name)


# The canonical instance imported elsewhere.
OBSERVATION = ObservationSpec()

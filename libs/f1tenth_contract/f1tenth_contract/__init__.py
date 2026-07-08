"""f1tenth_contract: the single source of truth for the observation/action layout.

Import this from both the training code (editable install) and the on-car RL
inference node (colcon --symlink-install). The C++ mirror lives in
``src/common/f1tenth_common`` and is checked against this package by a parity test.
"""

from f1tenth_contract.action import ActionSpec
from f1tenth_contract.observation import ObservationSpec

__all__ = ["ObservationSpec", "ActionSpec"]
__version__ = "0.0.0"

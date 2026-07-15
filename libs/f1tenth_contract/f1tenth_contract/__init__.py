"""f1tenth_contract: the shared observation/action format.

Import this from both the training code (editable install) and the on-car RL
inference stack (colcon symlink-install). It documents the observation/action
*format* (dimensions and field slices); the field-building math lives with each
consumer. The C++ mirror in ``src/common/f1tenth_common`` and the deployed
``interfaces.py`` are checked against these values by parity tests.
"""

from f1tenth_contract.action import (
    ACT_LIMIT,
    ACTION,
    CLIP_ACTIONS,
    CONTROL_HZ,
    MAX_SPEED,
    MAX_STEER,
    ActionSpec,
)
from f1tenth_contract.observation import (
    NUM_ACTIONS,
    NUM_OBS,
    NUM_OBS_1V1,
    NUM_OBS_BASE,
    NUM_TYRE_LOAD,
    NUM_TYRE_SLIP,
    OBS_ANG_VEL,
    OBS_CENTERLINE_ANGLE,
    OBS_CENTERLINE_DISTANCE,
    OBS_CONTACT_FLAG,
    OBS_FUTURE_POINTS,
    OBS_LAST_ACTION,
    OBS_LIN_ACC,
    OBS_LIN_VEL,
    OBS_OPPONENT,
    OBS_TRACK_PROGRESS,
    OBS_TYRE_LOAD,
    OBS_TYRE_SLIP,
    OBSERVATION,
    OBSERVATION_1V1,
    OPPONENT_OBS_DIM,
    ObservationSpec,
    expected_num_obs,
)
from f1tenth_contract.policy import (
    OBS_PREPROCESSING_VERSION,
    POLICY_FORMAT_VERSION,
    validate_policy_artifact,
)

__all__ = [
    "ObservationSpec",
    "ActionSpec",
    "ACTION",
    "OBSERVATION",
    "OBSERVATION_1V1",
    "NUM_OBS",
    "NUM_OBS_BASE",
    "NUM_OBS_1V1",
    "OPPONENT_OBS_DIM",
    "NUM_ACTIONS",
    "NUM_TYRE_SLIP",
    "NUM_TYRE_LOAD",
    "expected_num_obs",
    "OBS_LIN_VEL",
    "OBS_ANG_VEL",
    "OBS_LIN_ACC",
    "OBS_LAST_ACTION",
    "OBS_TRACK_PROGRESS",
    "OBS_CENTERLINE_ANGLE",
    "OBS_CENTERLINE_DISTANCE",
    "OBS_CONTACT_FLAG",
    "OBS_FUTURE_POINTS",
    "OBS_TYRE_SLIP",
    "OBS_TYRE_LOAD",
    "OBS_OPPONENT",
    "MAX_SPEED",
    "MAX_STEER",
    "CLIP_ACTIONS",
    "ACT_LIMIT",
    "CONTROL_HZ",
    "POLICY_FORMAT_VERSION",
    "OBS_PREPROCESSING_VERSION",
    "validate_policy_artifact",
]
__version__ = "0.1.0"

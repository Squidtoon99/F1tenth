"""Shared Lee sensor-policy package for training and deploy."""

from f1tenth_policy.actor import (
    LidarCNNEncoder,
    SquashedGaussianLidarGRUActor,
    actor_architecture_from_module,
    actor_from_architecture,
    architectures_match,
    make_actor,
    normalize_actor_architecture,
)
from f1tenth_policy.artifact import (
    build_sensor_artifact_payload,
    load_sensor_artifact,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.current import (
    TRAINING_I_BRAKE_MAX_A,
    TRAINING_I_DRIVE_MAX_A,
    TRAINING_I_SLEW_A_PER_S,
    applied_current_fraction,
)
from f1tenth_policy.layout import (
    ACTOR_ARCHITECTURE_NAME,
    ACTOR_LAYOUT_VERSION,
    ACTOR_OBS_DIM,
    ARTIFACT_SCOPE_SIM_TRAINING,
    CONTROL_HZ,
    CRITIC_OBS_DIM,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    NUM_ACTIONS,
    OBS_PREPROCESSING_VERSION,
    PROPRIO_DIM,
    SENSOR_POLICY_FORMAT_VERSION,
    STEERING_ACTION_MODE,
    STEERING_DELTA_MAX_RAD,
)
from f1tenth_policy.normalizer import ObsNormalizer, load_obs_normalizer
from f1tenth_policy.steering import (
    integrate_steering_delta,
    steer_history_angles_and_deltas,
)

__all__ = [
    "ACTOR_ARCHITECTURE_NAME",
    "ACTOR_LAYOUT_VERSION",
    "ACTOR_OBS_DIM",
    "ARTIFACT_SCOPE_SIM_TRAINING",
    "CONTROL_HZ",
    "CRITIC_OBS_DIM",
    "GRU_HIDDEN_DIM",
    "LIDAR_DIM",
    "LidarCNNEncoder",
    "NUM_ACTIONS",
    "OBS_PREPROCESSING_VERSION",
    "ObsNormalizer",
    "PROPRIO_DIM",
    "SENSOR_POLICY_FORMAT_VERSION",
    "STEERING_ACTION_MODE",
    "STEERING_DELTA_MAX_RAD",
    "SquashedGaussianLidarGRUActor",
    "actor_architecture_from_module",
    "actor_from_architecture",
    "TRAINING_I_BRAKE_MAX_A",
    "TRAINING_I_DRIVE_MAX_A",
    "TRAINING_I_SLEW_A_PER_S",
    "applied_current_fraction",
    "architectures_match",
    "build_sensor_artifact_payload",
    "integrate_steering_delta",
    "load_obs_normalizer",
    "load_sensor_artifact",
    "make_actor",
    "normalize_actor_architecture",
    "steer_history_angles_and_deltas",
    "validate_sensor_policy_artifact",
]

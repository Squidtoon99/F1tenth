from __future__ import annotations

import math

NATIVE_NUM_BEAMS = 1081
LIDAR_DIM = NATIVE_NUM_BEAMS
LIDAR_FOV_DEG = 270.0
LIDAR_ANGLE_MIN = math.radians(-0.5 * LIDAR_FOV_DEG)
LIDAR_ANGLE_INCREMENT = math.radians(LIDAR_FOV_DEG / (LIDAR_DIM - 1))
LIDAR_RANGE_MIN = 0.06
LIDAR_RANGE_MAX = 30.0

IMU_DIM = 6
PROPRIO_DIM = 16
ACTOR_OBS_DIM = 1097
NUM_ACTIONS = 2
CRITIC_OBS_DIM = 392
DEFAULT_ACTION_HORIZON = 1
ACTION_CHUNK_POLICY_FORMAT_VERSION = 5

LIDAR_START = 0
IMU_START = LIDAR_DIM
VESC_SPEED = IMU_START + IMU_DIM
VESC_CURRENT = VESC_SPEED + 1
THROTTLE_CURRENT = VESC_CURRENT + 1
THROTTLE_PRED = THROTTLE_CURRENT + 1
STEER_T = THROTTLE_PRED + 1
STEER_T1 = STEER_T + 1
STEER_T2 = STEER_T1 + 1
STEER_DELTA0 = STEER_T2 + 1
STEER_DELTA1 = STEER_DELTA0 + 1
STEER_DELTA2 = STEER_DELTA1 + 1

assert ACTOR_OBS_DIM == STEER_DELTA2 + 1
assert PROPRIO_DIM == ACTOR_OBS_DIM - LIDAR_DIM

GRAVITY_MS2 = 9.81

ACTOR_LAYOUT_VERSION = 2
SENSOR_POLICY_FORMAT_VERSION = 4
OBS_PREPROCESSING_VERSION = 5
ARTIFACT_SCOPE_SIM_TRAINING = "simulation_training_only"
CONTROL_HZ = 20.0
HISTORY_REF_HZ = 10.0
HISTORY_STRIDE = max(1, int(round(CONTROL_HZ / HISTORY_REF_HZ)))
STEER_HISTORY = 1 + 3 * HISTORY_STRIDE
LONG_HISTORY = 1 + HISTORY_STRIDE
TWIST_VX_SIGN_BODY_FORWARD = 1.0
ACTOR_ARCHITECTURE_NAME = "lidar_cnn_gru"
LIDAR_DREAMER_ARCHITECTURE_NAME = "lidar_dreamer_ls3c"
TINYLIDARNET_ARCHITECTURE_NAME = "tinylidarnet_gru"
STEERING_ACTION_MODE = "delta"
STEERING_DELTA_MAX_RAD = math.pi / 120.0
MAX_STEER_RAD = 0.33
CLIP_ACTIONS = 1.0
ACT_LIMIT = 1.0
OBS_NORM_CLIP = 10.0
OBS_NORM_EPS = 1e-8
GRU_HIDDEN_DIM = 512
CNN_PROJECTION_DIM = 256
ALLOWED_LIDAR_POOL_BINS = frozenset((16, 32, 64))


def actor_action_dim(horizon=DEFAULT_ACTION_HORIZON, plant_dim=NUM_ACTIONS):
    horizon = int(horizon)
    plant_dim = int(plant_dim)
    if horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {horizon}")
    if plant_dim < 1:
        raise ValueError(f"plant_dim must be >= 1, got {plant_dim}")
    return plant_dim * horizon


def executed_plant_action(actions, *, plant_dim=NUM_ACTIONS):
    """First plant command of a receding-horizon chunk; unused tail is ignored."""
    plant_dim = int(plant_dim)
    if plant_dim < 1:
        raise ValueError(f"plant_dim must be >= 1, got {plant_dim}")
    width = int(actions.shape[-1])
    if width < plant_dim:
        raise ValueError(
            f"action last dim={width} is smaller than plant_dim={plant_dim}"
        )
    return actions[..., :plant_dim]

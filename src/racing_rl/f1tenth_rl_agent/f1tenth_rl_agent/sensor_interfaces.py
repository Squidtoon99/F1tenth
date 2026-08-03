"""Topics, layout offsets, and constants for the 1,097-D sensor racer."""

from __future__ import annotations

from f1tenth_policy.current import applied_current_fraction  # noqa: F401
from f1tenth_policy.layout import (  # noqa: F401
    ACTOR_ARCHITECTURE_NAME,
    ACTOR_LAYOUT_VERSION,
    ACTOR_OBS_DIM as NUM_OBS,
    ARTIFACT_SCOPE_SIM_TRAINING,
    CONTROL_HZ,
    GRAVITY_MS2,
    GRU_HIDDEN_DIM,
    IMU_DIM,
    IMU_START,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_FOV_DEG,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    LIDAR_START,
    MAX_STEER_RAD,
    NUM_ACTIONS,
    OBS_NORM_CLIP,
    OBS_NORM_EPS,
    OBS_PREPROCESSING_VERSION,
    PROPRIO_DIM,
    SENSOR_POLICY_FORMAT_VERSION as POLICY_FORMAT_VERSION,
    STEER_DELTA0,
    STEER_DELTA1,
    STEER_DELTA2,
    STEER_T,
    STEER_T1,
    STEER_T2,
    STEERING_ACTION_MODE,
    STEERING_DELTA_MAX_RAD,
    THROTTLE_CURRENT,
    THROTTLE_PRED,
    VESC_CURRENT,
    VESC_SPEED,
)

# --- Racer command interface --------------------------------------------------
TOPIC_DESIRED_ACTUATOR = "/rl/actuator/desired"
TOPIC_APPLIED_ACTUATOR = "/rl/actuator/applied"

# --- Optional diagnostics (disabled by default) -------------------------------
TOPIC_OBSERVATION = "/sensor_racer/observation"
TOPIC_DIAGNOSTICS = "/sensor_racer/diagnostics"
TOPIC_IMU_RAW_RECORD = "/sensor_racer/imu_raw_record"
TOPIC_IMU_ACTOR_RECORD = "/sensor_racer/imu_actor_record"

# --- Sensor inputs ------------------------------------------------------------
TOPIC_SCAN = "/scan"
TOPIC_IMU = "/sensors/imu/raw"
TOPIC_ODOM = "/odom"

STEER_HISTORY = 4
HIDDEN_LAYERS = [1024, 1024, 1024]
ACT_LIMIT = 1.0
CLIP_ACTIONS = 1.0

# Diagnostics layout (/sensor_racer/diagnostics).
DIAG_PREPROCESS_MS = 0
DIAG_INFER_MS = 1
DIAG_SCAN_AGE_S = 2
DIAG_IMU_AGE_S = 3
DIAG_ODOM_AGE_S = 4
DIAG_APPLIED_AGE_S = 5
DIAG_DEADLINE_MISS = 6
DIAG_STALE_SENSORS = 7
DIAG_VALID_TICK = 8
DIAG_APPLIED_SOURCE = 9
DIAG_GRU_RESET = 10
DIAG_H2D_MS = 11
DIAG_D2H_MS = 12
DIAG_MAP_MS = 13
DIAG_PUBLISH_MS = 14
DIAG_TOTAL_MS = 15
DIAG_CONSECUTIVE_SAFE = 16
DIAG_GRU_RESET_REASON = 17
DIAG_GRU_RESETS = 18
DIAG_LEN = 19

GRU_RESET_NONE = 0
GRU_RESET_UNUSABLE_INPUT = 1
GRU_RESET_DEADLINE_MISS = 2
GRU_RESET_APPLIED_INVALID = 3
GRU_RESET_APPLIED_SAFE = 4
GRU_RESET_APPLIED_TELEOP = 5
GRU_RESET_APPLIED_SAFETY = 6
GRU_RESET_OBSERVATION_NONFINITE = 7
GRU_RESET_ACTION_NONFINITE = 8
GRU_RESET_DUAL_CURRENT = 9

"""Shared topic names, array layouts, and constants for the f1tenth_rl_agent stack.

This module is the code mirror of ``ros2_deploy/INTERFACES.md``. Every node imports
from here so the contract stays consistent across independently-developed nodes.
"""

from __future__ import annotations

# --- Simulator topics (f1tenth_gym_ros) ---------------------------------------
TOPIC_ODOM = "/ego_racecar/odom"
TOPIC_OPP_ODOM = "/ego_racecar/opp_odom"
TOPIC_OPP_RACE_ODOM = "/opp_racecar/odom"
TOPIC_OPP_DRIVE = "/opp_drive"
TOPIC_MAP = "/map"
TOPIC_DRIVE = "/drive"
TOPIC_INITIALPOSE = "/initialpose"
TOPIC_GOAL_POSE = "/goal_pose"

# --- Internal agent topics ----------------------------------------------------
TOPIC_TRACK_CENTERLINE = "/rl/track/centerline"
TOPIC_TRACK_WIDTHS = "/rl/track/widths"
TOPIC_TRACK_MARKERS = "/rl/track/markers"
TOPIC_OBSERVATION = "/rl/observation"
TOPIC_ACTION = "/rl/action"
TOPIC_FUTURE_POINTS = "/rl/obs_debug/future_points"
TOPIC_OBS_DEBUG_SCALARS = "/rl/obs_debug/scalars"
TOPIC_OBS_DEBUG_OPPONENT = "/rl/obs_debug/opponent"
TOPIC_METRICS = "/rl/metrics"

# --- Frames -------------------------------------------------------------------
FRAME_MAP = "map"
FRAME_BASE_LINK = "ego_racecar/base_link"

# --- Dimensions ---------------------------------------------------------------
NUM_OBS_BASE = 384
OPPONENT_OBS_DIM = 8
NUM_OBS = NUM_OBS_BASE + OPPONENT_OBS_DIM
NUM_OBS_1V1 = NUM_OBS
NUM_ACTIONS = 2
NUM_TYRE_SLIP = 8  # [slip_ratio x4, slip_angle x4] per training env
NUM_TYRE_LOAD = 4  # per-wheel normal-load ratio Fz / Fz_static

# Observation field slices (start, stop) within the 384-dim vector.
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
OBS_TYRE_LOAD = (380, 384)
OBS_OPPONENT = (384, 392)

# Match DEFAULT_CONFIG["obs"]["obs_scales"] and clip_obs in config.py. The trainer
# now standardizes observations with a running ObsNormalizer, so the env-side fixed
# scales are 1.0 and clip_obs is only a loose guard. The per-feature scaling that
# matters at deploy is the normalizer (see OBS_NORM_* below and policy_inference).
OBS_LIN_VEL_SCALE = 1.0
OBS_ANG_VEL_SCALE = 1.0
OBS_LIN_ACC_SCALE = 1.0
OBS_CLIP = 50.0

# ObsNormalizer parameters (DEFAULT_CONFIG["obs"]["norm_clip"], ["norm_eps"]).
# Applied at inference to raw observations using the checkpoint's saved obs_norm
# stats, exactly as the trainer applies them before the networks.
OBS_NORM_CLIP = 10.0
OBS_NORM_EPS = 1e-8

# Metrics layout.
METRICS_LEN = 6
METRICS_LAP_COUNT = 0
METRICS_LAST_LAP_TIME = 1
METRICS_MAX_PROGRESS = 2
METRICS_LATERAL_ERROR = 3
METRICS_OOB_FLAG = 4
METRICS_SPEED = 5

# obs_debug scalar layout (/rl/obs_debug/scalars). Decoded by obs_debug_node from
# the raw /rl/observation vector so all fields can be plotted as time series. See
# INTERFACES.md for the full table; keep these indices in sync there.
OBS_DEBUG_LIN_VEL_X = 0
OBS_DEBUG_LIN_VEL_Y = 1
OBS_DEBUG_ANG_VEL_Z = 2
OBS_DEBUG_LIN_ACC_X = 3
OBS_DEBUG_LIN_ACC_Y = 4
OBS_DEBUG_LAST_THROTTLE = 5
OBS_DEBUG_LAST_STEER = 6
OBS_DEBUG_PROGRESS_COS = 7
OBS_DEBUG_PROGRESS_SIN = 8
OBS_DEBUG_HEADING_ERR = 9
OBS_DEBUG_LATERAL_ERR = 10
OBS_DEBUG_CONTACT_FLAG = 11
OBS_DEBUG_SPEED = 12
OBS_DEBUG_MIN_LEFT_MARGIN = 13
OBS_DEBUG_MIN_RIGHT_MARGIN = 14
OBS_DEBUG_OPP_REL_X = 15
OBS_DEBUG_OPP_REL_Y = 16
OBS_DEBUG_OPP_REL_VX = 17
OBS_DEBUG_OPP_REL_VY = 18
OBS_DEBUG_OPP_REL_AX = 19
OBS_DEBUG_OPP_REL_AY = 20
OBS_DEBUG_OPP_GAP_NORM = 21
OBS_DEBUG_OPP_LATERAL = 22
OBS_DEBUG_OPP_PRESENT = 23
OBS_DEBUG_MAX_SLIP_RATIO = 24
OBS_DEBUG_MAX_SLIP_ANGLE = 25
OBS_DEBUG_LEN = 26

# --- Policy / vehicle constants (DEFAULT_CONFIG in config.py) -----------------
MAX_SPEED = 15.0
MAX_STEER = 0.33  # radians at |steer| == 1.0 (real servo hard-clamp; matches training)
CLIP_ACTIONS = 1.0
CONTACT_MARGIN_M = 0.08
FUTURE_TRACK_NUM_POINTS = 60
FUTURE_TRACK_HORIZON_S = 6.0
FUTURE_TRACK_MIN_LOOKAHEAD_M = 5.0
FUTURE_TRACK_WIDTH = 2.2  # deprecated: corridor edges use CSV w_tr_*_m
HIDDEN_LAYERS = [512, 512, 512]
ACT_LIMIT = 1.0
CONTROL_HZ = 10.0
POLICY_FORMAT_VERSION = 2
LONGITUDINAL_MODE = "force"


def expected_num_obs() -> int:
    """Return the fixed policy observation dimension."""
    return NUM_OBS


def default_obs_cfg(enable_opponent_obs: bool = False) -> dict:
    """Return the obs_cfg dict expected by obs_core.build_observation."""
    return {
        "num_obs": expected_num_obs(),
        "base_num_obs": NUM_OBS_BASE,
        "enable_opponent_obs": enable_opponent_obs,
        "opponent_obs_dim": OPPONENT_OBS_DIM,
        "zero_opponent_obs": False,
        "obs_scales": {
            "lin_vel": OBS_LIN_VEL_SCALE,
            "ang_vel": OBS_ANG_VEL_SCALE,
            "lin_acc": OBS_LIN_ACC_SCALE,
        },
        "clip_obs": OBS_CLIP,
        "contact_margin_m": CONTACT_MARGIN_M,
        "future_track_num_points": FUTURE_TRACK_NUM_POINTS,
        "future_track_horizon_s": FUTURE_TRACK_HORIZON_S,
        "future_track_min_lookahead_m": FUTURE_TRACK_MIN_LOOKAHEAD_M,
        "future_track_width": FUTURE_TRACK_WIDTH,
    }

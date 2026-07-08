"""Pure-Python tests for the obs_debug decode/transform helpers.

Runs without ROS 2: only the numpy helpers in obs_debug_viz are exercised (the
marker builder imports ROS message types lazily and is not covered here).
"""

import math

import numpy as np

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import obs_debug_viz as viz


NUM_POINTS = ifc.FUTURE_TRACK_NUM_POINTS


def _make_obs(num_points=NUM_POINTS, *, with_opponent=True):
    """Build a synthetic observation with distinctive, decodable values."""
    dim = ifc.NUM_OBS_1V1 if with_opponent else ifc.NUM_OBS_BASE
    obs = np.zeros(dim, dtype=np.float32)
    obs[0:2] = [1.5, -0.5]          # lin vel
    obs[2] = 0.3                    # ang vel
    obs[3:5] = [2.0, 0.1]           # lin acc
    obs[5:7] = [0.8, -0.2]          # last action
    obs[7:9] = [0.6, 0.8]           # progress cos/sin
    obs[9] = 0.05                   # heading err
    obs[10] = -0.12                 # lateral err
    obs[11] = 1.0                   # contact flag
    # tyre slip [372:380]: 4 ratios, 4 angles
    obs[372:376] = [0.1, -0.4, 0.2, 0.05]
    obs[376:380] = [-0.3, 0.15, 0.25, -0.6]
    return obs


def test_decode_scalars_basic_fields():
    obs = _make_obs()
    s = viz.decode_scalars(obs, NUM_POINTS)

    assert s.shape == (ifc.OBS_DEBUG_LEN,)
    assert s[ifc.OBS_DEBUG_LIN_VEL_X] == np.float32(1.5)
    assert s[ifc.OBS_DEBUG_LIN_VEL_Y] == np.float32(-0.5)
    assert s[ifc.OBS_DEBUG_ANG_VEL_Z] == np.float32(0.3)
    assert s[ifc.OBS_DEBUG_LAST_THROTTLE] == np.float32(0.8)
    assert s[ifc.OBS_DEBUG_LAST_STEER] == np.float32(-0.2)
    assert s[ifc.OBS_DEBUG_HEADING_ERR] == np.float32(0.05)
    assert s[ifc.OBS_DEBUG_LATERAL_ERR] == np.float32(-0.12)
    assert s[ifc.OBS_DEBUG_CONTACT_FLAG] == np.float32(1.0)
    assert s[ifc.OBS_DEBUG_SPEED] == np.float32(math.hypot(1.5, -0.5))


def test_decode_scalars_slip_summary():
    obs = _make_obs()
    s = viz.decode_scalars(obs, NUM_POINTS)
    # max abs of first 4 ratios = 0.4, of last 4 angles = 0.6
    assert s[ifc.OBS_DEBUG_MAX_SLIP_RATIO] == np.float32(0.4)
    assert s[ifc.OBS_DEBUG_MAX_SLIP_ANGLE] == np.float32(0.6)


def test_corridor_margins_constant_corridor():
    # center y=0, left y=+0.7, right y=-0.5 for every sample -> margins 0.7 / 0.5
    future = np.zeros((3, NUM_POINTS, 2), dtype=np.float32)
    future[1, :, 1] = 0.7
    future[2, :, 1] = -0.5
    min_left, min_right = viz.corridor_margins(future)
    assert min_left == np.float32(0.7)
    assert min_right == np.float32(0.5)


def test_corridor_margins_picks_minimum():
    future = np.zeros((3, NUM_POINTS, 2), dtype=np.float32)
    future[1, :, 1] = 1.0
    future[2, :, 1] = -1.0
    # tighten one sample on each side
    future[1, 10, 1] = 0.2
    future[2, 20, 1] = -0.3
    min_left, min_right = viz.corridor_margins(future)
    assert min_left == np.float32(0.2)
    assert min_right == np.float32(0.3)


def test_decode_scalars_margins_wired_from_future():
    obs = _make_obs()
    start, stop = ifc.OBS_FUTURE_POINTS
    fut = np.zeros((3, NUM_POINTS, 2), dtype=np.float32)
    fut[1, :, 1] = 0.6
    fut[2, :, 1] = -0.4
    obs[start:stop] = fut.reshape(-1)
    s = viz.decode_scalars(obs, NUM_POINTS)
    np.testing.assert_allclose(s[ifc.OBS_DEBUG_MIN_LEFT_MARGIN], 0.6, atol=1e-5)
    np.testing.assert_allclose(s[ifc.OBS_DEBUG_MIN_RIGHT_MARGIN], 0.4, atol=1e-5)


def test_opponent_present_decoded():
    obs = _make_obs(with_opponent=True)
    start, _ = ifc.OBS_OPPONENT
    obs[start:start + 7] = [2.0, 0.5, -1.0, 0.2, 0.3, -0.1, 1.0]
    s = viz.decode_scalars(obs, NUM_POINTS)
    assert s[ifc.OBS_DEBUG_OPP_REL_X] == np.float32(2.0)
    assert s[ifc.OBS_DEBUG_OPP_REL_Y] == np.float32(0.5)
    assert s[ifc.OBS_DEBUG_OPP_PRESENT] == np.float32(1.0)


def test_opponent_zero_when_solo_obs():
    obs = _make_obs(with_opponent=False)
    s = viz.decode_scalars(obs, NUM_POINTS)
    assert s[ifc.OBS_DEBUG_OPP_PRESENT] == np.float32(0.0)
    assert s[ifc.OBS_DEBUG_OPP_REL_X] == np.float32(0.0)


def test_opponent_world_xy_ahead_of_ego():
    # ego at (10, 5) facing +y (yaw=pi/2); opponent 2 m straight ahead in ego frame
    wx, wy = viz.opponent_world_xy(2.0, 0.0, 10.0, 5.0, math.pi / 2.0)
    assert math.isclose(wx, 10.0, abs_tol=1e-6)
    assert math.isclose(wy, 7.0, abs_tol=1e-6)


def test_opponent_world_xy_identity_at_origin():
    wx, wy = viz.opponent_world_xy(1.0, 2.0, 0.0, 0.0, 0.0)
    assert math.isclose(wx, 1.0, abs_tol=1e-6)
    assert math.isclose(wy, 2.0, abs_tol=1e-6)

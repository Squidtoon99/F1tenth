"""Decode and visualization helpers for the RL observation debug node.

The pure-numpy functions here decode the raw ``/rl/observation`` vector into the
quantities that matter for diagnosing wall collisions (corridor margins, lateral
error, opponent position, ...). They are kept free of ROS dependencies so they can
be unit-tested directly; the marker builder below is the only function that touches
ROS message types and is shared with ``observation_builder_node``.
"""

from __future__ import annotations

import math

import numpy as np

from f1tenth_rl_agent import interfaces as ifc


def future_block(obs: np.ndarray, num_points: int) -> np.ndarray:
    """Return the future track block reshaped to ``(3, num_points, 2)``.

    Curves are ordered center / left / right, each an ego-frame ``(x, y)`` point.
    """
    start, stop = ifc.OBS_FUTURE_POINTS
    flat = np.asarray(obs[start:stop], dtype=np.float32)
    return flat.reshape(3, num_points, 2)


def corridor_margins(future: np.ndarray) -> tuple[float, float]:
    """Minimum left/right lateral margins (m) of the observed corridor ahead.

    ``future`` is the ``(3, num_points, 2)`` block in the ego frame. The margins are
    measured laterally (ego y) between the center curve and each boundary; the
    minimum over all samples shows the tightest point the policy currently sees.
    """
    center_y = future[0, :, 1]
    left_y = future[1, :, 1]
    right_y = future[2, :, 1]
    min_left = float(np.min(left_y - center_y))
    min_right = float(np.min(center_y - right_y))
    return min_left, min_right


def opponent_world_xy(
    rel_x: float, rel_y: float, ego_x: float, ego_y: float, ego_yaw: float
) -> tuple[float, float]:
    """Reconstruct the opponent world position from its ego-frame relative offset."""
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    wx = ego_x + c * rel_x - s * rel_y
    wy = ego_y + s * rel_x + c * rel_y
    return wx, wy


def decode_scalars(obs: np.ndarray, num_points: int) -> np.ndarray:
    """Decode the raw observation into the fixed ``OBS_DEBUG_*`` scalar layout."""
    obs = np.asarray(obs, dtype=np.float32)
    out = np.zeros(ifc.OBS_DEBUG_LEN, dtype=np.float32)

    vx, vy = float(obs[0]), float(obs[1])
    out[ifc.OBS_DEBUG_LIN_VEL_X] = vx
    out[ifc.OBS_DEBUG_LIN_VEL_Y] = vy
    out[ifc.OBS_DEBUG_ANG_VEL_Z] = float(obs[ifc.OBS_ANG_VEL[0]])
    out[ifc.OBS_DEBUG_LIN_ACC_X] = float(obs[ifc.OBS_LIN_ACC[0]])
    out[ifc.OBS_DEBUG_LIN_ACC_Y] = float(obs[ifc.OBS_LIN_ACC[0] + 1])
    out[ifc.OBS_DEBUG_LAST_THROTTLE] = float(obs[ifc.OBS_LAST_ACTION[0]])
    out[ifc.OBS_DEBUG_LAST_STEER] = float(obs[ifc.OBS_LAST_ACTION[0] + 1])
    out[ifc.OBS_DEBUG_PROGRESS_COS] = float(obs[ifc.OBS_TRACK_PROGRESS[0]])
    out[ifc.OBS_DEBUG_PROGRESS_SIN] = float(obs[ifc.OBS_TRACK_PROGRESS[0] + 1])
    out[ifc.OBS_DEBUG_HEADING_ERR] = float(obs[ifc.OBS_CENTERLINE_ANGLE[0]])
    out[ifc.OBS_DEBUG_LATERAL_ERR] = float(obs[ifc.OBS_CENTERLINE_DISTANCE[0]])
    out[ifc.OBS_DEBUG_CONTACT_FLAG] = float(obs[ifc.OBS_CONTACT_FLAG[0]])
    out[ifc.OBS_DEBUG_SPEED] = float(math.hypot(vx, vy))

    min_left, min_right = corridor_margins(future_block(obs, num_points))
    out[ifc.OBS_DEBUG_MIN_LEFT_MARGIN] = min_left
    out[ifc.OBS_DEBUG_MIN_RIGHT_MARGIN] = min_right

    slip_start, slip_stop = ifc.OBS_TYRE_SLIP
    slip = np.abs(obs[slip_start:slip_stop])
    if slip.size >= 8:
        out[ifc.OBS_DEBUG_MAX_SLIP_RATIO] = float(np.max(slip[:4]))
        out[ifc.OBS_DEBUG_MAX_SLIP_ANGLE] = float(np.max(slip[4:8]))

    opp_start, opp_stop = ifc.OBS_OPPONENT
    if obs.shape[0] >= opp_stop:
        opp = obs[opp_start:opp_stop]
        out[ifc.OBS_DEBUG_OPP_REL_X] = float(opp[0])
        out[ifc.OBS_DEBUG_OPP_REL_Y] = float(opp[1])
        out[ifc.OBS_DEBUG_OPP_REL_VX] = float(opp[2])
        out[ifc.OBS_DEBUG_OPP_REL_VY] = float(opp[3])
        out[ifc.OBS_DEBUG_OPP_GAP_NORM] = float(opp[4])
        out[ifc.OBS_DEBUG_OPP_LATERAL] = float(opp[5])
        out[ifc.OBS_DEBUG_OPP_PRESENT] = float(opp[6])

    return out


def build_future_markers(
    future: np.ndarray,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    frame_id: str,
    stamp,
):
    """Build a MarkerArray of the ego-frame future curves transformed into world.

    ``future`` is the ``(3, num_points, 2)`` block. Returns a
    ``visualization_msgs/MarkerArray`` with one LINE_STRIP per curve (center, left,
    right). Imports of ROS message types are local so the pure decode helpers above
    can be used without a ROS environment.
    """
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    colors = [(1.0, 1.0, 0.0, 1.0), (1.0, 0.2, 0.2, 1.0), (0.2, 0.4, 1.0, 1.0)]
    arr = MarkerArray()
    for curve_idx in range(future.shape[0]):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "future_points"
        m.id = curve_idx
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = colors[curve_idx]
        m.pose.orientation.w = 1.0
        for k in range(future.shape[1]):
            xe, ye = float(future[curve_idx, k, 0]), float(future[curve_idx, k, 1])
            wx = ego_x + c * xe - s * ye
            wy = ego_y + s * xe + c * ye
            m.points.append(Point(x=wx, y=wy, z=0.0))
        arr.markers.append(m)
    return arr

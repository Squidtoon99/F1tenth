"""Unit tests for the quasi-static tyre-load estimate (ADR 0002 follow-up).

Pure numpy: no ROS, no torch, no genesis. Verifies the closed-form load-transfer
signs and the at-rest sentinel used when body accelerations are zero.
"""

import numpy as np

from f1tenth_rl_agent.obs_core import quasi_static_load_ratio


def test_at_rest_is_static_ratio():
    ratio = quasi_static_load_ratio(0.0, 0.0)
    assert np.allclose(ratio, [1.0, 1.0, 1.0, 1.0])


def test_forward_accel_shifts_load_rearward():
    # +ax (forward) loads the rear axle and unloads the front. Order [LR, RR, LF, RF].
    ratio = quasi_static_load_ratio(4.0, 0.0)
    assert ratio[0] > 1.0 and ratio[1] > 1.0  # rear
    assert ratio[2] < 1.0 and ratio[3] < 1.0  # front
    # Pure longitudinal transfer leaves the left/right wheels of an axle balanced.
    assert np.isclose(ratio[0], ratio[1])
    assert np.isclose(ratio[2], ratio[3])


def test_left_turn_shifts_load_to_right_wheels():
    # +ay (leftward accel, i.e. a left turn) loads the right (outer) wheels.
    ratio = quasi_static_load_ratio(0.0, 4.0)
    assert ratio[1] > ratio[0]  # RR > LR
    assert ratio[3] > ratio[2]  # RF > LF
    # Pure lateral transfer keeps each axle's mean at the static ratio.
    assert np.isclose(0.5 * (ratio[0] + ratio[1]), 1.0)
    assert np.isclose(0.5 * (ratio[2] + ratio[3]), 1.0)


def test_never_negative_under_extreme_accel():
    ratio = quasi_static_load_ratio(-50.0, 50.0)
    assert (ratio >= 0.0).all()

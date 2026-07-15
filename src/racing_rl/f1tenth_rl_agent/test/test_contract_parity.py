"""Parity: f1tenth_contract (shared format) vs the deployed interfaces.py.

f1tenth_contract is the additive source-of-truth for the observation/action
*format*. The deployed nodes still read their layout from ``interfaces.py`` (the
math lives there). This test asserts the two agree so the contract cannot silently
drift from what actually ships on the car. Pure Python - no torch/genesis needed.
"""

import pytest

contract = pytest.importorskip("f1tenth_contract")

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402


def test_dimensions_match():
    assert contract.NUM_OBS == ifc.NUM_OBS
    assert contract.NUM_OBS_BASE == ifc.NUM_OBS_BASE
    assert contract.NUM_OBS_1V1 == ifc.NUM_OBS_1V1
    assert contract.OPPONENT_OBS_DIM == ifc.OPPONENT_OBS_DIM
    assert contract.NUM_ACTIONS == ifc.NUM_ACTIONS
    assert contract.NUM_TYRE_SLIP == ifc.NUM_TYRE_SLIP
    assert contract.NUM_TYRE_LOAD == ifc.NUM_TYRE_LOAD


def test_opponent_block_is_six_dims_without_presence():
    """The 1v1 layout appends a 6-dim opponent block (no presence flag): 384 -> 390."""
    assert ifc.OPPONENT_OBS_DIM == 6
    assert ifc.NUM_OBS_1V1 == ifc.NUM_OBS_BASE + 6 == 390
    assert tuple(ifc.OBS_OPPONENT) == (ifc.NUM_OBS_BASE, ifc.NUM_OBS_1V1)


def test_field_slices_match():
    pairs = {
        "OBS_LIN_VEL": (contract.OBS_LIN_VEL, ifc.OBS_LIN_VEL),
        "OBS_ANG_VEL": (contract.OBS_ANG_VEL, ifc.OBS_ANG_VEL),
        "OBS_LIN_ACC": (contract.OBS_LIN_ACC, ifc.OBS_LIN_ACC),
        "OBS_LAST_ACTION": (contract.OBS_LAST_ACTION, ifc.OBS_LAST_ACTION),
        "OBS_TRACK_PROGRESS": (contract.OBS_TRACK_PROGRESS, ifc.OBS_TRACK_PROGRESS),
        "OBS_CENTERLINE_ANGLE": (
            contract.OBS_CENTERLINE_ANGLE,
            ifc.OBS_CENTERLINE_ANGLE,
        ),
        "OBS_CENTERLINE_DISTANCE": (
            contract.OBS_CENTERLINE_DISTANCE,
            ifc.OBS_CENTERLINE_DISTANCE,
        ),
        "OBS_CONTACT_FLAG": (contract.OBS_CONTACT_FLAG, ifc.OBS_CONTACT_FLAG),
        "OBS_FUTURE_POINTS": (contract.OBS_FUTURE_POINTS, ifc.OBS_FUTURE_POINTS),
        "OBS_TYRE_SLIP": (contract.OBS_TYRE_SLIP, ifc.OBS_TYRE_SLIP),
        "OBS_TYRE_LOAD": (contract.OBS_TYRE_LOAD, ifc.OBS_TYRE_LOAD),
        "OBS_OPPONENT": (contract.OBS_OPPONENT, ifc.OBS_OPPONENT),
    }
    for name, (a, b) in pairs.items():
        assert tuple(a) == tuple(b), f"{name}: contract {a} != interfaces {b}"


def test_action_mapping_constants_match():
    assert contract.MAX_SPEED == ifc.MAX_SPEED
    assert contract.MAX_STEER == ifc.MAX_STEER
    assert contract.CLIP_ACTIONS == ifc.CLIP_ACTIONS
    assert contract.ACT_LIMIT == ifc.ACT_LIMIT
    assert contract.CONTROL_HZ == ifc.CONTROL_HZ


def test_expected_num_obs_matches():
    assert contract.expected_num_obs() == ifc.expected_num_obs() == 390


def test_contract_fields_contiguous():
    """The canonical fields must tile [0, NUM_OBS) with no gaps."""
    cursor = 0
    for _name, start, stop in contract.OBSERVATION.fields:
        assert start == cursor, f"gap/overlap before {_name}: {start} != {cursor}"
        cursor = stop
    assert cursor == contract.NUM_OBS

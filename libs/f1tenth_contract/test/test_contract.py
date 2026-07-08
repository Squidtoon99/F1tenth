"""Contract self-consistency tests.

These check the observation/action *format* is internally consistent (dims,
contiguous field slices). Cross-checks against the deployed ``interfaces.py`` and
the C++ mirror live in their respective packages (f1tenth_rl_agent /
f1tenth_common) so those consumers cannot silently drift from this contract.
"""

from f1tenth_contract import (
    NUM_OBS_1V1,
    NUM_OBS_BASE,
    OPPONENT_OBS_DIM,
    OBSERVATION,
    OBSERVATION_1V1,
    ActionSpec,
    ObservationSpec,
    expected_num_obs,
)
from f1tenth_contract.observation import OBS_FIELDS_BASE


def test_action_dim_is_two():
    assert ActionSpec().dim == 2
    assert ActionSpec().index_of("steer") == 1


def test_base_observation_dim():
    assert OBSERVATION.dim == NUM_OBS_BASE == 380


def test_opponent_observation_dim():
    assert NUM_OBS_1V1 == NUM_OBS_BASE + OPPONENT_OBS_DIM == 387
    assert OBSERVATION_1V1.dim == NUM_OBS_1V1


def test_expected_num_obs():
    assert expected_num_obs(False) == 380
    assert expected_num_obs(True) == 387


def test_base_fields_are_contiguous():
    offset = 0
    for _name, start, stop in OBS_FIELDS_BASE:
        assert start == offset, f"gap/overlap at {start} (expected {offset})"
        assert stop > start
        offset = stop
    assert offset == NUM_OBS_BASE


def test_index_and_slice_lookup():
    spec = ObservationSpec(fields=(("a", 0, 1), ("b", 1, 4)))
    assert spec.dim == 4
    assert spec.index_of("a") == 0
    assert spec.index_of("b") == 1
    assert spec.slice_of("b") == (1, 4)

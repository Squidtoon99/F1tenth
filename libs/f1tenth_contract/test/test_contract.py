"""Basic contract sanity tests + a place to export parity fixtures.

The C++ mirror in src/common/f1tenth_common is validated against the values
produced here (see that package's test_obs_parity.cpp).
"""

from f1tenth_contract import ActionSpec, ObservationSpec


def test_action_dim_is_two():
    assert ActionSpec().dim == 2


def test_observation_index_lookup():
    spec = ObservationSpec(fields=(("a", 1), ("b", 3)))
    assert spec.dim == 4
    assert spec.index_of("a") == 0
    assert spec.index_of("b") == 1

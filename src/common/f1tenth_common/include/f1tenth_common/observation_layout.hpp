// C++ mirror of the observation/action layout defined in Python at
// libs/f1tenth_contract. The on-car C++ vehicle node cannot import Python, so
// this header must stay in sync with the Python contract. A parity test
// (test/test_obs_parity.cpp) guards against drift.
//
// Scaffold placeholder: fill in fields/offsets to match the Python contract when
// the observation space is finalized.
#ifndef F1TENTH_COMMON__OBSERVATION_LAYOUT_HPP_
#define F1TENTH_COMMON__OBSERVATION_LAYOUT_HPP_

#include <cstddef>

namespace f1tenth_common
{

// TODO: keep these in lockstep with libs/f1tenth_contract.
struct ObservationLayout
{
  // Example placeholders; replace with the real contract.
  static constexpr std::size_t kObservationDim = 0;
  static constexpr std::size_t kActionDim = 2;  // (throttle, steer)
};

}  // namespace f1tenth_common

#endif  // F1TENTH_COMMON__OBSERVATION_LAYOUT_HPP_

// Copyright 2026 F1TENTH Racing Stack contributors
//
// C++ mirror of the observation/action format defined in Python at
// libs/f1tenth_contract. The on-car C++ nodes cannot import Python, so this header
// must stay in sync with the Python contract. A parity test
// (test/test_obs_parity.cpp) guards against drift.
//
// Format only: dimensions and the [start, stop) slice of each field. The
// field-building math lives with each consumer.
#ifndef F1TENTH_COMMON__OBSERVATION_LAYOUT_HPP_
#define F1TENTH_COMMON__OBSERVATION_LAYOUT_HPP_

#include <cstddef>

namespace f1tenth_common
{

// Mirror of libs/f1tenth_contract (observation.py / action.py).
struct ObservationLayout
{
  // Dimensions.
  static constexpr std::size_t kObservationDim = 384;       // base (solo)
  static constexpr std::size_t kOpponentObsDim = 6;         // appended for 1v1
  static constexpr std::size_t kObservationDim1v1 = 390;    // base + opponent
  static constexpr std::size_t kActionDim = 2;              // (throttle, steer)
  static constexpr std::size_t kTyreSlipDim = 8;
  static constexpr std::size_t kTyreLoadDim = 4;

  // Field slices [start, stop) within the base 384-dim vector.
  static constexpr std::size_t kLinVelStart = 0, kLinVelStop = 2;
  static constexpr std::size_t kAngVelStart = 2, kAngVelStop = 3;
  static constexpr std::size_t kLinAccStart = 3, kLinAccStop = 5;
  static constexpr std::size_t kLastActionStart = 5, kLastActionStop = 7;
  static constexpr std::size_t kTrackProgressStart = 7, kTrackProgressStop = 9;
  static constexpr std::size_t kCenterlineAngleStart = 9, kCenterlineAngleStop = 10;
  static constexpr std::size_t kCenterlineDistanceStart = 10, kCenterlineDistanceStop = 11;
  static constexpr std::size_t kContactFlagStart = 11, kContactFlagStop = 12;
  static constexpr std::size_t kFuturePointsStart = 12, kFuturePointsStop = 372;
  static constexpr std::size_t kTyreSlipStart = 372, kTyreSlipStop = 380;
  static constexpr std::size_t kTyreLoadStart = 380, kTyreLoadStop = 384;
  static constexpr std::size_t kOpponentStart = 384, kOpponentStop = 390;
};

}  // namespace f1tenth_common

#endif  // F1TENTH_COMMON__OBSERVATION_LAYOUT_HPP_

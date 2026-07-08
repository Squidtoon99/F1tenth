// Copyright 2026 F1TENTH Racing Stack contributors
//
// Parity test: verifies the C++ observation mirror matches the Python contract
// (libs/f1tenth_contract) dimensions and field slices.
#include <gtest/gtest.h>

#include "f1tenth_common/observation_layout.hpp"

using L = f1tenth_common::ObservationLayout;

TEST(ObservationParity, Dimensions)
{
  EXPECT_EQ(L::kObservationDim, 380u);
  EXPECT_EQ(L::kOpponentObsDim, 7u);
  EXPECT_EQ(L::kObservationDim1v1, 387u);
  EXPECT_EQ(L::kObservationDim + L::kOpponentObsDim, L::kObservationDim1v1);
  EXPECT_EQ(L::kActionDim, 2u);
  EXPECT_EQ(L::kTyreSlipDim, 8u);
}

TEST(ObservationParity, BaseFieldsContiguous)
{
  // Fields must tile [0, 380) with no gaps/overlaps, in order.
  EXPECT_EQ(L::kLinVelStart, 0u);
  EXPECT_EQ(L::kLinVelStop, L::kAngVelStart);
  EXPECT_EQ(L::kAngVelStop, L::kLinAccStart);
  EXPECT_EQ(L::kLinAccStop, L::kLastActionStart);
  EXPECT_EQ(L::kLastActionStop, L::kTrackProgressStart);
  EXPECT_EQ(L::kTrackProgressStop, L::kCenterlineAngleStart);
  EXPECT_EQ(L::kCenterlineAngleStop, L::kCenterlineDistanceStart);
  EXPECT_EQ(L::kCenterlineDistanceStop, L::kContactFlagStart);
  EXPECT_EQ(L::kContactFlagStop, L::kFuturePointsStart);
  EXPECT_EQ(L::kFuturePointsStop, L::kTyreSlipStart);
  EXPECT_EQ(L::kTyreSlipStop, L::kObservationDim);
  EXPECT_EQ(L::kOpponentStart, L::kObservationDim);
  EXPECT_EQ(L::kOpponentStop, L::kObservationDim1v1);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Parity test placeholder: verifies the C++ observation mirror matches the
// Python contract (libs/f1tenth_contract) on shared fixtures.
//
// When the contract is implemented, load the fixtures (e.g. a JSON dumped by the
// Python side) and assert the C++ layout produces identical values.
#include <gtest/gtest.h>

#include "f1tenth_common/observation_layout.hpp"

TEST(ObservationParity, ActionDimIsTwo)
{
  EXPECT_EQ(f1tenth_common::ObservationLayout::kActionDim, 2u);
}

// TODO: add a fixture-based parity test against libs/f1tenth_contract once the
// observation layout is finalized.

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

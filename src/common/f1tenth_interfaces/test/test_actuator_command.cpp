// Copyright 2026 F1TENTH Racing Stack contributors
//
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>

#include "f1tenth_interfaces/msg/actuator_command.hpp"

TEST(ActuatorCommandSerialization, round_trip_preserves_fields) {
  f1tenth_interfaces::msg::ActuatorCommand msg;
  msg.header.stamp.sec = 42;
  msg.header.stamp.nanosec = 123456789;
  msg.header.frame_id = "base_link";
  msg.generation = 7;
  msg.observation_stamp.sec = 41;
  msg.observation_stamp.nanosec = 987654321;
  msg.drive_current_a = 3.5;
  msg.brake_current_a = 0.0;
  msg.servo_position = 0.4495;
  msg.longitudinal = 0.25;
  msg.steering = -0.12;
  msg.source = f1tenth_interfaces::msg::ActuatorCommand::SOURCE_RL;

  rclcpp::Serialization<f1tenth_interfaces::msg::ActuatorCommand> serializer;
  rclcpp::SerializedMessage serialized;
  serializer.serialize_message(&msg, &serialized);

  f1tenth_interfaces::msg::ActuatorCommand restored;
  serializer.deserialize_message(&serialized, &restored);

  EXPECT_EQ(restored.header.stamp.sec, 42);
  EXPECT_EQ(restored.header.stamp.nanosec, 123456789u);
  EXPECT_EQ(restored.header.frame_id, "base_link");
  EXPECT_EQ(restored.generation, 7u);
  EXPECT_EQ(restored.observation_stamp.sec, 41);
  EXPECT_EQ(restored.observation_stamp.nanosec, 987654321u);
  EXPECT_DOUBLE_EQ(restored.drive_current_a, 3.5);
  EXPECT_DOUBLE_EQ(restored.brake_current_a, 0.0);
  EXPECT_DOUBLE_EQ(restored.servo_position, 0.4495);
  EXPECT_FLOAT_EQ(restored.longitudinal, 0.25f);
  EXPECT_FLOAT_EQ(restored.steering, -0.12f);
  EXPECT_EQ(restored.source, f1tenth_interfaces::msg::ActuatorCommand::SOURCE_RL);
}

TEST(ActuatorCommandConstants, source_values_are_stable) {
  EXPECT_EQ(f1tenth_interfaces::msg::ActuatorCommand::SOURCE_SAFE, 0);
  EXPECT_EQ(f1tenth_interfaces::msg::ActuatorCommand::SOURCE_RL, 1);
  EXPECT_EQ(f1tenth_interfaces::msg::ActuatorCommand::SOURCE_TELEOP, 2);
  EXPECT_EQ(f1tenth_interfaces::msg::ActuatorCommand::SOURCE_SAFETY, 3);
}

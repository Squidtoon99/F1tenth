// drive_node: convert policy actions into Ackermann drive commands on the car.
//
// Lean C++ port of the sim drive_command_node. Subscribes to /rl/action and
// publishes /drive (AckermannDriveStamped) straight to the f1tenth_stack
// ackermann_mux autonomous input. A watchdog commands a safe stop if actions stop
// arriving. speed_limit_mps caps the commanded speed for staged shakedown runs;
// teleop on the mux always overrides regardless.
//
// motor_mode:
//   "speed" (default) — longitudinal action maps to drive.speed (ERPM path).
//   "force" — longitudinal action maps to drive.acceleration in [-1,1]; speed=0.
//             Requires vesc_actuator (enabled) to convert acceleration to
//             motor current / brake current (see ADR 0005).
#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include "f1tenth_rl_vehicle/rl_obs_core.hpp"

namespace f1tenth_rl_vehicle
{

class DriveNode : public rclcpp::Node
{
public:
  DriveNode()
  : rclcpp::Node("drive")
  {
    max_speed_ = declare_parameter<double>("max_speed", 15.0);
    max_steer_ = declare_parameter<double>("max_steer", 0.44);
    clip_actions_ = declare_parameter<double>("clip_actions", 1.0);
    speed_limit_mps_ = declare_parameter<double>("speed_limit_mps", 15.0);
    watchdog_timeout_s_ = declare_parameter<double>("watchdog_timeout_s", 0.5);
    brake_behavior_ = declare_parameter<std::string>("brake_behavior", "stop");
    motor_mode_ = declare_parameter<std::string>("motor_mode", "speed");
    enable_output_filter_ = declare_parameter<bool>("enable_output_filter", true);
    t_delta_ = declare_parameter<double>("t_delta", 0.1);
    control_dt_ = declare_parameter<double>("control_dt", 0.1);
    steer_lag_alpha_ = lagAlpha(control_dt_, t_delta_);
    const std::string action_topic =
      declare_parameter<std::string>("action_topic", "/rl/action");
    const std::string drive_topic = declare_parameter<std::string>("drive_topic", "/drive");
    frame_id_ = declare_parameter<std::string>("frame_id", "base_link");

    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);
    action_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      action_topic, 10,
      [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->onAction(*msg);});

    watchdog_ = create_wall_timer(
      std::chrono::duration<double>(0.1),
      [this]() {this->onWatchdog();});

    RCLCPP_INFO(
      get_logger(),
      "drive ready: motor_mode=%s max_speed=%.2f max_steer=%.3f speed_limit=%.2f "
      "watchdog=%.2fs output_filter=%s t_delta=%.3f alpha=%.3f",
      motor_mode_.c_str(), max_speed_, max_steer_, speed_limit_mps_, watchdog_timeout_s_,
      enable_output_filter_ ? "on" : "off", t_delta_, steer_lag_alpha_);
  }

private:
  void publishDrive(double speed, double steering_angle, double acceleration)
  {
    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = now();
    msg.header.frame_id = frame_id_;
    msg.drive.speed = static_cast<float>(speed);
    msg.drive.steering_angle = static_cast<float>(steering_angle);
    msg.drive.acceleration = static_cast<float>(acceleration);
    drive_pub_->publish(msg);
  }

  void onAction(const std_msgs::msg::Float32MultiArray & msg)
  {
    if (msg.data.size() < 2) {
      RCLCPP_WARN(get_logger(), "action message has < 2 elements; ignoring");
      return;
    }

    double speed = 0.0;
    double acceleration = 0.0;
    double steering_angle = 0.0;

    if (motor_mode_ == "force") {
      auto [long_cmd, steer] = mapActionToForce(
        msg.data[0], msg.data[1], max_steer_, clip_actions_);
      acceleration = long_cmd;
      steering_angle = steer;
      speed = 0.0;
    } else {
      auto [spd, steer] = mapActionToDrive(
        msg.data[0], msg.data[1], max_speed_, max_steer_, clip_actions_, brake_behavior_);
      speed = spd;
      steering_angle = steer;
      // Staging speed cap (does not affect steering).
      if (speed > speed_limit_mps_) {
        speed = speed_limit_mps_;
      }
    }

    if (enable_output_filter_) {
      filtered_steer_ = stepFirstOrderLag(filtered_steer_, steering_angle, steer_lag_alpha_);
      steering_angle = filtered_steer_;
    }
    publishDrive(speed, steering_angle, acceleration);
    last_action_time_ = now();
    have_action_ = true;
  }

  void onWatchdog()
  {
    if (!have_action_) {
      return;
    }
    const double elapsed = (now() - last_action_time_).seconds();
    if (elapsed > watchdog_timeout_s_) {
      filtered_steer_ = 0.0;
      // Force mode: acceleration=-1 requests full brake via vesc_actuator.
      const double accel = (motor_mode_ == "force") ? -1.0 : 0.0;
      publishDrive(0.0, 0.0, accel);
    }
  }

  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr action_sub_;
  rclcpp::TimerBase::SharedPtr watchdog_;

  double max_speed_ = 15.0;
  double max_steer_ = 0.44;
  double clip_actions_ = 1.0;
  double speed_limit_mps_ = 15.0;
  double watchdog_timeout_s_ = 0.5;
  std::string brake_behavior_ = "stop";
  std::string motor_mode_ = "speed";
  std::string frame_id_ = "base_link";
  bool enable_output_filter_ = true;
  double t_delta_ = 0.1;
  double control_dt_ = 0.1;
  double steer_lag_alpha_ = 0.5;
  double filtered_steer_ = 0.0;

  bool have_action_ = false;
  rclcpp::Time last_action_time_;
};

}  // namespace f1tenth_rl_vehicle

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<f1tenth_rl_vehicle::DriveNode>());
  rclcpp::shutdown();
  return 0;
}

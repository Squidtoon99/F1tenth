// vehicle_obs_node: build the policy observation on the real car (384 base, or 390
// with the 6-dim opponent block appended at [384:390) for the 1v1 layout).
//
// Merges the sim stack's track_server + observation_builder (+ odom adapter) into a
// single lean C++ node. It loads the training centerline CSV directly, samples the
// latest map-frame pose (particle filter) and body-frame twist (VESC odom) on a
// fixed 10 Hz timer, reconstructs the exact training observation via rl_obs_core,
// and publishes /rl/observation.
//
// Tyre-slip observation dims [372:380] default to zeros (no per-wheel sensing on the
// car), matching the gym deploy. When 'enable_slip_estimation' is true the node
// instead estimates the 8-dim block ([slip_ratio x4, slip_angle x4], wheel order
// [LR, RR, LF, RF]) from the VESC IMU (sensors/imu/raw: lateral accel + yaw gyro)
// fused with a particle-filter ground-velocity estimate:
//   - A complementary filter blends IMU lateral-accel integration (drift-prone but
//     low-noise high-freq) with PF-derived lateral velocity (noisy but drift-free)
//     to recover body lateral velocity -> per-wheel slip angle.
//   - Longitudinal slip ratio compares the VESC wheel speed (ERPM-derived /odom vx)
//     against the PF ground forward speed (captures wheelspin / brake lockup).
//   - The undriven FRONT slip-ratio channels are physically ~0 but the training
//     distribution puts them near a sim free-wheel artifact mean; they (and the
//     low-speed fallback) are filled from 'slip_obs_mean' so they normalize to ~0.
// Acceleration is a fixed-step finite difference of body velocity at the control rate.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include "f1tenth_rl_vehicle/rl_obs_core.hpp"

namespace f1tenth_rl_vehicle
{

namespace
{
// Base (solo) observation dimension; the opponent block is appended at [384:390).
// Mirror of f1tenth_common::ObservationLayout::kObservationDim.
constexpr int kBaseObsDim = 384;

double quatToYaw(double x, double y, double z, double w)
{
  double siny_cosp = 2.0 * (w * z + x * y);
  double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}
}  // namespace

class VehicleObsNode : public rclcpp::Node
{
public:
  VehicleObsNode()
  : rclcpp::Node("vehicle_obs")
  {
    const std::string track_csv = declare_parameter<std::string>("track_csv", "");
    control_hz_ = declare_parameter<double>("control_hz", 10.0);
    const std::string pose_topic =
      declare_parameter<std::string>("pose_topic", "/pf/pose/odom");
    const std::string twist_topic =
      declare_parameter<std::string>("twist_topic", "/odom");
    const std::string action_topic =
      declare_parameter<std::string>("action_topic", "/rl/action");
    const std::string obs_topic =
      declare_parameter<std::string>("obs_topic", "/rl/observation");
    twist_in_world_frame_ = declare_parameter<bool>("twist_in_world_frame", false);

    enable_opponent_obs_ = declare_parameter<bool>("enable_opponent_obs", false);
    zero_opponent_obs_ = declare_parameter<bool>("zero_opponent_obs", false);
    const std::string opp_odom_topic =
      declare_parameter<std::string>("opponent_odom_topic", "/rl/opponent/odom");
    opponent_timeout_s_ = declare_parameter<double>("opponent_timeout_s", 0.5);

    // --- tyre-slip estimation (off by default -> zeros, gym parity) -----------
    enable_slip_estimation_ = declare_parameter<bool>("enable_slip_estimation", false);
    const std::string imu_topic =
      declare_parameter<std::string>("imu_topic", "/sensors/imu/raw");
    // VESC firmware reports accel in g and gyro in deg/s; the driver copies them
    // through unscaled, so convert here. Signs/axes are calibrated on-car.
    imu_accel_to_ms2_ = declare_parameter<double>("imu_accel_to_ms2", 9.80665);
    imu_gyro_to_rads_ = declare_parameter<double>("imu_gyro_to_rads", M_PI / 180.0);
    imu_ay_sign_ = declare_parameter<double>("imu_ay_sign", 1.0);
    imu_yaw_rate_sign_ = declare_parameter<double>("imu_yaw_rate_sign", 1.0);
    imu_use_for_yaw_rate_ = declare_parameter<bool>("imu_use_for_yaw_rate", true);
    // Vehicle geometry (f1tenth_sim VehicleParams / URDF defaults).
    wheel_radius_ = declare_parameter<double>("wheel_radius_m", 0.05);
    lf_ = declare_parameter<double>("lf_m", 0.1773);
    lr_ = declare_parameter<double>("lr_m", 0.1477);
    track_width_ = declare_parameter<double>("track_width_m", 0.20);
    half_track_ = 0.5 * track_width_;
    max_steer_ = declare_parameter<double>("max_steer_rad", 0.33);
    // --- quasi-static tyre-load estimation (off by default -> static ratio 1.0,
    // matching the deploy default and the C++ parity fixture; ADR 0002 follow-up).
    enable_load_estimation_ = declare_parameter<bool>("enable_load_estimation", false);
    cg_height_ = declare_parameter<double>("cg_height_m", 0.05);
    roll_stiffness_front_ = declare_parameter<double>("roll_stiffness_front", 0.5);
    slip_min_lat_ = declare_parameter<double>("slip_min_lat", 0.2);
    slip_min_active_long_ = declare_parameter<double>("slip_min_active_long", 0.1);
    slip_min_passive_long_ = declare_parameter<double>("slip_min_passive_long", 0.4);
    // Complementary-filter time constant for body lateral velocity (s) and a
    // low-pass for the PF ground forward speed used in slip ratio.
    vy_filter_tau_s_ = declare_parameter<double>("vy_filter_tau_s", 0.5);
    vx_ground_lp_alpha_ = declare_parameter<double>("vx_ground_lp_alpha", 0.5);
    // Below this ground speed slip is ill-defined -> emit slip_obs_mean.
    slip_speed_min_ = declare_parameter<double>("slip_speed_min_mps", 0.3);
    // Per-channel training means (checkpoint obs_norm). Used as the low-speed
    // fallback and to fill the undriven front slip-ratio channels.
    slip_obs_mean_ = declare_parameter<std::vector<double>>(
      "slip_obs_mean",
      std::vector<double>{-0.033, -0.034, 0.969, 0.970, -0.011, -0.006, -0.113, -0.030});
    if (slip_obs_mean_.size() != 8) {
      slip_obs_mean_.assign(8, 0.0);
    }

    ObsConfig cfg;
    cfg.num_obs = static_cast<int>(declare_parameter<int>("num_obs", 384));
    cfg.future_track_num_points =
      static_cast<int>(declare_parameter<int>("future_track_num_points", 60));
    cfg.future_track_horizon_s = declare_parameter<double>("future_track_horizon_s", 6.0);
    cfg.future_track_min_lookahead_m =
      declare_parameter<double>("future_track_min_lookahead_m", 5.0);
    cfg.future_track_width = declare_parameter<double>("future_track_width", 2.2);
    cfg.contact_margin_m = declare_parameter<double>("contact_margin_m", 0.08);
    cfg.clip_obs = declare_parameter<double>("clip_obs", 50.0);
    cfg.lin_vel_scale = declare_parameter<double>("lin_vel_scale", 1.0);
    cfg.ang_vel_scale = declare_parameter<double>("ang_vel_scale", 1.0);
    cfg.lin_acc_scale = declare_parameter<double>("lin_acc_scale", 1.0);
    const int coarse_stride = static_cast<int>(declare_parameter<int>("coarse_stride", 10));

    // 1v1: append the 6-dim opponent block. If num_obs was left at the solo
    // default, bump it to the 1v1 size so the assembled vector matches the policy.
    cfg.enable_opponent_obs = enable_opponent_obs_;
    cfg.zero_opponent_obs = zero_opponent_obs_;
    opponent_obs_dim_ = cfg.opponent_obs_dim;
    if (enable_opponent_obs_ && cfg.num_obs < kBaseObsDim + cfg.opponent_obs_dim) {
      cfg.num_obs = kBaseObsDim + cfg.opponent_obs_dim;
    }

    if (track_csv.empty()) {
      throw std::runtime_error("vehicle_obs: 'track_csv' parameter is required");
    }
    TrackData track = loadTrackCsv(track_csv);
    builder_ = std::make_unique<TrackObservationBuilder>(
      track.x, track.y, track.w_left, track.w_right, cfg, coarse_stride);
    RCLCPP_INFO(
      get_logger(), "Loaded track '%s' (%d points); obs dim %d at %.1f Hz",
      track_csv.c_str(), builder_->numCenterlinePoints(), cfg.num_obs, control_hz_);

    obs_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(obs_topic, 10);

    pose_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      pose_topic, 10,
      [this](nav_msgs::msg::Odometry::SharedPtr msg) {this->onPose(*msg);});
    twist_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      twist_topic, 10,
      [this](nav_msgs::msg::Odometry::SharedPtr msg) {this->onTwist(*msg);});
    action_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      action_topic, 10,
      [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->onAction(*msg);});
    if (enable_opponent_obs_ && !zero_opponent_obs_) {
      opp_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        opp_odom_topic, 10,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {this->onOpponent(*msg);});
    }
    if (enable_slip_estimation_) {
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::SharedPtr msg) {this->onImu(*msg);});
      RCLCPP_INFO(
        get_logger(),
        "slip estimation ON (imu='%s', accel x%.3f, gyro x%.5f, use_imu_yaw=%d)",
        imu_topic.c_str(), imu_accel_to_ms2_, imu_gyro_to_rads_,
        static_cast<int>(imu_use_for_yaw_rate_));
    }

    const double period = (control_hz_ > 0.0) ? 1.0 / control_hz_ : 0.1;
    timer_ = create_wall_timer(
      std::chrono::duration<double>(period),
      [this]() {this->onTimer();});
  }

private:
  void onPose(const nav_msgs::msg::Odometry & msg)
  {
    pos_x_ = msg.pose.pose.position.x;
    pos_y_ = msg.pose.pose.position.y;
    const auto & q = msg.pose.pose.orientation;
    yaw_ = quatToYaw(q.x, q.y, q.z, q.w);

    // Differentiate PF pose to recover a drift-free (but noisy) ground velocity in
    // the body frame. Used only by the slip estimator; the policy's body velocity
    // obs still comes from the VESC twist.
    if (enable_slip_estimation_) {
      const rclcpp::Time stamp(msg.header.stamp, RCL_ROS_TIME);
      if (have_prev_pf_) {
        const double dt = (stamp - prev_pf_stamp_).seconds();
        if (dt > 1e-3 && dt < 0.5) {
          const double vwx = (pos_x_ - prev_pf_x_) / dt;
          const double vwy = (pos_y_ - prev_pf_y_) / dt;
          const double c = std::cos(yaw_);
          const double s = std::sin(yaw_);
          pf_vx_body_ = c * vwx + s * vwy;
          pf_vy_body_ = -s * vwx + c * vwy;
          have_pf_vel_ = true;
        }
      }
      prev_pf_x_ = pos_x_;
      prev_pf_y_ = pos_y_;
      prev_pf_stamp_ = stamp;
      have_prev_pf_ = true;
    }
    have_pose_ = true;
  }

  void onImu(const sensor_msgs::msg::Imu & msg)
  {
    imu_ay_ = imu_ay_sign_ * imu_accel_to_ms2_ * msg.linear_acceleration.y;
    imu_yaw_rate_ = imu_yaw_rate_sign_ * imu_gyro_to_rads_ * msg.angular_velocity.z;
    have_imu_ = true;
  }

  void onTwist(const nav_msgs::msg::Odometry & msg)
  {
    twist_vx_ = msg.twist.twist.linear.x;
    twist_vy_ = msg.twist.twist.linear.y;
    wz_ = msg.twist.twist.angular.z;
    have_twist_ = true;
  }

  void onAction(const std_msgs::msg::Float32MultiArray & msg)
  {
    if (msg.data.size() >= 2) {
      last_throttle_ = msg.data[0];
      last_steer_ = msg.data[1];
    }
  }

  void onOpponent(const nav_msgs::msg::Odometry & msg)
  {
    opp_x_ = msg.pose.pose.position.x;
    opp_y_ = msg.pose.pose.position.y;
    opp_vx_ = msg.twist.twist.linear.x;  // world frame
    opp_vy_ = msg.twist.twist.linear.y;
    opp_stamp_ = now();
    have_opp_ = true;
  }

  // Estimate the 8-dim slip block [slip_ratio x4, slip_angle x4], wheel order
  // [LR, RR, LF, RF], from IMU + PF + VESC wheel speed. 'vesc_vx' is the body
  // forward velocity from the VESC twist (== driven wheel speed). dt is the
  // control period. Falls back to slip_obs_mean below slip_speed_min.
  std::array<double, 8> estimateSlipBlock(double vesc_vx, double dt)
  {
    std::array<double, 8> fallback;
    for (int i = 0; i < 8; ++i) {fallback[i] = slip_obs_mean_[i];}

    // Yaw rate: IMU gyro (preferred) or kinematic VESC odom.
    const double r = (imu_use_for_yaw_rate_ && have_imu_) ? imu_yaw_rate_ : wz_;

    // Longitudinal ground speed: low-pass the PF estimate; fall back to VESC.
    const double vx_meas = have_pf_vel_ ? pf_vx_body_ : vesc_vx;
    vx_ground_ = vx_ground_lp_alpha_ * vx_ground_ + (1.0 - vx_ground_lp_alpha_) * vx_meas;

    // Body lateral velocity via complementary filter: integrate (ay - r*vx) and
    // pull toward the (drift-free) PF lateral velocity.
    const double alpha = vy_filter_tau_s_ / (vy_filter_tau_s_ + dt);
    const double ay = have_imu_ ? imu_ay_ : 0.0;
    double vy_pred = vy_ground_ + dt * (ay - r * vx_ground_);
    const double vy_pf = have_pf_vel_ ? pf_vy_body_ : 0.0;
    vy_ground_ = alpha * vy_pred + (1.0 - alpha) * vy_pf;

    const double speed = std::hypot(vx_ground_, vy_ground_);
    if (speed < slip_speed_min_) {
      return fallback;
    }

    // Per-wheel ground velocity (body frame), then rotate fronts by steer angle.
    const double delta = std::max(-1.0, std::min(1.0, last_steer_)) * max_steer_;
    const double xs[4] = {-lr_, -lr_, lf_, lf_};       // LR, RR, LF, RF
    const double ys[4] = {half_track_, -half_track_, half_track_, -half_track_};
    std::array<double, 4> v_fwd, v_lat, spin;
    for (int i = 0; i < 4; ++i) {
      const double vix = vx_ground_ - r * ys[i];
      const double viy = vy_ground_ + r * xs[i];
      const double d = (i >= 2) ? delta : 0.0;  // only front wheels steer
      v_fwd[i] = std::cos(d) * vix + std::sin(d) * viy;
      v_lat[i] = -std::sin(d) * vix + std::cos(d) * viy;
      // Rear wheels are driven: wheel speed == VESC speed. Front wheels are
      // undriven (free-rolling): wheel speed tracks ground -> slip_ratio ~ 0,
      // overwritten with the training mean below.
      spin[i] = ((i < 2) ? vesc_vx : v_fwd[i]) / std::max(wheel_radius_, 1e-6);
    }
    // Rear wheels carry drive/brake torque (active) when throttle is applied;
    // fronts are free-rolling (passive) and their slip ratio is pinned below.
    const bool drive_active = std::abs(last_throttle_) > 1e-3;
    const std::array<bool, 4> active = {drive_active, drive_active, false, false};
    std::array<double, 8> slip = computeTyreSlip(
      v_fwd, v_lat, spin, wheel_radius_, active,
      slip_min_lat_, slip_min_active_long_, slip_min_passive_long_);
    // Front slip-ratio channels are a sim free-wheel artifact; pin to training
    // mean so they normalize to ~0 rather than a large OOD value.
    slip[2] = slip_obs_mean_[2];
    slip[3] = slip_obs_mean_[3];
    return slip;
  }

  void onTimer()
  {
    if (!have_pose_ || !have_twist_) {
      return;
    }

    double vx = twist_vx_;
    double vy = twist_vy_;
    if (twist_in_world_frame_) {
      double c = std::cos(yaw_);
      double s = std::sin(yaw_);
      double bx = c * vx + s * vy;
      double by = -s * vx + c * vy;
      vx = bx;
      vy = by;
    }

    // Fixed-step body acceleration at the control rate (not per odom callback).
    const double dt = (control_hz_ > 0.0) ? 1.0 / control_hz_ : 0.1;
    double ax = 0.0;
    double ay = 0.0;
    if (have_prev_vel_) {
      ax = (vx - prev_vx_) / dt;
      ay = (vy - prev_vy_) / dt;
    }
    prev_vx_ = vx;
    prev_vy_ = vy;
    have_prev_vel_ = true;

    VehicleState st;
    st.pos_x = pos_x_;
    st.pos_y = pos_y_;
    st.yaw = yaw_;
    st.vx = vx;
    st.vy = vy;
    st.wz = wz_;
    st.ax = ax;
    st.ay = ay;
    st.last_throttle = last_throttle_;
    st.last_steer = last_steer_;
    // tyre_slip defaults to zeros; estimate it from IMU + PF when enabled.
    if (enable_slip_estimation_) {
      st.tyre_slip = estimateSlipBlock(vx, dt);
    }
    // tyre_load defaults to the static ratio (1.0); estimate the quasi-static load
    // transfer from body accel when enabled (mirrors observation_builder_node).
    if (enable_load_estimation_) {
      st.tyre_load = computeQuasiStaticLoad(
        ax, ay, cg_height_, lf_, lr_, track_width_, roll_stiffness_front_);
    }

    OpponentState opp;
    bool opp_confident = false;
    if (enable_opponent_obs_ && have_opp_) {
      const double age = (now() - opp_stamp_).seconds();
      if (age <= opponent_timeout_s_) {
        opp_confident = true;
        opp.present = true;
        opp.pos_x = opp_x_;
        opp.pos_y = opp_y_;
        opp.vx = opp_vx_;
        opp.vy = opp_vy_;
      }
    }

    std::vector<float> obs = builder_->build(st, opp);
    // Mask seam: zero the opponent block when detection is not confident. Future
    // work will drive this from an LSTM certainty threshold instead of timeout.
    if (enable_opponent_obs_ && (zero_opponent_obs_ || !opp_confident)) {
      const int opp_base = kBaseObsDim;
      const int opp_dim = opponent_obs_dim_;
      for (int i = 0; i < opp_dim && opp_base + i < static_cast<int>(obs.size()); ++i) {
        obs[opp_base + i] = 0.0f;
      }
    }
    std_msgs::msg::Float32MultiArray out;
    out.data = std::move(obs);
    obs_pub_->publish(out);
  }

  std::unique_ptr<TrackObservationBuilder> builder_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obs_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr twist_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr action_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr opp_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  double control_hz_ = 10.0;
  bool twist_in_world_frame_ = false;

  bool enable_opponent_obs_ = false;
  bool zero_opponent_obs_ = false;
  int opponent_obs_dim_ = 6;
  double opponent_timeout_s_ = 0.5;
  bool have_opp_ = false;
  double opp_x_ = 0.0, opp_y_ = 0.0, opp_vx_ = 0.0, opp_vy_ = 0.0;
  rclcpp::Time opp_stamp_{0, 0, RCL_ROS_TIME};

  bool have_pose_ = false;
  bool have_twist_ = false;
  double pos_x_ = 0.0, pos_y_ = 0.0, yaw_ = 0.0;
  double twist_vx_ = 0.0, twist_vy_ = 0.0, wz_ = 0.0;
  double last_throttle_ = 0.0, last_steer_ = 0.0;

  bool have_prev_vel_ = false;
  double prev_vx_ = 0.0, prev_vy_ = 0.0;

  // --- tyre-slip estimation state -------------------------------------------
  bool enable_slip_estimation_ = false;
  double imu_accel_to_ms2_ = 9.80665, imu_gyro_to_rads_ = M_PI / 180.0;
  double imu_ay_sign_ = 1.0, imu_yaw_rate_sign_ = 1.0;
  bool imu_use_for_yaw_rate_ = true;
  double wheel_radius_ = 0.05, lf_ = 0.1773, lr_ = 0.1477;
  double track_width_ = 0.20, half_track_ = 0.10;
  double max_steer_ = 0.33;

  // --- quasi-static tyre-load estimation state ------------------------------
  bool enable_load_estimation_ = false;
  double cg_height_ = 0.05, roll_stiffness_front_ = 0.5;
  double slip_min_lat_ = 0.2, slip_min_active_long_ = 0.1, slip_min_passive_long_ = 0.4;
  double vy_filter_tau_s_ = 0.5, vx_ground_lp_alpha_ = 0.5, slip_speed_min_ = 0.3;
  std::vector<double> slip_obs_mean_;

  bool have_imu_ = false;
  double imu_ay_ = 0.0, imu_yaw_rate_ = 0.0;

  bool have_prev_pf_ = false, have_pf_vel_ = false;
  double prev_pf_x_ = 0.0, prev_pf_y_ = 0.0;
  rclcpp::Time prev_pf_stamp_{0, 0, RCL_ROS_TIME};
  double pf_vx_body_ = 0.0, pf_vy_body_ = 0.0;
  double vx_ground_ = 0.0, vy_ground_ = 0.0;
};

}  // namespace f1tenth_rl_vehicle

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<f1tenth_rl_vehicle::VehicleObsNode>());
  rclcpp::shutdown();
  return 0;
}

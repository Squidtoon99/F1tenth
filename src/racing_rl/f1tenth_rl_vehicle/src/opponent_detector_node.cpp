// opponent_detector_node: LiDAR-based 1v1 opponent detection on the real car.
//
// Subscribes to the raw LiDAR scan and the map-frame ego pose (particle filter),
// transforms each beam into the map frame, clusters returns, gates them to the
// drivable corridor (centerline + width CSV) to reject walls, associates the best
// candidate across frames, and estimates a smoothed world-frame velocity. The
// confirmed opponent is published as nav_msgs/Odometry on /rl/opponent/odom for
// vehicle_obs to fold into the 6-dim opponent block of the 390-dim observation.
//
// Confirmation is persistence-based, so a stationary opponent is still reported
// (with velocity ~ 0). A static laser-mount offset (lidar_offset_*) is used instead
// of tf2 to keep the on-car build dependency-light.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "f1tenth_rl_vehicle/rl_obs_core.hpp"

namespace f1tenth_rl_vehicle
{

namespace
{
double quatToYaw(double x, double y, double z, double w)
{
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}
}  // namespace

class OpponentDetectorNode : public rclcpp::Node
{
public:
  OpponentDetectorNode()
  : rclcpp::Node("opponent_detector")
  {
    const std::string track_csv = declare_parameter<std::string>("track_csv", "");
    const std::string scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
    const std::string pose_topic =
      declare_parameter<std::string>("pose_topic", "/pf/pose/odom");
    opp_odom_topic_ =
      declare_parameter<std::string>("opponent_odom_topic", "/rl/opponent/odom");
    const std::string marker_topic =
      declare_parameter<std::string>("marker_topic", "/rl/opponent/marker");
    publish_marker_ = declare_parameter<bool>("publish_marker", true);
    frame_id_ = declare_parameter<std::string>("frame_id", "map");

    // Static laser pose in the ego base_link frame (avoids a tf2 dependency).
    lidar_offset_x_ = declare_parameter<double>("lidar_offset_x", 0.0);
    lidar_offset_y_ = declare_parameter<double>("lidar_offset_y", 0.0);
    lidar_offset_yaw_ = declare_parameter<double>("lidar_offset_yaw", 0.0);

    DetectorConfig dcfg;
    dcfg.cluster_gap_m = declare_parameter<double>("cluster_gap_m", dcfg.cluster_gap_m);
    dcfg.min_opponent_size_m =
      declare_parameter<double>("min_opponent_size_m", dcfg.min_opponent_size_m);
    dcfg.max_opponent_size_m =
      declare_parameter<double>("max_opponent_size_m", dcfg.max_opponent_size_m);
    dcfg.boundary_margin_m =
      declare_parameter<double>("boundary_margin_m", dcfg.boundary_margin_m);
    dcfg.foreground_jump_m =
      declare_parameter<double>("foreground_jump_m", dcfg.foreground_jump_m);
    dcfg.max_range_m = declare_parameter<double>("max_range_m", dcfg.max_range_m);
    dcfg.min_cluster_beams =
      static_cast<int>(declare_parameter<int>("min_cluster_beams", dcfg.min_cluster_beams));
    dcfg.min_ego_dist_m = declare_parameter<double>("min_ego_dist_m", dcfg.min_ego_dist_m);
    dcfg.min_extent_m = declare_parameter<double>("min_extent_m", dcfg.min_extent_m);
    dcfg.min_visible_angle_deg =
      declare_parameter<double>("min_visible_angle_deg", dcfg.min_visible_angle_deg);
    dcfg.max_assoc_dist_m =
      declare_parameter<double>("max_assoc_dist_m", dcfg.max_assoc_dist_m);
    dcfg.vel_alpha = declare_parameter<double>("vel_alpha", dcfg.vel_alpha);
    dcfg.min_track_age_frames =
      static_cast<int>(declare_parameter<int>("min_track_age_frames", dcfg.min_track_age_frames));
    dcfg.min_speed_mps = declare_parameter<double>("min_speed_mps", dcfg.min_speed_mps);

    const int coarse_stride = static_cast<int>(declare_parameter<int>("coarse_stride", 10));

    if (track_csv.empty()) {
      throw std::runtime_error("opponent_detector: 'track_csv' parameter is required");
    }
    TrackData track = loadTrackCsv(track_csv);
    ObsConfig ocfg;  // geometry only; obs assembly params are irrelevant here.
    builder_ = std::make_unique<TrackObservationBuilder>(
      track.x, track.y, track.w_left, track.w_right, ocfg, coarse_stride);
    detector_ = std::make_unique<OpponentDetector>(*builder_, dcfg);
    RCLCPP_INFO(
      get_logger(), "opponent_detector ready (track %d pts); scan='%s' pose='%s' -> '%s'",
      builder_->numCenterlinePoints(), scan_topic.c_str(), pose_topic.c_str(),
      opp_odom_topic_.c_str());

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(opp_odom_topic_, 10);
    if (publish_marker_) {
      marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(marker_topic, 1);
    }

    pose_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      pose_topic, 10,
      [this](nav_msgs::msg::Odometry::SharedPtr msg) {this->onPose(*msg);});
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {this->onScan(*msg);});
  }

private:
  void onPose(const nav_msgs::msg::Odometry & msg)
  {
    ego_x_ = msg.pose.pose.position.x;
    ego_y_ = msg.pose.pose.position.y;
    const auto & q = msg.pose.pose.orientation;
    ego_yaw_ = quatToYaw(q.x, q.y, q.z, q.w);
    have_pose_ = true;
  }

  void onScan(const sensor_msgs::msg::LaserScan & scan)
  {
    if (!have_pose_) {
      return;
    }

    const double c_ego = std::cos(ego_yaw_);
    const double s_ego = std::sin(ego_yaw_);
    const double c_off = std::cos(lidar_offset_yaw_);
    const double s_off = std::sin(lidar_offset_yaw_);

    const size_t n = scan.ranges.size();
    std::vector<ScanPoint> beams(n);
    for (size_t i = 0; i < n; ++i) {
      const double r = static_cast<double>(scan.ranges[i]);
      ScanPoint pt;
      const bool valid = std::isfinite(r) && r >= scan.range_min && r <= scan.range_max;
      if (valid) {
        const double ang = scan.angle_min + static_cast<double>(i) * scan.angle_increment;
        const double lx = r * std::cos(ang);
        const double ly = r * std::sin(ang);
        // laser -> base_link (static offset)
        const double bx = lidar_offset_x_ + c_off * lx - s_off * ly;
        const double by = lidar_offset_y_ + s_off * lx + c_off * ly;
        // base_link -> map (ego pose)
        pt.x = ego_x_ + c_ego * bx - s_ego * by;
        pt.y = ego_y_ + s_ego * bx + c_ego * by;
        pt.valid = true;
      }
      beams[i] = pt;
    }

    const double stamp =
      static_cast<double>(scan.header.stamp.sec) + scan.header.stamp.nanosec * 1e-9;
    const DetectionResult det = detector_->update(
      beams, ego_x_, ego_y_, stamp, scan.angle_increment);
    if (!det.present) {
      return;  // staleness handled by the consumer's timeout.
    }

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = scan.header.stamp;
    odom.header.frame_id = frame_id_;
    odom.pose.pose.position.x = det.pos_x;
    odom.pose.pose.position.y = det.pos_y;
    odom.pose.pose.orientation.w = 1.0;
    odom.twist.twist.linear.x = det.vx;  // world frame
    odom.twist.twist.linear.y = det.vy;
    odom_pub_->publish(odom);

    if (publish_marker_ && marker_pub_) {
      visualization_msgs::msg::Marker m;
      m.header = odom.header;
      m.ns = "opponent";
      m.id = 0;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = odom.pose.pose;
      m.scale.x = 0.4;
      m.scale.y = 0.4;
      m.scale.z = 0.4;
      m.color.r = 1.0;
      m.color.g = 0.1;
      m.color.b = 0.1;
      m.color.a = 0.9;
      marker_pub_->publish(m);
    }
  }

  std::unique_ptr<TrackObservationBuilder> builder_;
  std::unique_ptr<OpponentDetector> detector_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

  std::string opp_odom_topic_;
  std::string frame_id_;
  bool publish_marker_ = true;

  double lidar_offset_x_ = 0.0;
  double lidar_offset_y_ = 0.0;
  double lidar_offset_yaw_ = 0.0;

  bool have_pose_ = false;
  double ego_x_ = 0.0, ego_y_ = 0.0, ego_yaw_ = 0.0;
};

}  // namespace f1tenth_rl_vehicle

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<f1tenth_rl_vehicle::OpponentDetectorNode>());
  rclcpp::shutdown();
  return 0;
}

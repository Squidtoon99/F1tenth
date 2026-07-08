# Normalized policy action (throttle, steer) in [-1, 1], before mapping to a
# concrete AckermannDriveStamped command by the drive layer.
#
# Scaffold placeholder — extend as needed.
std_msgs/Header header
float32 throttle
float32 steer

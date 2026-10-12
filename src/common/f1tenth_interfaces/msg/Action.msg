# Normalized policy action (throttle, steer) in [-1, 1], before mapping to a
# concrete AckermannDriveStamped command by the drive layer.
std_msgs/Header header
float32 throttle
float32 steer

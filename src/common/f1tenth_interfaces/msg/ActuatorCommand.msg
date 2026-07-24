# Atomic desired or applied physical actuator command for the RL current gate.
# Same message type on /rl/actuator/desired and /rl/actuator/applied.
#
# header.stamp — command generation/publication time.
# generation — monotonic sequence assigned by the publisher for freshness checks.
# observation_stamp — originating sensor-policy observation time (zero when unused).
#
# drive_current_a / brake_current_a — motor currents in amperes (mutually exclusive).
# servo_position — calibrated VESC servo command.
# longitudinal / steering — normalized effort in [-1, 1] for policy history reconstruction.
#
# source — post-arbitration command origin on applied feedback; desired RL commands
# use SOURCE_RL until the gate overwrites source and physical fields.

std_msgs/Header header
uint64 generation
builtin_interfaces/Time observation_stamp

float64 drive_current_a
float64 brake_current_a
float64 servo_position

float32 longitudinal
float32 steering

uint8 SOURCE_SAFE=0
uint8 SOURCE_RL=1
uint8 SOURCE_TELEOP=2
uint8 SOURCE_SAFETY=3

uint8 source

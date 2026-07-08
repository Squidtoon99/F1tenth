# Flat observation vector published to the policy/inference layer.
# The precise layout (indices, dimensions) is defined once in
# libs/f1tenth_contract and mirrored in C++ in f1tenth_common. This message is
# intentionally a generic float vector so the contract can evolve without
# regenerating interfaces.
#
# Scaffold placeholder — extend as needed.
std_msgs/Header header
float32[] data

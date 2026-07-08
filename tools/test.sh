#!/usr/bin/env bash
# colcon test wrapper. Runs unit tests + ament lint for OUR packages, but skips
# the vendored upstream packages (vesc, ackermann_mux, f1tenth_stack) so their
# lint/copyright style does not fail our build. We still BUILD them (tools/build.sh)
# because our packages depend on them; we just do not test/lint vendored code.
#
# Usage:
#   tools/test.sh                       # test everything except vendored
#   tools/test.sh --packages-select f1tenth_rl_agent   # pass extra colcon args
#
# Run inside the dev container (ROS 2 sourced).
# No `-e`: we want colcon test-result to run and report even when a test fails.
# No `-u`: the ROS setup.bash dereferences unset vars and would abort under it.
set -o pipefail
cd "$(dirname "$0")/.."   # repo root

# Vendored upstream packages excluded from testing/linting. Keep in sync with
# src/vehicle/VENDORED.md.
VENDORED=(
  vesc
  vesc_driver
  vesc_msgs
  vesc_ackermann
  ackermann_mux
  f1tenth_stack
)

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" 2>/dev/null || \
  echo "warning: could not source ROS; are you inside the dev container?" >&2

colcon test --packages-skip "${VENDORED[@]}" "$@"
status=$?
# Always print the result breakdown, then propagate the test status.
colcon test-result --verbose || true
exit "${status}"

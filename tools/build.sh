#!/usr/bin/env bash
# colcon build wrapper. Builds the whole workspace, or just one module group.
#
# Usage:
#   tools/build.sh                 # build everything (src/ + libs/)
#   tools/build.sh -t control      # build only packages under src/control
#   tools/build.sh -t racing_rl -- --event-handlers console_direct+   # pass colcon args
#
# Run inside the dev container (ROS 2 sourced). Uses --symlink-install so Python
# edits need no rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

GROUP=""
usage() {
  echo "Usage: $0 [-t <group>] [-- <extra colcon args>]"
  echo "  <group>: a directory name under src/ (e.g. control, mapping, racing_rl) or 'libs'"
}

while getopts ":t:h" opt; do
  case "${opt}" in
    t) GROUP="${OPTARG}" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" 2>/dev/null || \
  echo "warning: could not source ROS; are you inside the dev container?" >&2

if [ -n "${GROUP}" ]; then
  BASE="src/${GROUP}"
  [ "${GROUP}" = "libs" ] && BASE="libs"
  if [ ! -d "${BASE}" ]; then
    echo "No such group directory: ${BASE}" >&2
    exit 1
  fi
  mapfile -t PKGS < <(colcon list --names-only --base-paths "${BASE}")
  if [ "${#PKGS[@]}" -eq 0 ]; then
    echo "No packages found under ${BASE}" >&2
    exit 1
  fi
  echo "==> Building group '${GROUP}': ${PKGS[*]}"
  colcon build --symlink-install --packages-up-to "${PKGS[@]}" "$@"
else
  echo "==> Building the whole workspace"
  colcon build --symlink-install "$@"
fi

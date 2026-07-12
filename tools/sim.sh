#!/usr/bin/env bash
# Run + certify the deployed RL stack against the f1tenth gym, two-container style:
#
#   [novnc]  X server + web VNC  ->  http://localhost:8080/vnc.html
#   [sim]    external gym bridge (fork image)  ->  /scan, /ego_racecar/odom, /drive
#   [agent]  our dev image + live source        ->  observation -> policy -> /drive
#
# The two ROS containers share a docker bridge network + ROS_DOMAIN_ID so DDS
# discovery connects them. Foxglove: connect Studio to ws://localhost:8765.
#
# Commands:
#   tools/sim.sh up        # import sources, build, launch the agent graph
#   tools/sim.sh validate  # run the closed-loop acceptance gate (in the agent container)
#   tools/sim.sh logs      # follow sim bridge logs
#   tools/sim.sh down      # stop everything
#
# Env:
#   CHECKPOINT_DIR  host dir holding the trained .pt (mounted read-only at /policies)
#   CKPT            checkpoint filename inside that dir (default: policy.pt)
#   STACK           agent graph: "vehicle" (on-car C++, default)
#   MODE            "dev" (live source, default) or "release" (built runtime image)
#   ROS_DOMAIN_ID   DDS domain shared by both containers (default: 42)
#
# The vehicle stack (default) is the release gate: it runs the exact on-car C++
# autonomy graph (vehicle_obs -> policy_inference -> drive) against the gym's
# ground-truth odom, with the 6-dim opponent block zeroed for solo racing.
set -eo pipefail
cd "$(dirname "$0")/.."   # repo root

COMPOSE_FILE="deploy/docker/docker-compose.sim.yml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
CKPT="${CKPT:-policy.pt}"
STACK="${STACK:-vehicle}"
MODE="${MODE:-dev}"

# Observation dim the running graph must emit: solo racing uses the 390-dim 1v1
# layout with the opponent block zeroed, so the validator expects 390.
EXPECT_OBS_DIM="${EXPECT_OBS_DIM:-390}"
SPEED_LIMIT_MPS="${SPEED_LIMIT_MPS:-2.0}"

compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

die() { echo "ERROR: $*" >&2; exit 1; }

check_stack() {
  case "${STACK}" in
    vehicle) : ;;
    *) die "STACK must be 'vehicle' (got '${STACK}')" ;;
  esac
}

check_checkpoint() {
  local path="${CHECKPOINT_DIR:-}/${CKPT}"
  [ -n "${CHECKPOINT_DIR:-}" ] || die "CHECKPOINT_DIR is unset; point it at the dir holding ${CKPT}"
  [ -f "${path}" ] || die "checkpoint '${path}' not found (set CHECKPOINT_DIR and CKPT)"
}

verify_track_assets() {
  # The gym bridge and the agent package each ship the IV_2026 assets; they must
  # match or the policy sees a different track than it is evaluated on.
  local gym="sim/f1tenth_gym_ros/maps/IV_2026_SIM_centerline.csv"
  local agent="src/racing_rl/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv"
  if [ -f "${gym}" ] && [ -f "${agent}" ]; then
    cmp -s "${gym}" "${agent}" || die "IV_2026 centerline differs between gym (${gym}) and agent (${agent})"
    echo "==> IV_2026 centerline matches across gym and agent assets"
  else
    echo "WARNING: could not verify IV_2026 assets (missing ${gym} or ${agent})" >&2
  fi
}

import_sim_sources() {
  if [ -d sim/f1tenth_gym_ros/.git ] && [ -d sim/f1tenth_gym_ros/f1tenth_gym/.git ]; then
    echo "==> sim sources already imported (sim/f1tenth_gym_ros)"
    return
  fi
  echo "==> Importing gym bridge + engine (vcs import sim < sim/f1tenth_gym_ros.repos)"
  # vcstool lives in the base image; run it there so the host needs nothing extra.
  docker run --rm -v "${PWD}:/ws" -w /ws f1tenth-base:latest \
    bash -lc "vcs import sim < sim/f1tenth_gym_ros.repos"
}

agent_launch_cmd() {
  # The command the agent container runs to bring up the selected graph. Both
  # graphs read the policy at /policies/${CKPT} and the IV_2026 track.
  echo "ros2 launch f1tenth_bringup sim.launch.py \
    stack:=${STACK} \
    checkpoint_path:=/policies/${CKPT}"
}

check_mode() {
  case "${MODE}" in
    dev|release) : ;;
    *) die "MODE must be 'dev' or 'release' (got '${MODE}')" ;;
  esac
}

up() {
  check_stack
  check_mode
  check_checkpoint
  import_sim_sources
  verify_track_assets
  echo "==> Starting novnc + gym bridge (ROS_DOMAIN_ID=${ROS_DOMAIN_ID})"
  compose up -d novnc sim
  if [ "${MODE}" = "release" ]; then
    echo "==> Launching ${STACK} graph from the generic runtime image (/config + /policies)"
    # Override the image CMD (race.launch.py) so no hardware drivers start in sim.
    compose run --rm --service-ports runtime \
      ros2 launch f1tenth_bringup sim.launch.py \
      stack:="${STACK}" checkpoint_path:="/policies/${CKPT}"
    return
  fi
  echo "==> Building the agent workspace (racing_rl + control + bringup + common)"
  compose run --rm --entrypoint bash agent -lc "
    source /opt/ros/humble/setup.bash &&
    colcon build --symlink-install \
      --packages-up-to f1tenth_rl_agent f1tenth_rl_vehicle f1tenth_control f1tenth_bringup f1tenth_common"
  echo "==> Launching the ${STACK} agent graph (policy: /policies/${CKPT})"
  echo "    Foxglove: ws://localhost:8765   |   RViz over VNC: http://localhost:8080/vnc.html"
  compose run --rm --service-ports --entrypoint bash agent -lc "
    source /opt/ros/humble/setup.bash &&
    source install/setup.bash &&
    $(agent_launch_cmd)"
}

validate() {
  check_stack
  echo "==> Closed-loop acceptance gate (${STACK} stack, expect ${EXPECT_OBS_DIM}-dim obs)"
  compose run --rm --entrypoint bash agent -lc "
    source /opt/ros/humble/setup.bash &&
    source install/setup.bash &&
    python3 src/racing_rl/f1tenth_rl_agent/test/validate_closed_loop_gym.py \
      --expect-obs-dim ${EXPECT_OBS_DIM} \
      --speed-limit-mps ${SPEED_LIMIT_MPS} \
      ${VALIDATE_ARGS:-}"
}

CMD="${1:-up}"
case "${CMD}" in
  up)       up ;;
  validate) validate ;;
  down)     compose down ;;
  logs)     compose logs -f sim ;;
  *)
    echo "usage: tools/sim.sh [up|validate|logs|down]" >&2
    echo "  env: STACK=vehicle MODE=dev|release CHECKPOINT_DIR=/abs/dir CKPT=policy.pt" >&2
    exit 1
    ;;
esac

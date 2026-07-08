#!/usr/bin/env bash
# Run the deployed RL stack against the f1tenth gym, two-container style:
#
#   [novnc]  X server + web VNC  ->  http://localhost:8080/vnc.html
#   [sim]    external gym bridge (fork image)  ->  /scan, /ego_racecar/odom, /drive
#   [agent]  our dev image + live source        ->  observation -> policy -> /drive
#
# The two ROS containers share a docker bridge network + ROS_DOMAIN_ID so DDS
# discovery connects them. Foxglove: connect Studio to ws://localhost:8765.
#
# Usage:
#   CHECKPOINT_DIR=/abs/dir CKPT=policy.pt tools/sim.sh up   # import, build, run
#   tools/sim.sh down                                        # stop everything
#   tools/sim.sh logs                                        # follow sim logs
#
# Env:
#   CHECKPOINT_DIR  host dir holding the trained .pt (mounted read-only at /checkpoints)
#   CKPT            checkpoint filename inside that dir (default: policy.pt)
#   ROS_DOMAIN_ID   DDS domain shared by both containers (default: 42)
set -eo pipefail
cd "$(dirname "$0")/.."   # repo root

COMPOSE_FILE="deploy/docker/docker-compose.sim.yml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
CKPT="${CKPT:-policy.pt}"

compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

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

CMD="${1:-up}"
case "${CMD}" in
  up)
    import_sim_sources
    echo "==> Starting novnc + gym bridge (ROS_DOMAIN_ID=${ROS_DOMAIN_ID})"
    compose up -d novnc sim
    echo "==> Building the agent workspace (f1tenth_rl_agent + f1tenth_control)"
    compose run --rm --entrypoint bash agent -lc "
      source /opt/ros/humble/setup.bash &&
      colcon build --symlink-install \
        --packages-up-to f1tenth_rl_agent f1tenth_control"
    if [ ! -f "${CHECKPOINT_DIR:-/nonexistent}/${CKPT}" ]; then
      echo "WARNING: checkpoint '${CHECKPOINT_DIR:-<unset>}/${CKPT}' not found on host." >&2
      echo "         Set CHECKPOINT_DIR (and CKPT) so /checkpoints/${CKPT} exists." >&2
    fi
    echo "==> Launching the RL agent (checkpoint: /checkpoints/${CKPT})"
    echo "    Foxglove: ws://localhost:8765   |   RViz over VNC: http://localhost:8080/vnc.html"
    compose run --rm --service-ports --entrypoint bash agent -lc "
      source /opt/ros/humble/setup.bash &&
      source install/setup.bash &&
      ros2 launch f1tenth_rl_agent bringup_agent_launch.py \
        checkpoint_path:=/checkpoints/${CKPT}"
    ;;
  down)
    compose down
    ;;
  logs)
    compose logs -f sim
    ;;
  *)
    echo "usage: tools/sim.sh [up|down|logs]" >&2
    exit 1
    ;;
esac

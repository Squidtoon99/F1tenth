#!/usr/bin/env bash
# One-command dev environment. Builds the base + dev images and drops you into a
# shell with the live source mounted at /ws.
#
# Usage:
#   tools/dev.sh up        # build images (if needed) and enter a dev shell
#   tools/dev.sh build     # (re)build the base + dev images
#   tools/dev.sh shell     # enter a dev shell (assumes images exist)
#   tools/dev.sh down      # stop/remove the dev container
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

COMPOSE_FILE="deploy/docker/docker-compose.dev.yml"
ROS_DISTRO="${ROS_DISTRO:-humble}"

compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

build_base() {
  echo "==> Building base image (f1tenth-base:latest)"
  docker buildx build --load \
    -f deploy/docker/Dockerfile.base \
    --build-arg ROS_DISTRO="${ROS_DISTRO}" \
    -t f1tenth-base:latest .
}

CMD="${1:-up}"
case "${CMD}" in
  build)
    build_base
    compose build
    ;;
  up)
    build_base
    compose build
    echo "==> Entering dev shell (source mounted at /ws). Run ./tools/build.sh to build."
    compose run --rm dev bash
    ;;
  shell)
    compose run --rm dev bash
    ;;
  down)
    compose down
    ;;
  *)
    echo "usage: tools/dev.sh [up|build|shell|down]" >&2
    exit 1
    ;;
esac

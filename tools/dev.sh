#!/usr/bin/env bash
# One-command dev environment. Builds the base + dev images and drops you into a
# shell with the live source mounted at /ws.
#
# Usage:
#   tools/dev.sh up        # build images (if needed) and enter a dev shell
#   tools/dev.sh build     # (re)build the base + dev images
#   tools/dev.sh pull      # pull the prebuilt seed images from Docker Hub instead
#   tools/dev.sh shell     # enter a dev shell (assumes images exist)
#   tools/dev.sh down      # stop/remove the dev container
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

COMPOSE_FILE="deploy/docker/docker-compose.dev.yml"
ROS_DISTRO="${ROS_DISTRO:-humble}"
# Prebuilt seed images on Docker Hub (amd64). Teammates on amd64 can `pull` these
# instead of building locally. On Apple Silicon they run under emulation, so an
# arm64 host is usually better off building natively (`build`).
SEED_REGISTRY="${SEED_REGISTRY:-squidtoon99}"

compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

build_base() {
  echo "==> Building base image (f1tenth-base:latest)"
  docker buildx build --load \
    -f deploy/docker/Dockerfile.base \
    --build-arg ROS_DISTRO="${ROS_DISTRO}" \
    -t f1tenth-base:latest .
}

pull_seed() {
  echo "==> Pulling seed images from ${SEED_REGISTRY} (amd64)"
  docker pull "${SEED_REGISTRY}/f1tenth-base:latest"
  docker pull "${SEED_REGISTRY}/f1tenth-dev:latest"
  docker tag "${SEED_REGISTRY}/f1tenth-base:latest" f1tenth-base:latest
  docker tag "${SEED_REGISTRY}/f1tenth-dev:latest" f1tenth-dev:latest
  echo "==> Tagged as f1tenth-base:latest / f1tenth-dev:latest. Run: tools/dev.sh shell"
}

CMD="${1:-up}"
case "${CMD}" in
  build)
    build_base
    compose build
    ;;
  pull)
    pull_seed
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
    echo "usage: tools/dev.sh [up|build|pull|shell|down]" >&2
    exit 1
    ;;
esac

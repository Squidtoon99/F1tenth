#!/usr/bin/env bash
# One-time host setup for developing this repo. Checks prerequisites and enables
# cross-architecture Docker builds (so you can build the Jetson arm64 image on an
# x86 machine). Does not install Docker for you.
set -euo pipefail

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MISSING: $1 — please install it ($2)" >&2
    return 1
  fi
  echo "ok: $1"
}

echo "==> Checking prerequisites"
missing=0
need git "https://git-scm.com/" || missing=1
need docker "https://docs.docker.com/get-docker/" || missing=1
if ! docker buildx version >/dev/null 2>&1; then
  echo "MISSING: docker buildx — install/enable the Buildx plugin" >&2
  missing=1
else
  echo "ok: docker buildx"
fi

if [ "${missing}" -ne 0 ]; then
  echo "==> Resolve the missing prerequisites above, then re-run." >&2
  exit 1
fi

echo "==> Enabling cross-arch emulation (for building linux/arm64 on x86)"
docker run --privileged --rm tonistiigi/binfmt --install arm64 || \
  echo "note: could not register binfmt; only needed for cross-arch image builds."

echo "==> Creating a buildx builder (idempotent)"
docker buildx create --name f1tenth --use >/dev/null 2>&1 || docker buildx use f1tenth

echo "==> Done. Next: ./tools/dev.sh up"

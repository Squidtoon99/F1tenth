#!/usr/bin/env bash
# Container entrypoint for the generic car image.
# Sources ROS + the baked workspace, then applies the per-car overlay mounted at
# /config (identity + params + map). The overlay is never baked into the image, so
# one image serves every car.
set -euo pipefail

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "/ws/install/setup.bash"

if [ -f /config/car.yaml ]; then
  echo "[entrypoint] using per-car overlay: $(grep -m1 '^car_id' /config/car.yaml || echo '(car_id not set)')"
else
  echo "[entrypoint] WARNING: no /config/car.yaml mounted; running with generic defaults."
fi

# RL policy (if racing with the RL stack) is mounted read-only at /policies.
exec "$@"

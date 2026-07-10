#!/usr/bin/env bash
# Container entrypoint for the generic car image.
# Sources ROS + the baked workspace, then applies the per-car overlay mounted at
# /config (identity + params + map). The overlay is never baked into the image, so
# one image serves every car.
# No `-u`: ROS setup.bash dereferences unset vars (AMENT_TRACE_SETUP_FILES, ...) and
# would abort under it.
set -eo pipefail

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "/ws/install/setup.bash"

if [ -f /config/car.yaml ]; then
  echo "[entrypoint] using per-car overlay: $(grep -m1 '^car_id' /config/car.yaml || echo '(car_id not set)')"
else
  echo "[entrypoint] WARNING: no /config/car.yaml mounted; running with generic defaults."
fi

# The per-car ROS param overlay (/config/params.yaml) is consumed directly by the
# launch files (race.launch.py overlay_params_file / car.launch.py vesc_config),
# so it is node-scoped and needs no parsing here. Just surface its presence.
if [ -f /config/params.yaml ]; then
  echo "[entrypoint] per-car params overlay present: /config/params.yaml"
else
  echo "[entrypoint] WARNING: no /config/params.yaml; nodes use package defaults."
fi

# RL policy (if racing with the RL stack) is mounted read-only at /policies.
if [ -f /policies/policy.pt ]; then
  echo "[entrypoint] policy present: /policies/policy.pt"
else
  echo "[entrypoint] WARNING: no /policies/policy.pt mounted; the RL stack will not start."
fi

exec "$@"

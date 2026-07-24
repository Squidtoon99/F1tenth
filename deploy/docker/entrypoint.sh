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

if [ -f /config/maps/map.yaml ]; then
  echo "[entrypoint] map overlay present: /config/maps/map.yaml"
else
  echo "[entrypoint] WARNING: no /config/maps/map.yaml; localization needs a map."
fi
if [ -f /policies/policy.pt ]; then
  echo "[entrypoint] policy present: /policies/policy.pt"
else
  echo "[entrypoint] WARNING: no /policies/policy.pt mounted; the 390-D RL stack will not start."
fi
if [ -f /policies/sensor_policy.pt ] || ls /policies/sensor_policy*.pt >/dev/null 2>&1; then
  echo "[entrypoint] sensor-policy checkpoint present under /policies"
else
  echo "[entrypoint] NOTE: no sensor_policy*.pt under /policies (ok for race.launch)."
fi
if [ -f /etc/f1tenth/sensor_policy_pins.json ]; then
  echo "[entrypoint] sensor-policy pins: $(tr -d '\n' < /etc/f1tenth/sensor_policy_pins.json)"
fi

exec "$@"

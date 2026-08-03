#!/usr/bin/env bash
# Non-powered sensor-policy image smoke: ROS/Torch/CUDA, format-4 artifact load,
# executable inventory, and a driver/gate-free dry-run launch.
#
# Usage (inside the running image or via docker run):
#   smoke_sensor_policy.sh
#   CHECKPOINT_PATH=/policies/e2e_policy.pt REQUIRE_CUDA=1 smoke_sensor_policy.sh
#
# Env:
#   CHECKPOINT_PATH   format-4 checkpoint (default /policies/e2e_policy.pt)
#   REQUIRE_CUDA      1 = fail if CUDA unavailable (default 1 on arm64 Jetson)
#   LAUNCH_TIMEOUT_S  dry-run launch timeout (default 12)
set -eo pipefail

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source /ws/install/setup.bash

REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/policies/e2e_policy.pt}"
LAUNCH_TIMEOUT_S="${LAUNCH_TIMEOUT_S:-12}"

echo "==> Image pins"
if [ -f /etc/f1tenth/sensor_policy_pins.json ]; then
  cat /etc/f1tenth/sensor_policy_pins.json
else
  echo "warning: /etc/f1tenth/sensor_policy_pins.json missing" >&2
fi

echo "==> Torch / CUDA"
python3 - <<'PY'
import torch

print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

if [ "${REQUIRE_CUDA}" = "1" ]; then
  python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA required (REQUIRE_CUDA=1) but torch.cuda.is_available() is False", file=sys.stderr)
    sys.exit(1)
PY
else
  echo "    REQUIRE_CUDA=0 — skipping CUDA availability gate"
fi

echo "==> rclpy"
python3 -c "import rclpy; print('rclpy ok')"

if [ -f "${CHECKPOINT_PATH}" ]; then
  echo "==> Load format-4 artifact: ${CHECKPOINT_PATH}"
  python3 - <<PY
import torch
from f1tenth_rl_agent.policy_model import load_sensor_actor, load_sensor_obs_norm
from f1tenth_rl_agent import sensor_interfaces as si

path = "${CHECKPOINT_PATH}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = load_sensor_actor(path, "actor", device)
norm = load_sensor_obs_norm(path, device, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
obs = torch.zeros(1, si.NUM_OBS, device=device)
with torch.inference_mode():
    action, hidden = actor(obs, deterministic=True, with_logprob=False)
print(
    f"artifact_ok obs_dim={si.NUM_OBS} layout={si.ACTOR_LAYOUT_VERSION} "
    f"format={si.POLICY_FORMAT_VERSION} gru_hidden={si.GRU_HIDDEN_DIM} "
    f"action={tuple(float(x) for x in action.reshape(-1).tolist())}"
)
PY
else
  echo "==> No checkpoint at ${CHECKPOINT_PATH} — skipping artifact load"
fi

echo "==> Expected executables (sensor_policy.launch.py graph)"
EXPECTED=(
  "f1tenth_rl_agent sensor_racer"
  "f1tenth_control rl_current_gate"
  "f1tenth_control rl_deadman_gate"
  "f1tenth_control safety"
  "vesc_driver vesc_driver_node"
  "vesc_ackermann vesc_to_odom_node"
  "urg_node urg_node_driver"
  "joy joy_node"
  "joy_teleop joy_teleop"
)

missing=0
for spec in "${EXPECTED[@]}"; do
  pkg="${spec%% *}"
  exe="${spec#* }"
  prefix="$(ros2 pkg prefix "${pkg}" 2>/dev/null || true)"
  if [ -z "${prefix}" ]; then
    echo "MISSING package ${pkg} (for ${exe})" >&2
    missing=1
    continue
  fi
  path="${prefix}/lib/${pkg}/${exe}"
  if [ ! -x "${path}" ]; then
    echo "MISSING executable ${path}" >&2
    missing=1
  else
    echo "ok ${pkg}/${exe}"
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "executable inventory failed" >&2
  exit 1
fi

echo "==> Launch dry-run (drivers/gate/racer disabled, dry_run:=true)"
set +e
timeout "${LAUNCH_TIMEOUT_S}" ros2 launch f1tenth_bringup sensor_policy.launch.py \
  dry_run:=true \
  enable_drivers:=false \
  enable_gate:=false \
  enable_racer:=false \
  enable_safety:=false \
  device:=cuda
launch_rc=$?
set -e
if [ "${launch_rc}" -eq 124 ]; then
  echo "launch dry-run reached timeout (${LAUNCH_TIMEOUT_S}s) — treat as pass"
elif [ "${launch_rc}" -ne 0 ]; then
  echo "launch dry-run failed (rc=${launch_rc})" >&2
  exit "${launch_rc}"
else
  echo "launch dry-run exited cleanly"
fi

echo "==> smoke_sensor_policy.sh PASS"

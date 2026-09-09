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
#   STACK           agent graph: "vehicle" (on-car C++, default) or
#                   "sensor_policy" (1097-D sensor_racer + gym_sensor_bridge)
#   MODE            "dev" (live source, default) or "release" (built runtime image)
#   ROS_DOMAIN_ID   DDS domain shared by both containers (default: 42)
#
# The vehicle stack (default) is the release gate: it runs the exact on-car C++
# autonomy graph (vehicle_obs -> policy_inference -> drive) against the gym's
# ground-truth odom, with the 6-dim opponent block zeroed for solo racing.
# STACK=sensor_policy is a courtyard smoke loop, not the 3-lap IV_2026 gate.
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

compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

die() { echo "ERROR: $*" >&2; exit 1; }

check_stack() {
  case "${STACK}" in
    vehicle|sensor_policy) : ;;
    *) die "STACK must be 'vehicle' or 'sensor_policy' (got '${STACK}')" ;;
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
  else
    echo "==> Importing gym bridge + engine (vcs import sim < sim/f1tenth_gym_ros.repos)"
    # vcstool lives in the base image; run it there so the host needs nothing extra.
    docker run --rm -v "${PWD}:/ws" -w /ws f1tenth-base:latest \
      bash -lc "vcs import sim < sim/f1tenth_gym_ros.repos"
  fi
  # vcs import via docker leaves root-owned files; map wiring needs to write.
  if [ ! -w sim/f1tenth_gym_ros/maps ] || [ ! -w sim/f1tenth_gym_ros/config ]; then
    docker run --rm -v "${PWD}:/ws" -w /ws f1tenth-base:latest \
      bash -lc "chown -R $(id -u):$(id -g) sim/f1tenth_gym_ros"
  fi
}

wire_courtyard_gym() {
  local maps="sim/f1tenth_gym_ros/maps"
  local cfg="sim/f1tenth_gym_ros/config/sim.yaml"
  local assets="training/assets"
  [ -f "${cfg}" ] || die "gym sim.yaml missing (${cfg}); import gym sources first"
  [ -f "${assets}/courtyard_2.yaml" ] || die "missing ${assets}/courtyard_2.yaml"
  [ -f "${assets}/courtyard_2.png" ] || die "missing ${assets}/courtyard_2.png"
  [ -f "${assets}/courtyard_2_centerline.csv" ] || \
    die "missing ${assets}/courtyard_2_centerline.csv"
  cp -f "${assets}/courtyard_2.png" "${assets}/courtyard_2_centerline.csv" "${maps}/"
  # Gym TrackSpec rejects ROS map keys like mode:; keep occupancy fields only.
  python3 - "${assets}/courtyard_2.yaml" "${maps}/courtyard_2.yaml" <<'PY'
import sys

src, dst = sys.argv[1], sys.argv[2]
keep = ("image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh")
out = []
for line in open(src, encoding="utf-8"):
    key = line.split(":", 1)[0].strip()
    if key in keep:
        out.append(line if line.endswith("\n") else line + "\n")
open(dst, "w", encoding="utf-8").write("".join(out))
open(dst.replace("courtyard_2.yaml", "courtyard_2_map.yaml"), "w", encoding="utf-8").write(
    "".join(out)
)
PY
  python3 - "${cfg}" "${assets}/courtyard_2_centerline.csv" <<'PY'
import math
import re
import sys

cfg_path, cl_path = sys.argv[1], sys.argv[2]
rows = []
with open(cl_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("x_m"):
            continue
        parts = line.split(",")
        rows.append((float(parts[0]), float(parts[1])))
        if len(rows) == 2:
            break
if len(rows) < 2:
    raise SystemExit(f"{cl_path}: need two centerline rows for spawn heading")
sx, sy = rows[0]
stheta = math.atan2(rows[1][1] - rows[0][1], rows[1][0] - rows[0][0])
text = open(cfg_path, encoding="utf-8").read()
replacements = {
    "map_path": "'maps/courtyard_2'",
    "sx": f"{sx:.6f}",
    "sy": f"{sy:.6f}",
    "stheta": f"{stheta:.6f}",
    "async_mode": "False",
    "kb_teleop": "False",
    "scan_num_beams": "1081",
    "scan_range_min": "0.06",
    "scan_range_max": "30.0",
}
for key, value in replacements.items():
    text, n = re.subn(
        rf"^([ \t]*{key}:[ \t]*).*$",
        rf"\g<1>{value}",
        text,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"{cfg_path}: failed to set {key}")
open(cfg_path, "w", encoding="utf-8").write(text)
print(f"==> courtyard gym map courtyard_2 spawn=({sx:.3f}, {sy:.3f}, {stheta:.3f})")
PY
}

restore_vehicle_gym() {
  local cfg="sim/f1tenth_gym_ros/config/sim.yaml"
  if [ -d sim/f1tenth_gym_ros/.git ] && [ -f "${cfg}" ]; then
    git -C sim/f1tenth_gym_ros checkout -- config/sim.yaml
  fi
}

agent_launch_cmd() {
  # The command the agent container runs to bring up the selected graph.
  # vehicle reads IV_2026; sensor_policy uses the courtyard map wired below.
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
  if [ "${STACK}" = "sensor_policy" ]; then
    wire_courtyard_gym
  else
    restore_vehicle_gym
    verify_track_assets
  fi
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
  [ "${STACK}" = "vehicle" ] || \
    die "validate is vehicle-only (got STACK=${STACK}); do not use it for sensor_policy"
  echo "==> Closed-loop acceptance gate (${STACK} stack, expect ${EXPECT_OBS_DIM}-dim obs)"
  compose run --rm --entrypoint bash agent -lc "
    source /opt/ros/humble/setup.bash &&
    source install/setup.bash &&
    python3 src/racing_rl/f1tenth_rl_agent/test/validate_closed_loop_gym.py \
      --expect-obs-dim ${EXPECT_OBS_DIM} \
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
    echo "  env: STACK=vehicle|sensor_policy MODE=dev|release CHECKPOINT_DIR=/abs/dir CKPT=policy.pt" >&2
    exit 1
    ;;
esac

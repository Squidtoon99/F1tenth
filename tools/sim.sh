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
#   tools/sim.sh certify   # one-shot: up → score laps/crashes/lap time → down
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
# The vehicle stack is the legacy 392-D C++ gate (vehicle_obs -> policy_inference
# -> drive). STACK=sensor_policy is the 1,097-D on-car graph
# (sensor_racer -> rl_current_gate -> gym_sensor_bridge -> /drive) on courtyard_2.
set -eo pipefail
cd "$(dirname "$0")/.."   # repo root

COMPOSE_FILE="deploy/docker/docker-compose.sim.yml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# Keep gym inference off the training GPU (entropy-ab and overnight runs).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
CKPT="${CKPT:-policy.pt}"
STACK="${STACK:-vehicle}"
MODE="${MODE:-dev}"

# Observation dim: 392-D vehicle graph (legacy) or 1097-D sensor_racer.
EXPECT_OBS_DIM="${EXPECT_OBS_DIM:-}"

compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

die() { echo "ERROR: $*" >&2; exit 1; }

stop_stray_agents() {
  # compose run one-offs are not compose services; `compose down` leaves them
  # on ROS_DOMAIN_ID and they steal /drive + validate's docker exec target.
  docker rm -f f1tenth-gym-agent >/dev/null 2>&1 || true
  local ids=""
  ids="$(docker ps -aq --filter name=docker-agent-run 2>/dev/null || true)"
  if [ -n "${ids}" ]; then
    # shellcheck disable=SC2086
    docker rm -f ${ids} >/dev/null 2>&1 || true
  fi
}

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
    # Mount only sim/ so a git worktree's .git pointer (outside this tree) is not
    # visible inside the container.
    if [ -e sim/f1tenth_gym_ros ]; then
      docker run --rm -v "${PWD}/sim:/sim" -w /sim f1tenth-base:latest \
        bash -lc "rm -rf /sim/f1tenth_gym_ros"
    fi
    docker run --rm -v "${PWD}/sim:/sim" -w /sim f1tenth-base:latest \
      bash -lc "vcs import . < f1tenth_gym_ros.repos"
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
    "async_mode": "True",
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
  # 1081-beam lidar inside every physics step is ~400 ms and caps the graph
  # at ~2.5 Hz. Raycast only on the 20 Hz publish timer; keep 100 Hz ST.
  python3 - <<'PY'
from pathlib import Path

def apply(path, replacements):
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new, label in replacements:
        if new in text and old not in text:
            print(f"==> already patched: {label}")
            continue
        if old not in text:
            raise SystemExit(f"{path}: string not found for: {label}")
        text = text.replace(old, new, 1)
        print(f"==> {label}")
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")

bridge = Path("sim/f1tenth_gym_ros/f1tenth_gym_ros/gym_bridge.py")
sim = Path("sim/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym/envs/simulator.py")
apply(
    bridge,
    (
        (
            "self.timer = self.create_timer(0.004, self.timer_callback)",
            "self.timer = self.create_timer(0.05, self.timer_callback)",
            "gym lidar publish timer 250 Hz -> 20 Hz",
        ),
        (
            "self.drive_timer = self.create_timer(0.05, self.drive_timer_callback)",
            "self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)",
            "restore gym physics timer 100 Hz",
        ),
        (
            "timestep=0.05,",
            "timestep=0.01,",
            "restore gym timestep 0.01",
        ),
        (
            "            self._update_sim_state()\n\n        if not self.get_parameter('async_mode').value:",
            "            self._update_sim_state()\n"
            "        self.env.unwrapped.sim.scan_every_step = False\n\n"
            "        if not self.get_parameter('async_mode').value:",
            "defer gym lidar to 20 Hz publish timer",
        ),
        (
            "        if self.sim_paused:\n"
            "            return  # Skip stepping the sim if paused\n"
            "        \n"
            "        ts = self.get_clock().now().to_msg()",
            "        if self.sim_paused:\n"
            "            return  # Skip stepping the sim if paused\n"
            "        \n"
            "        sim = self.env.unwrapped.sim\n"
            "        if not getattr(sim, 'scan_every_step', True) and sim.scan_enabled:\n"
            "            sim._update_scans()\n"
            "            self._update_sim_state()\n"
            "        ts = self.get_clock().now().to_msg()",
            "raycast 1081-beam lidar only when publishing",
        ),
    ),
)
apply(
    sim,
    (
        (
            "        if self.scan_enabled:\n"
            "            self._update_scans()\n"
            "        else:\n"
            "            self.state.scans.fill(0.0)\n"
            "            self.state.collisions.fill(0.0)\n"
            "\n"
            "        self._update_agent_collisions()",
            "        if self.scan_enabled and getattr(self, \"scan_every_step\", True):\n"
            "            self._update_scans()\n"
            "        elif not self.scan_enabled:\n"
            "            self.state.scans.fill(0.0)\n"
            "            self.state.collisions.fill(0.0)\n"
            "\n"
            "        if getattr(self, \"scan_every_step\", True) or not self.scan_enabled:\n"
            "            self._update_agent_collisions()",
            "skip lidar raycast on physics steps",
        ),
    ),
)
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
    colcon build --symlink-install --base-paths src libs \
      --packages-up-to f1tenth_rl_agent f1tenth_rl_vehicle f1tenth_control \
        f1tenth_bringup f1tenth_common f1tenth_policy"
  echo "==> Launching the ${STACK} agent graph (policy: /policies/${CKPT})"
  echo "    Foxglove: ws://localhost:8765   |   RViz over VNC: http://localhost:8080/vnc.html"
  compose run --rm --service-ports --entrypoint bash agent -lc "
    source /opt/ros/humble/setup.bash &&
    source install/setup.bash &&
    $(agent_launch_cmd)"
}

validate() {
  check_stack
  local expect="${EXPECT_OBS_DIM}"
  local script="src/racing_rl/f1tenth_rl_agent/test/validate_closed_loop_gym.py"
  if [ "${STACK}" = "sensor_policy" ]; then
    expect="${expect:-1097}"
    script="src/racing_rl/f1tenth_rl_agent/test/validate_closed_loop_sensor_policy.py"
  else
    expect="${expect:-390}"
  fi
  echo "==> Closed-loop acceptance gate (${STACK} stack, expect ${expect}-dim obs)"
  # Prefer the live graph container so DDS stays on-box. A second compose-run
  # agent on the docker bridge only sees a few Hz of /desired.
  local agent_cid=""
  agent_cid="$(docker ps --filter name=f1tenth-gym-agent --filter status=running --format '{{.ID}}' | head -1)"
  if [ -z "${agent_cid}" ]; then
    agent_cid="$(docker ps --filter ancestor=f1tenth-dev:latest --filter status=running --format '{{.ID}}' | head -1)"
  fi
  if [ -n "${agent_cid}" ]; then
    echo "==> Running gate inside agent container ${agent_cid}"
    docker exec \
      -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
      -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" \
      "${agent_cid}" bash -lc "
        source /opt/ros/humble/setup.bash &&
        source /ws/install/setup.bash &&
        python3 ${script} --expect-obs-dim ${expect} ${VALIDATE_ARGS:-}"
    return
  fi
  compose run --rm -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --entrypoint bash agent -lc "
    source /opt/ros/humble/setup.bash &&
    source install/setup.bash &&
    python3 ${script} --expect-obs-dim ${expect} ${VALIDATE_ARGS:-}"
}

wait_sim_ready() {
  echo "==> Waiting for gym bridge"
  local cid=""
  local i
  for i in $(seq 1 90); do
    cid="$(compose ps -q sim 2>/dev/null || true)"
    if [ -n "${cid}" ] && docker logs "${cid}" 2>&1 | grep -q "Running in asynchronous mode"; then
      echo "==> Gym bridge ready"
      return 0
    fi
    sleep 2
  done
  die "gym bridge did not become ready"
}

wait_graph_ready() {
  local cid="$1"
  local needle="sensor_racer 1097-D"
  if [ "${STACK}" = "vehicle" ]; then
    needle="policy_inference"
  fi
  echo "==> Waiting for graph (${needle})"
  local i
  for i in $(seq 1 90); do
    if docker logs "${cid}" 2>&1 | grep -q "${needle}"; then
      echo "==> Graph ready"
      return 0
    fi
    sleep 2
  done
  die "agent graph did not become ready (${needle})"
}

certify() {
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
  echo "==> Certify ${STACK} ${CKPT} (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}')"
  compose down >/dev/null 2>&1 || true
  stop_stray_agents
  compose up -d novnc sim
  if [ "${MODE}" = "dev" ]; then
    echo "==> Building the agent workspace"
    compose run --rm --entrypoint bash agent -lc "
      source /opt/ros/humble/setup.bash &&
      colcon build --symlink-install --base-paths src libs \
        --packages-up-to f1tenth_rl_agent f1tenth_rl_vehicle f1tenth_control \
          f1tenth_bringup f1tenth_common f1tenth_policy"
    compose run -d --name f1tenth-gym-agent --service-ports --entrypoint bash agent -lc "
      source /opt/ros/humble/setup.bash &&
      source install/setup.bash &&
      $(agent_launch_cmd)"
  else
    compose run -d --name f1tenth-gym-agent --service-ports runtime \
      ros2 launch f1tenth_bringup sim.launch.py \
      stack:="${STACK}" checkpoint_path:="/policies/${CKPT}"
  fi
  # Start the agent while gym is still coming up so FastDDS meets the bridge
  # as a late-starting publisher, then wait for both before scoring.
  wait_graph_ready f1tenth-gym-agent
  wait_sim_ready
  local rc=0
  VALIDATE_ARGS="${VALIDATE_ARGS:---duration-s 90 --min-laps 1 --json-out /ws/gym_score.json}"
  validate || rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "==> Certify failed; last gym / agent logs:" >&2
    compose logs --tail 80 sim >&2 || true
    docker logs --tail 80 f1tenth-gym-agent >&2 || true
  fi
  stop_stray_agents
  compose down >/dev/null 2>&1 || true
  if [ "${STACK}" != "sensor_policy" ]; then
    restore_vehicle_gym
  fi
  return "${rc}"
}

down() {
  stop_stray_agents
  compose down
}

CMD="${1:-up}"
case "${CMD}" in
  up)       up ;;
  validate) validate ;;
  certify)  certify ;;
  down)     down ;;
  logs)     compose logs -f sim ;;
  *)
    echo "usage: tools/sim.sh [up|validate|certify|logs|down]" >&2
    echo "  env: STACK=vehicle|sensor_policy MODE=dev|release CHECKPOINT_DIR=/abs/dir CKPT=policy.pt" >&2
    exit 1
    ;;
esac

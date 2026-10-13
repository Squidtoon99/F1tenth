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
# (sensor_racer -> rl_current_gate -> gym_sensor_bridge -> /drive) on Austin.
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

wire_austin_gym() {
  local maps="sim/f1tenth_gym_ros/maps"
  local cfg="sim/f1tenth_gym_ros/config/sim.yaml"
  local assets="training/assets"
  [ -f "${cfg}" ] || die "gym sim.yaml missing (${cfg}); import gym sources first"
  [ -f "${assets}/Austin.yaml" ] || die "missing ${assets}/Austin.yaml"
  [ -f "${assets}/Austin.png" ] || die "missing ${assets}/Austin.png"
  [ -f "${assets}/Austin_centerline.csv" ] || \
    die "missing ${assets}/Austin_centerline.csv"
  cp -f "${assets}/Austin.png" "${assets}/Austin_centerline.csv" "${maps}/"
  # Gym TrackSpec rejects ROS map keys like mode:; keep occupancy fields only.
  python3 - "${assets}/Austin.yaml" "${maps}/Austin.yaml" <<'PY'
import sys

src, dst = sys.argv[1], sys.argv[2]
keep = ("image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh")
out = []
for line in open(src, encoding="utf-8"):
    key = line.split(":", 1)[0].strip()
    if key in keep:
        out.append(line if line.endswith("\n") else line + "\n")
open(dst, "w", encoding="utf-8").write("".join(out))
open(dst.replace("Austin.yaml", "Austin_map.yaml"), "w", encoding="utf-8").write(
    "".join(out)
)
PY
  python3 - "${cfg}" "${assets}/Austin_centerline.csv" <<'PY'
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
    "map_path": "'maps/Austin'",
    "sx": f"{sx:.6f}",
    "sy": f"{sy:.6f}",
    "stheta": f"{stheta:.6f}",
    "async_mode": "False",
    "use_sim_time": "False",
    "kb_teleop": "False",
    "scan_num_beams": "1081",
    "scan_range_min": "0.06",
    "scan_range_max": "30.0",
    "lidar_noise_std": "0.0",
    "lidar_base_link_to_lidar_tf": "[0.27, 0.0, 0.0]",
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
print(f"==> Austin gym map Austin spawn=({sx:.3f}, {sy:.3f}, {stheta:.3f})")
PY
  # 1081-beam lidar inside every physics step caps the graph at ~2.5 Hz.
  # Skip raycast on ST steps; snapshot-scan on a 20 Hz Python thread.
  # Re-apply patches on clean fork sources so leftover live edits cannot
  # miss an insert needle.
  if [ -d sim/f1tenth_gym_ros/.git ]; then
    git -C sim/f1tenth_gym_ros checkout -- \
      f1tenth_gym_ros/gym_bridge.py launch/rl_agent_sim_launch.py
  fi
  if [ -d sim/f1tenth_gym_ros/f1tenth_gym/.git ]; then
    git -C sim/f1tenth_gym_ros/f1tenth_gym checkout -- \
      f1tenth_gym/envs/simulator.py \
      f1tenth_gym/envs/lidar/laser_models.py
  fi
  python3 - <<'PY'
from pathlib import Path

def apply(path, replacements):
    text = path.read_text(encoding="utf-8")
    changed = False
    for item in replacements:
        old, new, label, *rest = item
        optional = bool(rest[0]) if rest else False
        if new in text:
            print(f"==> already patched: {label}")
            continue
        if old not in text:
            if optional:
                print(f"==> skip (not applicable): {label}")
                continue
            raise SystemExit(f"{path}: string not found for: {label}")
        text = text.replace(old, new, 1)
        print(f"==> {label}")
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")

def patch_gym_bridge(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    orig = text
    old_occ = (
        "    occupancy_map[occupancy_map <= 128] = 0.0\n"
        "    occupancy_map[occupancy_map > 128] = 255.0\n"
    )
    new_occ = (
        "    unknown = occupancy_map == 205\n"
        "    occupancy_map[occupancy_map <= 128] = 0.0\n"
        "    occupancy_map[occupancy_map > 128] = 255.0\n"
        "    occupancy_map[unknown] = 205.0\n"
    )
    if new_occ not in text and old_occ in text:
        text = text.replace(old_occ, new_occ, 1)
        print("==> keep ROS unknown cells so gym LiDAR matches Warp trinary")
    if "import threading" not in text:
        text = text.replace("import pathlib\n", "import pathlib\nimport threading\nimport time\n", 1)
        print("==> import threading for off-executor lidar")
    if "from rclpy.qos import qos_profile_sensor_data" not in text:
        text = text.replace(
            "from rclpy.node import Node\n",
            "from rclpy.node import Node\nfrom rclpy.qos import qos_profile_sensor_data\n",
            1,
        )
        print("==> import sensor-data QoS for gym scan")
    old_scan_pub = (
        "self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)"
    )
    new_scan_pub = (
        "self.ego_scan_pub = self.create_publisher("
        "LaserScan, ego_scan_topic, qos_profile_sensor_data)"
    )
    if old_scan_pub in text:
        text = text.replace(old_scan_pub, new_scan_pub, 1)
        print("==> publish gym /scan best-effort so foxglove cannot stall physics")
    old_sync_drive = (
        "        if not self.get_parameter('async_mode').value:\n"
        "            # step the sim immediately and publish odom and scan\n"
        "            self.drive_timer_callback()\n"
        "            self.timer_callback()\n"
    )
    new_sync_drive = (
        "        if not self.get_parameter('async_mode').value:\n"
        "            stamp = (\n"
        "                drive_msg.header.stamp.sec\n"
        "                + drive_msg.header.stamp.nanosec * 1e-9\n"
        "            )\n"
            "            if stamp <= getattr(self, '_last_drive_stamp', -1.0):\n"
            "                return\n"
            "            self._last_drive_stamp = stamp\n"
            "            self.drive_timer_callback()\n"
            "            self.timer_callback()\n"
    )
    if old_sync_drive in text:
        text = text.replace(old_sync_drive, new_sync_drive, 1)
        print("==> lockstep: one physics step per unique /drive stamp")
    ready = "        self.sim_paused = False"
    ready_log = (
        "        self.sim_paused = not (self.get_parameter('async_mode').value)\n"
        "        self._spawn_pose = list(self.ego_pose)\n"
        "        self._force_obs_ns = 0\n"
        "        self.get_logger().info(\n"
        "            'GymBridge ready (paused at spawn)' if self.sim_paused\n"
        "            else 'GymBridge ready')\n"
    )
    if "GymBridge ready" not in text and ready in text:
        text = text.replace(ready, ready_log, 1)
        print("==> log GymBridge ready after publishers exist")
    if "paused at spawn" not in text and "GymBridge ready')" in text:
        old_ready = (
            "        self.sim_paused = False\n"
            "        self.get_logger().info('GymBridge ready')\n"
        )
        if old_ready in text:
            text = text.replace(old_ready, ready_log, 1)
            print("==> start lockstep paused at the configured spawn")
    old_odom_pub = (
        "self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)"
    )
    new_odom_pub = (
        "self.ego_odom_pub = self.create_publisher("
        "Odometry, ego_odom_topic, qos_profile_sensor_data)"
    )
    if old_odom_pub in text:
        text = text.replace(old_odom_pub, new_odom_pub, 1)
        print("==> publish gym odom best-effort to match scan QoS")
    text = text.replace(
        "from rclpy.callback_groups import MutuallyExclusiveCallbackGroup\n", ""
    )
    text = text.replace("from rclpy.executors import MultiThreadedExecutor\n", "")
    text = text.replace(", callback_group=self._lidar_cg", "")
    text = text.replace(", callback_group=self._physics_cg", "")
    text = text.replace("self.timer = self.create_timer(0.004, self.timer_callback)",
                        "self.timer = self.create_timer(0.05, self.timer_callback)")
    text = text.replace("self.timer = self.create_timer(1, self.timer_callback)",
                        "self.timer = self.create_timer(0.05, self.timer_callback)")
    text = text.replace("self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)",
                        "self.drive_timer = self.create_timer(0.05, self.drive_timer_callback)")
    if "timestep=0.05,\n" not in text:
        text = text.replace("timestep=0.01,\n", "timestep=0.05,\n", 1)
        print("==> gym timestep 0.05 (20 Hz, 1.0x realtime)")
    mt = (
        "    gym_bridge = GymBridge()\n"
        "    executor = MultiThreadedExecutor(num_threads=2)\n"
        "    executor.add_node(gym_bridge)\n"
        "    executor.spin()"
    )
    if mt in text:
        text = text.replace(mt, "    gym_bridge = GymBridge()\n    rclpy.spin(gym_bridge)", 1)
        print("==> spin gym bridge single-threaded")
    cg_block = (
        "        self._physics_cg = MutuallyExclusiveCallbackGroup()\n"
        "        self._lidar_cg = MutuallyExclusiveCallbackGroup()\n"
    )
    if cg_block in text:
        text = text.replace(cg_block, "", 1)
        print("==> drop unused gym callback groups")
    if "scan_every_step = False" not in text:
        needle = "            self._update_sim_state()\n\n        if not self.get_parameter('async_mode').value:"
        if needle not in text:
            raise SystemExit(f"{path}: async_mode insert point missing")
        text = text.replace(
            needle,
            "            self._update_sim_state()\n"
            "        self.env.unwrapped.sim.scan_every_step = False\n"
            "        if self.env.unwrapped.sim.scan_enabled:\n"
            "            self.env.unwrapped.sim._update_scans()\n"
            "            self.env.unwrapped.sim._update_scans()\n"
            "            self._update_sim_state()\n\n"
            "        if not self.get_parameter('async_mode').value:",
            1,
        )
        print("==> defer gym lidar and warm numba")
    thread_start = (
        "        self._scan_thread = threading.Thread("
        "target=self._lidar_loop, daemon=True)\n"
        "        self._scan_thread.start()\n"
    )
    if thread_start in text:
        text = text.replace(thread_start, "", 1)
        print("==> disable gym lidar thread")
    lock_only = (
        "        self._scan_lock = threading.Lock()\n"
        "        self._scan_stop = False\n"
    )
    if "self._scan_lock" not in text:
        start = (
            "        self.env.unwrapped.sim.scan_every_step = False\n"
            "        if self.env.unwrapped.sim.scan_enabled:\n"
            "            self.env.unwrapped.sim._update_scans()\n"
            "            self.env.unwrapped.sim._update_scans()\n"
            "            self._update_sim_state()\n"
        )
        if start not in text:
            start = "        self.env.unwrapped.sim.scan_every_step = False\n"
        text = text.replace(start, start + lock_only, 1)
        print("==> allocate gym lidar lock before physics timers")
    timer_scan = (
        "        sim = self.env.unwrapped.sim\n"
        "        if sim.scan_enabled and sim.scan_sims:\n"
        "            pose = np.array(sim.state.poses[0], copy=True)\n"
        "            pose = sim._lidar_pose_from_base(pose)\n"
        "            self.ego_scan = [float(x) for x in sim.scan_sims[0].scan(pose, None)]\n"
    )
    unmounted_scan = (
        "            pose = np.array(sim.state.poses[0], copy=True)\n"
        "            self.ego_scan = [float(x) for x in sim.scan_sims[0].scan(pose, None)]\n"
    )
    mounted_scan = (
        "            pose = np.array(sim.state.poses[0], copy=True)\n"
        "            pose = sim._lidar_pose_from_base(pose)\n"
        "            self.ego_scan = [float(x) for x in sim.scan_sims[0].scan(pose, None)]\n"
    )
    if unmounted_scan in text:
        text = text.replace(unmounted_scan, mounted_scan, 1)
        print("==> raycast gym lidar from the LiDAR mount frame")
    if timer_scan not in text:
        needle = (
            "    def timer_callback(self):\n"
            "        if self.sim_paused:\n"
            "            return  # Skip stepping the sim if paused\n"
            "        \n"
            "        ts = self.get_clock().now().to_msg()\n"
        )
        if needle not in text:
            if "sim.scan_sims[0].scan(pose" in text:
                print("==> already patched: raycast gym lidar on the publish timer")
            else:
                raise SystemExit(f"{path}: timer_callback insert missing")
        text = text.replace(
            needle,
            "    def timer_callback(self):\n"
            "        if self.sim_paused:\n"
            "            return  # Skip stepping the sim if paused\n"
            "        \n"
            + timer_scan
            + "        ts = self.get_clock().now().to_msg()\n",
            1,
        )
        print("==> raycast gym lidar on the 20 Hz publish timer")
    if "with self._scan_lock:" not in text.split("def drive_timer_callback", 1)[-1][:800]:
        old = (
            "        if not self.has_opp:\n"
            "            _, _, self.done, _, _ = self.env.step(\n"
            "                np.array([[self.ego_steer, self.ego_requested_speed]])\n"
            "            )\n"
        )
        new = (
            "        with self._scan_lock:\n"
            "            if not self.has_opp:\n"
            "                _, _, self.done, _, _ = self.env.step(\n"
            "                    np.array([[self.ego_steer, self.ego_requested_speed]])\n"
            "                )\n"
        )
        if old not in text:
            raise SystemExit(f"{path}: drive step insert missing")
        text = text.replace(old, new, 1)
        old_else = (
            "        else:\n"
            "            _, _, self.done, _, _ = self.env.step(\n"
            "                np.array(\n"
            "                    [\n"
            "                        [self.ego_steer, self.ego_requested_speed],\n"
            "                        [self.opp_steer, self.opp_requested_speed],\n"
            "                    ]\n"
            "                )\n"
            "            )\n"
            "        self._update_sim_state()\n"
        )
        new_else = (
            "            else:\n"
            "                _, _, self.done, _, _ = self.env.step(\n"
            "                    np.array(\n"
            "                        [\n"
            "                            [self.ego_steer, self.ego_requested_speed],\n"
            "                            [self.opp_steer, self.opp_requested_speed],\n"
            "                        ]\n"
            "                    )\n"
            "                )\n"
            "            self._update_sim_state()\n"
        )
        if old_else not in text:
            raise SystemExit(f"{path}: drive opp-step insert missing")
        text = text.replace(old_else, new_else, 1)
        print("==> hold pose lock only around physics step")
    ray = (
        "        sim = self.env.unwrapped.sim\n"
        "        if not getattr(sim, 'scan_every_step', True) and sim.scan_enabled:\n"
        "            sim._update_scans()\n"
        "            self._update_sim_state()\n"
        "        ts = self.get_clock().now().to_msg()"
    )
    if ray in text:
        text = text.replace(
            ray,
            "        ts = self.get_clock().now().to_msg()",
            1,
        )
        print("==> publish cached scan without blocking physics")
    stale = "        self.ego_scan = list(scans[0])"
    fresh = (
        "        if getattr(self.env.unwrapped.sim, 'scan_every_step', True):\n"
        "            self.ego_scan = list(scans[0])"
    )
    if stale in text and fresh not in text:
        text = text.replace(stale, fresh, 1)
        print("==> do not clobber thread lidar from physics")
    odom_from_phys = (
        "        self._update_sim_state()\n"
        "        ts = self.get_clock().now().to_msg()\n"
        "        self._publish_odom(ts)\n"
    )
    if odom_from_phys not in text:
        old = (
            "        self._update_sim_state()\n"
            "        if self.get_parameter('use_sim_time_bridge').value:\n"
            "            clock_msg = Clock()"
        )
        new = (
            "        self._update_sim_state()\n"
            "        ts = self.get_clock().now().to_msg()\n"
            "        self._publish_odom(ts)\n"
            "        self._publish_transforms(ts)\n"
            "        self._publish_laser_transforms(ts)\n"
            "        self._publish_wheel_transforms(ts)\n"
            "        if self.get_parameter('use_sim_time_bridge').value:\n"
            "            clock_msg = Clock()"
        )
        if old not in text:
            print("==> already patched: publish odom from physics timer")
        else:
            text = text.replace(old, new, 1)
            print("==> publish odom from physics timer without waiting for lidar")
    busy = (
        "            scan = scanner.scan(pose, None)\n"
        "            self.ego_scan = [float(x) for x in scan]\n"
        "            time.sleep(0.0)\n"
    )
    paced = (
        "            t0 = time.perf_counter()\n"
        "            scan = scanner.scan(pose, None)\n"
        "            self.ego_scan = [float(x) for x in scan]\n"
        "            time.sleep(max(0.0, 0.05 - (time.perf_counter() - t0)))\n"
    )
    if busy in text:
        text = text.replace(busy, paced, 1)
        print("==> pace lidar thread at 20 Hz")
    sim_stamp = (
        "        ts = self.get_clock().now().to_msg()\n"
        "        if self.get_parameter('use_sim_time_bridge').value:\n"
        "            sim_time = self.env.unwrapped.sim_time\n"
        "            ts.sec = int(sim_time // 1.0)\n"
        "            ts.nanosec = int((sim_time % 1.0) * 1e9)\n"
    )
    wall_odom = "        ts = self.get_clock().now().to_msg()\n        t_pub = time.perf_counter()\n        self._publish_odom(ts)\n"
    sim_odom = sim_stamp + "        t_pub = time.perf_counter()\n        self._publish_odom(ts)\n"
    if wall_odom in text and sim_stamp not in text.split("def drive_timer_callback", 1)[-1][:2500]:
        text = text.replace(wall_odom, sim_odom, 1)
        print("==> stamp physics odom with sim time")
    wall_odom2 = (
        "        self._update_sim_state()\n"
        "        ts = self.get_clock().now().to_msg()\n"
        "        self._publish_odom(ts)\n"
    )
    sim_odom2 = (
        "        self._update_sim_state()\n"
        + sim_stamp
        + "        self._publish_odom(ts)\n"
    )
    if wall_odom2 in text:
        text = text.replace(wall_odom2, sim_odom2, 1)
        print("==> stamp physics odom with sim time")
    clock_after_scan = (
        "            ts.sec = int(sim_time // 1.0)\n"
        "            ts.nanosec = int((sim_time % 1.0) * 1e9)\n"
        "\n"
        "        # pub scans\n"
    )
    clock_pub_scan = (
        "            ts.sec = int(sim_time // 1.0)\n"
        "            ts.nanosec = int((sim_time % 1.0) * 1e9)\n"
        "            clock_msg = Clock()\n"
        "            clock_msg.clock = ts\n"
        "            self.clock_pub.publish(clock_msg)\n"
        "\n"
        "        # pub scans\n"
    )
    if clock_after_scan in text and "clock_msg.clock = ts" not in text:
        text = text.replace(clock_after_scan, clock_pub_scan, 1)
        print("==> publish /clock from the scan timer so lockstep can bootstrap")
    scan_only = (
        "        scan.ranges = self.ego_scan\n"
        "        self.ego_scan_pub.publish(scan)\n"
        "\n"
        "        if self.has_opp:\n"
    )
    scan_and_odom = (
        "        scan.ranges = self.ego_scan\n"
        "        self.ego_scan_pub.publish(scan)\n"
        "        self._publish_odom(ts)\n"
        "\n"
        "        if self.has_opp:\n"
    )
    if scan_only in text and "self.ego_scan_pub.publish(scan)\n        self._publish_odom(ts)" not in text:
        text = text.replace(scan_only, scan_and_odom, 1)
        print("==> heartbeat last odom with the scan timer")
    clock_create = "            self.clock_pub = self.create_publisher(Clock, '/clock', 10)\n"
    clock_best_effort = (
        "            self.clock_pub = self.create_publisher(\n"
        "                Clock, '/clock', qos_profile_sensor_data)\n"
    )
    clock_init = (
        "            self.clock_pub = self.create_publisher(Clock, '/clock', 10)\n"
        "            clock_msg = Clock()\n"
        "            sim_time = float(self.env.unwrapped.sim_time)\n"
        "            clock_msg.clock.sec = int(sim_time // 1.0)\n"
        "            clock_msg.clock.nanosec = int((sim_time % 1.0) * 1e9)\n"
        "            self.clock_pub.publish(clock_msg)\n"
    )
    old_clock_if = (
        "        if self.get_parameter('use_sim_time_bridge').value:\n"
        "            self.get_logger().info('Using simulation time. "
        "Will publish /clock topic. Drive and odom will be as fast as possible.')\n"
    )
    new_clock_if = (
        "        if (not self.get_parameter('async_mode').value) or "
        "self.get_parameter('use_sim_time_bridge').value:\n"
        "            self.get_logger().info('Using simulation time. "
        "Will publish /clock topic. Drive and odom will be as fast as possible.')\n"
    )
    if old_clock_if in text:
        text = text.replace(old_clock_if, new_clock_if, 1)
        print("==> always publish /clock in lockstep mode")
    if clock_create in text and "sim_time = float(self.env.unwrapped.sim_time)" not in text:
        text = text.replace(clock_create, clock_init, 1)
        print("==> publish initial /clock at sim t=0")
    elif clock_best_effort in text:
        text = text.replace(clock_best_effort, clock_init, 1)
        print("==> publish initial /clock at sim t=0")
    pause_skip = (
        "    def timer_callback(self):\n"
        "        if self.sim_paused:\n"
        "            return  # Skip stepping the sim if paused\n"
    )
    pause_pub = (
        "    def timer_callback(self):\n"
        "        # Lockstep pause still publishes scan/odom; physics does not step.\n"
    )
    if pause_skip in text:
        text = text.replace(pause_skip, pause_pub, 1)
        print("==> publish observations while lockstep is paused")
    reset_skip = (
        "    def ego_reset_callback(self, pose_msg):\n"
        "        if self.sim_paused:\n"
        "            return  # Skip stepping the sim if paused\n\n"
    )
    if reset_skip in text:
        text = text.replace(
            reset_skip,
            "    def ego_reset_callback(self, pose_msg):\n",
            1,
        )
        print("==> /initialpose resets spawn even while paused")
    pause_cb = (
        "    def pause_callback(self, msg):\n"
        "        self.sim_paused = msg.data\n"
        "        self.get_logger().info("
        "f\"Simulation {'paused' if self.sim_paused else 'resumed'}\")\n"
    )
    pause_reset = (
        "    def pause_callback(self, msg):\n"
        "        was_paused = self.sim_paused\n"
        "        self.sim_paused = bool(msg.data)\n"
        "        if was_paused and not self.sim_paused:\n"
        "            spawn = getattr(self, '_spawn_pose', self.ego_pose)\n"
        "            if self.has_opp:\n"
        "                self.env.reset(options={\"poses\": np.array(\n"
        "                    [spawn, self.opp_pose])})\n"
        "            else:\n"
        "                self.env.reset(options={\"poses\": np.array([spawn])})\n"
        "            self._update_sim_state()\n"
        "            self._last_drive_stamp = -1.0\n"
        "            self.ego_requested_speed = 0.0\n"
        "            self.ego_steer = 0.0\n"
        "            self._force_obs_ns = 1\n"
        "            self.timer_callback()\n"
        "        self.get_logger().info("
        "f\"Simulation {'paused' if self.sim_paused else 'resumed'}\")\n"
    )
    if "was_paused = self.sim_paused" not in text and pause_cb in text:
        text = text.replace(pause_cb, pause_reset, 1)
        print("==> unpause resets the car to the configured spawn")
    force_stamp = (
        "            ts.sec = int(sim_time // 1.0)\n"
        "            ts.nanosec = int((sim_time % 1.0) * 1e9)\n"
        "            clock_msg = Clock()\n"
        "            clock_msg.clock = ts\n"
        "            self.clock_pub.publish(clock_msg)\n"
        "\n"
        "        # pub scans\n"
    )
    force_stamp_new = (
        "            ts.sec = int(sim_time // 1.0)\n"
        "            ts.nanosec = int((sim_time % 1.0) * 1e9)\n"
        "            extra = int(getattr(self, '_force_obs_ns', 0))\n"
        "            if extra:\n"
        "                nsec = int(ts.nanosec) + extra\n"
        "                ts.sec += nsec // 1_000_000_000\n"
        "                ts.nanosec = nsec % 1_000_000_000\n"
        "                self._force_obs_ns = 0\n"
        "            clock_msg = Clock()\n"
        "            clock_msg.clock = ts\n"
        "            self.clock_pub.publish(clock_msg)\n"
        "\n"
        "        # pub scans\n"
    )
    if "_force_obs_ns', 0))" not in text and force_stamp in text:
        text = text.replace(force_stamp, force_stamp_new, 1)
        print("==> unique scan stamp on unpause so the agent reticks")
    lockstep_ts = (
        "        ts = self.get_clock().now().to_msg()\n"
        "        if self.get_parameter('use_sim_time_bridge').value:\n"
    )
    lockstep_ts_new = (
        "        ts = self.get_clock().now().to_msg()\n"
        "        if not self.get_parameter('async_mode').value:\n"
        "            sim_time = float(self.env.unwrapped.sim_time)\n"
        "            ts.sec = int(sim_time // 1.0)\n"
        "            ts.nanosec = int((sim_time % 1.0) * 1e9)\n"
        "            extra = int(getattr(self, '_force_obs_ns', 0))\n"
        "            if extra:\n"
        "                nsec = int(ts.nanosec) + extra\n"
        "                ts.sec += nsec // 1_000_000_000\n"
        "                ts.nanosec = nsec % 1_000_000_000\n"
        "                self._force_obs_ns = 0\n"
        "            if hasattr(self, 'clock_pub'):\n"
        "                clock_msg = Clock()\n"
        "                clock_msg.clock = ts\n"
        "                self.clock_pub.publish(clock_msg)\n"
        "        elif self.get_parameter('use_sim_time_bridge').value:\n"
    )
    if "if not self.get_parameter('async_mode').value:" not in text.split(
        "def timer_callback", 1
    )[-1][:1800]:
        # Only rewrite the publish-timer stamp, not physics-timer stamps.
        needle = (
            "            self.ego_scan = [float(x) for x in sim.scan_sims[0].scan(pose, None)]\n"
            "        ts = self.get_clock().now().to_msg()\n"
            "        if self.get_parameter('use_sim_time_bridge').value:\n"
        )
        repl = (
            "            self.ego_scan = [float(x) for x in sim.scan_sims[0].scan(pose, None)]\n"
            + lockstep_ts_new
        )
        if needle in text:
            text = text.replace(needle, repl, 1)
            print("==> lockstep observation stamps use frozen sim time")
        elif lockstep_ts in text.split("def timer_callback", 1)[-1]:
            text = text.replace(lockstep_ts, lockstep_ts_new, 1)
            print("==> lockstep observation stamps use frozen sim time")
    if text != orig:
        path.write_text(text, encoding="utf-8")
    else:
        print("==> gym_bridge already patched")

bridge = Path("sim/f1tenth_gym_ros/f1tenth_gym_ros/gym_bridge.py")
sim = Path("sim/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym/envs/simulator.py")
lidar = Path("sim/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym/envs/lidar/laser_models.py")
launch = Path("sim/f1tenth_gym_ros/launch/rl_agent_sim_launch.py")
patch_gym_bridge(bridge)
if lidar.exists():
    apply(
        lidar,
        (
            (
                "        # get the distance transform\n"
                "        self.dt = get_dt(self.map_img, self.map_resolution)\n",
                "        # get the distance transform\n"
                "        bitmap = np.where(\n"
                "            (self.map_img == 0) | (self.map_img == 205), 0.0, 255.0\n"
                "        )\n"
                "        self.dt = get_dt(bitmap, self.map_resolution)\n"
                "        self.dt[self.map_img == 205] = self.max_range\n",
                "stamp unknown LiDAR cells as max-range like Warp",
            ),
        ),
    )
if launch.exists():
    launch_text = launch.read_text(encoding="utf-8")
    fox = "    ld.add_action(foxglove_bridge_node)\n"
    if fox in launch_text:
        launch_text = launch_text.replace(
            fox, "    # foxglove omitted: reliable scan/odom stalls gym\n", 1
        )
        print("==> disable foxglove so reliable odom cannot block the gym executor")
    old_clock = (
        "            'use_sim_time': False,\n"
        "            'use_sim_time_bridge': use_sim_time,\n"
    )
    new_clock = (
        "            'use_sim_time': False,\n"
        "            'use_sim_time_bridge': True,\n"
    )
    if old_clock in launch_text:
        launch_text = launch_text.replace(old_clock, new_clock, 1)
        print("==> gym node stays wall-clock; always publish /clock")
    launch.write_text(launch_text, encoding="utf-8")
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
  # vehicle reads IV_2026; sensor_policy uses the Austin map wired below.
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
    wire_austin_gym
  else
    restore_vehicle_gym
    verify_track_assets
  fi
  echo "==> Starting novnc + gym bridge (ROS_DOMAIN_ID=${ROS_DOMAIN_ID})"
  compose up -d novnc sim
  wait_sim_ready
  if [ "${MODE}" = "release" ]; then
    echo "==> Launching ${STACK} graph from the generic runtime image (/config + /policies)"
    # Override the image CMD (race.launch.py) so no hardware drivers start in sim.
    compose run --rm --service-ports runtime \
      ros2 launch f1tenth_bringup sim.launch.py \
      stack:="${STACK}" checkpoint_path:="/policies/${CKPT}"
    return
  fi
  echo "==> Building the agent workspace (racing_rl + control + bringup + common)"
  compose run --rm --entrypoint taskset agent -c 4-7 bash -lc "
    source /opt/ros/humble/setup.bash &&
    colcon build --symlink-install --base-paths src libs \
      --packages-up-to f1tenth_rl_agent f1tenth_rl_vehicle f1tenth_control \
        f1tenth_bringup f1tenth_common f1tenth_policy"
  echo "==> Launching the ${STACK} agent graph (policy: /policies/${CKPT})"
  echo "    Foxglove: ws://localhost:8765   |   RViz over VNC: http://localhost:8080/vnc.html"
    compose run --rm --service-ports --entrypoint taskset agent -c 4-7 bash -lc "
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
      -e ROS_LOCALHOST_ONLY=0 \
      "${agent_cid}" bash -lc "
        source /opt/ros/humble/setup.bash &&
        source /ws/install/setup.bash &&
        python3 ${script} --expect-obs-dim ${expect} ${VALIDATE_ARGS:-}"
    return
  fi
  compose run --rm -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --entrypoint taskset agent -c 4-7 bash -lc "
    source /opt/ros/humble/setup.bash &&
    source install/setup.bash &&
    python3 ${script} --expect-obs-dim ${expect} ${VALIDATE_ARGS:-}"
}

wait_sim_ready() {
  echo "==> Waiting for gym bridge"
  local cid=""
  local i
  for i in $(seq 1 150); do
    cid="$(compose ps -q sim 2>/dev/null || true)"
    if [ -n "${cid}" ] && docker logs "${cid}" 2>&1 | grep -q "GymBridge ready"; then
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
    wire_austin_gym
  else
    restore_vehicle_gym
    verify_track_assets
  fi
  echo "==> Certify ${STACK} ${CKPT} (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}')"
  compose down >/dev/null 2>&1 || true
  stop_stray_agents
  compose up -d novnc sim
  wait_sim_ready
  if [ "${MODE}" = "dev" ]; then
    echo "==> Building the agent workspace"
    compose run --rm --entrypoint taskset agent -c 4-7 bash -lc "
      source /opt/ros/humble/setup.bash &&
      colcon build --symlink-install --base-paths src libs \
        --packages-up-to f1tenth_rl_agent f1tenth_rl_vehicle f1tenth_control \
          f1tenth_bringup f1tenth_common f1tenth_policy"
    compose run -d --name f1tenth-gym-agent --service-ports --entrypoint taskset agent -c 4-7 bash -lc "
      source /opt/ros/humble/setup.bash &&
      source install/setup.bash &&
      $(agent_launch_cmd)"
  else
    compose run -d --name f1tenth-gym-agent --service-ports --entrypoint taskset runtime -c 4-7 \
      ros2 launch f1tenth_bringup sim.launch.py \
      stack:="${STACK}" checkpoint_path:="/policies/${CKPT}"
  fi
  wait_graph_ready f1tenth-gym-agent
  local rc=0
  VALIDATE_ARGS="${VALIDATE_ARGS:---duration-s 180 --min-laps 1 --repeats 3 --json-out /ws/gym_score.json}"
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

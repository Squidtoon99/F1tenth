import ast
import math
from pathlib import Path

import pytest

BRINGUP = Path(__file__).resolve().parents[1]
REPO = BRINGUP.parents[2]
SENSOR_POLICY_SIM = BRINGUP / "launch" / "sensor_policy_sim.launch.py"
SIM_LAUNCH = BRINGUP / "launch" / "sim.launch.py"
SENSOR_POLICY_LAUNCH = BRINGUP / "launch" / "sensor_policy.launch.py"
SIM_SH = REPO / "tools" / "sim.sh"
COMPOSE_SIM = REPO / "deploy" / "docker" / "docker-compose.sim.yml"
CENTERLINE = REPO / "training" / "assets" / "courtyard_2_centerline.csv"
SETUP_PY = (
    REPO / "src" / "racing_rl" / "f1tenth_rl_agent" / "setup.py"
)


def _launch_executables(launch_file: Path) -> set[str]:
    source = launch_file.read_text()
    tree = ast.parse(source)
    return {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Node"
        for keyword in node.keywords
        if keyword.arg == "executable"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    }


def _node_inline_parameters(launch_file: Path, variable_name: str) -> dict:
    tree = ast.parse(launch_file.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        for keyword in node.value.keywords:
            if keyword.arg != "parameters" or not isinstance(keyword.value, ast.List):
                continue
            inline = next(
                item for item in keyword.value.elts if isinstance(item, ast.Dict)
            )
            out = {}
            for key, value in zip(inline.keys, inline.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if isinstance(value, ast.Constant):
                        out[key.value] = value.value
                    else:
                        out[key.value] = value
            return out
    raise AssertionError(f"Node assignment {variable_name!r} not found")


def _courtyard_spawn():
    rows = []
    for line in CENTERLINE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("x_m"):
            continue
        parts = line.split(",")
        rows.append((float(parts[0]), float(parts[1])))
        if len(rows) == 2:
            break
    sx, sy = rows[0]
    stheta = math.atan2(rows[1][1] - rows[0][1], rows[1][0] - rows[0][0])
    return sx, sy, stheta


def test_sensor_policy_sim_graph_has_bridge_and_no_drivers():
    executables = _launch_executables(SENSOR_POLICY_SIM)

    assert executables == {"sensor_racer", "rl_current_gate", "gym_sensor_bridge"}
    assert "vesc_driver_node" not in executables
    assert "joy_node" not in executables
    assert "urg_node_driver" not in executables
    assert "safety" not in executables


def test_sensor_policy_sim_uses_si_imu_and_80_40():
    source = SENSOR_POLICY_SIM.read_text()
    racer = _node_inline_parameters(SENSOR_POLICY_SIM, "sensor_racer")
    gate = _node_inline_parameters(SENSOR_POLICY_SIM, "current_gate")

    assert racer["imu_accel_to_ms2"] == 1.0
    assert racer["imu_gyro_to_rads"] == 1.0
    assert racer["imu_ax_bias"] == 0.0
    assert racer["twist_vx_sign"] == 1.0
    assert racer["sensor_max_age_s"] == 1.0
    assert racer["steering_delta_max_rad"] == pytest.approx(math.pi / 60.0)
    assert racer["history_stride"] == 1
    assert gate["rl_command_timeout_s"] == 1.0
    bridge = _node_inline_parameters(SENSOR_POLICY_SIM, "gym_sensor_bridge")
    assert bridge["speed_scale"] == 5.0
    assert racer["publish_raw_observation"] is True
    assert racer["publish_diagnostics"] is True
    assert '"i_drive_max_a"' in source
    assert 'default_value="80.0"' in source
    assert 'default_value="40.0"' in source
    assert {"i_drive_max_a", "i_brake_max_a"} <= set(racer)
    assert {"i_drive_max_a", "i_brake_max_a", "i_brake_safe_a"} <= set(gate)
    assert "enable_drivers" not in source
    assert "/ego_racecar/odom" in source


def test_sim_launch_dispatches_sensor_policy_stack():
    source = SIM_LAUNCH.read_text()

    assert '"stack"' in source
    assert "sensor_policy_sim.launch.py" in source
    assert "sensor_policy" in source
    assert "is_sensor_policy" in source
    assert "is_vehicle" in source


def test_on_car_sensor_policy_launch_unchanged_defaults():
    source = SENSOR_POLICY_LAUNCH.read_text()

    assert 'default_value="5.0"' in source
    assert 'default_value="true"' in source
    assert "gym_sensor_bridge" not in source
    assert "history_stride" not in source
    assert "0.05235987755982988" not in source


def test_gym_sensor_bridge_is_registered():
    assert (
        "gym_sensor_bridge = f1tenth_rl_agent.gym_sensor_bridge_node:main"
        in SETUP_PY.read_text()
    )


def test_sim_sh_sensor_policy_wires_courtyard_and_validates_1097d():
    source = SIM_SH.read_text()
    sx, sy, stheta = _courtyard_spawn()

    assert "sensor_policy" in source
    assert "wire_courtyard_gym" in source
    assert "courtyard_2.yaml" in source
    assert "courtyard_2.png" in source
    assert "courtyard_2_centerline.csv" in source
    assert "validate_closed_loop_sensor_policy.py" in source
    assert "certify" in source
    assert "1097" in source
    assert "stop_stray_agents" in source
    assert "wait_sim_ready" in source
    assert "wait_graph_ready f1tenth-gym-agent" in source
    assert "Running in asynchronous mode" in source
    assert "name=f1tenth-gym-agent" in source
    assert "name=docker-agent-run" in source
    assert "--json-out /ws/gym_score.json" in source
    validator = (
        REPO
        / "src"
        / "racing_rl"
        / "f1tenth_rl_agent"
        / "test"
        / "validate_closed_loop_sensor_policy.py"
    )
    validator_src = validator.read_text()
    assert "_spin_until" in validator_src
    assert "timeout_sec=0.005" in validator_src
    assert "scan_every_step" in source
    assert "skip lidar raycast on physics steps" in source
    assert "disable gym lidar thread" in source
    assert "raycast gym lidar on the 20 Hz publish timer" in source
    assert "publish gym /scan best-effort so foxglove cannot stall physics" in source
    assert "publish odom from physics timer" in source
    assert "gym timestep 0.05 (20 Hz, 1.0x realtime)" in source
    bridge_src = (
        REPO
        / "src"
        / "racing_rl"
        / "f1tenth_rl_agent"
        / "f1tenth_rl_agent"
        / "gym_sensor_bridge_node.py"
    ).read_text()
    assert "create_timer(1.0 / 20.0, self._on_imu_hold)" in bridge_src
    racer_src = (
        REPO
        / "src"
        / "racing_rl"
        / "f1tenth_rl_agent"
        / "f1tenth_rl_agent"
        / "sensor_racer_node.py"
    ).read_text()
    assert "MultiThreadedExecutor" in racer_src
    assert "FASTDDS_BUILTIN_TRANSPORTS=UDPv4" in COMPOSE_SIM.read_text()
    assert "PYTHONPATH=/sim_ws/src/f1tenth_gym_ros/f1tenth_gym" in COMPOSE_SIM.read_text()
    assert "network_mode: \"service:sim\"" in COMPOSE_SIM.read_text()
    assert "validate is vehicle-only" not in source
    assert abs(sx - 0.89) < 0.02
    assert abs(sy - (-3.37)) < 0.02
    assert abs(stheta + 0.909) < 0.01

import ast
import math
from pathlib import Path

BRINGUP = Path(__file__).resolve().parents[1]
REPO = BRINGUP.parents[2]
SENSOR_POLICY_SIM = BRINGUP / "launch" / "sensor_policy_sim.launch.py"
SIM_LAUNCH = BRINGUP / "launch" / "sim.launch.py"
SENSOR_POLICY_LAUNCH = BRINGUP / "launch" / "sensor_policy.launch.py"
SIM_SH = REPO / "tools" / "sim.sh"
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


def test_gym_sensor_bridge_is_registered():
    assert (
        "gym_sensor_bridge = f1tenth_rl_agent.gym_sensor_bridge_node:main"
        in SETUP_PY.read_text()
    )


def test_sim_sh_sensor_policy_wires_courtyard_and_keeps_validate_vehicle():
    source = SIM_SH.read_text()
    sx, sy, stheta = _courtyard_spawn()

    assert "sensor_policy" in source
    assert "wire_courtyard_gym" in source
    assert "courtyard_2.yaml" in source
    assert "courtyard_2.png" in source
    assert "courtyard_2_centerline.csv" in source
    assert "validate is vehicle-only" in source
    assert abs(sx - 0.89) < 0.02
    assert abs(sy - (-3.37)) < 0.02
    assert abs(stheta + 0.909) < 0.01

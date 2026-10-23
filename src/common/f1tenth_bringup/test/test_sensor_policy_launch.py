import ast
from pathlib import Path

BRINGUP = Path(__file__).resolve().parents[1]
SENSOR_POLICY_LAUNCH = BRINGUP / "launch" / "sensor_policy.launch.py"
CAR_OVERLAY = BRINGUP.parents[2] / "deploy" / "cars" / "car01" / "params.yaml"
CAR_LAUNCH = BRINGUP / "launch" / "car.launch.py"
RACE_LAUNCH = BRINGUP / "launch" / "race.launch.py"
SENSOR_RACER_NODE = (
    Path(__file__).resolve().parents[3]
    / "racing_rl"
    / "f1tenth_rl_agent"
    / "f1tenth_rl_agent"
    / "sensor_racer_node.py"
)
SENSOR_POLICY_YAML = (
    Path(__file__).resolve().parents[3]
    / "racing_rl"
    / "f1tenth_rl_agent"
    / "config"
    / "sensor_policy.yaml"
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


def _launch_source(launch_file: Path = SENSOR_POLICY_LAUNCH) -> str:
    return launch_file.read_text()


def _node_inline_parameter_keys(variable_name: str) -> set[str]:
    tree = ast.parse(_launch_source())
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
            return {
                key.value
                for key in inline.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError(f"Node assignment {variable_name!r} not found")


def _overlay_node_block(node_name: str) -> str:
    source = CAR_OVERLAY.read_text()
    return source.split(f"\n{node_name}:\n", 1)[1].split("\n\n", 1)[0]


def test_sensor_policy_launch_graph_composition():
    executables = _launch_executables(SENSOR_POLICY_LAUNCH)

    assert "vesc_driver_node" in executables
    assert "vesc_to_odom_node" in executables
    assert "joy_node" in executables
    assert "joy_teleop" in executables
    assert "rl_deadman_gate" in executables
    assert "urg_node_driver" in executables
    assert "sensor_racer" in executables
    assert "rl_current_gate" in executables
    assert "safety" in executables


def test_sensor_policy_launch_no_unsafe_bypass():
    executables = _launch_executables(SENSOR_POLICY_LAUNCH)
    source = _launch_source()

    assert "vesc_actuator" not in executables
    assert "ackermann_to_vesc_node" not in executables
    assert "ackermann_mux" not in executables
    assert "direct_policy" not in source
    assert "Float32MultiArray" not in source
    assert "desired_action" not in source


def test_sensor_policy_launch_single_vesc_command_owner():
    source = _launch_source()

    assert "gate_remappings" in source
    assert "commands/motor/current_dryrun" in source
    assert "commands/motor/brake_dryrun" in source
    assert "commands/servo/position_dryrun" in source
    assert "remappings=gate_remappings" in source
    assert "remappings=" not in source.split("sensor_racer = Node")[1].split(
        "default_gate"
    )[0]


def test_sensor_policy_launch_dry_run_odom_config():
    source = _launch_source()
    dry_run_config = (BRINGUP / "config" / "vesc_to_odom_dryrun.yaml").read_text()

    assert '"vesc_to_odom_dryrun.yaml"' in source
    assert "use_servo_cmd_to_calc_angular_velocity: false" in dry_run_config


def test_sensor_policy_launch_staged_enable_flags():
    source = _launch_source()

    assert '"enable_drivers"' in source
    assert '"enable_teleop"' in source
    assert '"enable_teleop",\n                default_value="false"' in source
    assert "IfCondition(enable_teleop)" in source
    assert '"enable_deadman"' in source
    assert '"enable_racer"' in source
    assert '"enable_gate"' in source
    assert '"enable_safety"' in source
    assert "IfCondition(enable_drivers)" in source
    assert "IfCondition(enable_deadman)" in source
    assert "IfCondition(enable_racer)" in source
    assert "IfCondition(enable_gate)" in source
    assert "IfCondition(enable_safety)" in source
    assert '"enable_deadman",\n                default_value="true"' in source


def test_sensor_policy_launch_wires_gate_limits_and_topics():
    source = _launch_source()
    racer_params = _node_inline_parameter_keys("sensor_racer")
    gate_params = _node_inline_parameter_keys("current_gate")

    assert "rl_current_gate.yaml" in source
    assert {"i_drive_max_a", "i_brake_max_a", "i_slew_a_per_s"} <= racer_params
    assert {
        "i_drive_max_a",
        "i_brake_max_a",
        "i_brake_safe_a",
        "i_slew_a_per_s",
    } <= gate_params
    assert '"gate_config"' in source


def test_car_overlay_keeps_sensor_racer_and_gate_at_60a_drive_20a_brake():
    racer = _overlay_node_block("sensor_racer")
    gate = _overlay_node_block("rl_current_gate")
    source = _launch_source()

    assert '"i_drive_max_a",\n                default_value="60.0"' in source
    assert '"i_brake_max_a",\n                default_value="20.0"' in source
    assert '"i_slew_a_per_s",\n                default_value="200.0"' in source
    assert "i_drive_max_a: 60.0" in racer
    assert "i_brake_max_a: 20.0" in racer
    assert "i_slew_a_per_s: 200.0" in racer
    assert "control_hz: 20.0" in racer
    assert "i_drive_max_a: 60.0" in gate
    assert "i_brake_max_a: 20.0" in gate
    assert "i_slew_a_per_s: 200.0" in gate


def test_sensor_racer_node_does_not_publish_vesc_topics():
    source = SENSOR_RACER_NODE.read_text()

    assert "commands/motor/current" not in source
    assert "commands/motor/brake" not in source
    assert "commands/servo/position" not in source


def test_sensor_policy_yaml_uses_20hz_steering_delta():
    source = SENSOR_POLICY_YAML.read_text()

    assert "steering_delta_max_rad: 0.02617993877991494" in source
    assert "steering_delta_max_rad: 0.05235987755982988" not in source
    assert source.count("steering_delta_max_rad: 0.02617993877991494") == 2


def test_sensor_racer_adopts_checkpoint_steering_delta_in_load_policy():
    load_body = SENSOR_RACER_NODE.read_text().split("def _load_policy", 1)[1]
    load_body = load_body.split("def _stamp_age_s", 1)[0]

    assert "checkpoint_steering_delta_max_rad(" in load_body
    assert "self.steering_delta_max_rad = ckpt_delta" in load_body
    assert "expected_steering_delta_max_rad=self.steering_delta_max_rad" in load_body


def test_sensor_racer_blocks_non_rl_applied_before_pack():
    timer = SENSOR_RACER_NODE.read_text().split("def _on_timer", 1)[1]
    block_at = timer.find("if self._applied_blocks_rl_publish():")
    pack_at = timer.find("pack_observation(")
    infer_at = timer.find("infer_host_obs(")

    assert 0 <= block_at < pack_at < infer_at
    assert "self._reset_runtime(" in timer[block_at:pack_at]
    assert "applied_long = 0.0" in timer
    assert "vesc_current = 0.0" in timer
    assert "VESC_CURRENT_SCALE_A * si.applied_current_fraction" in timer


def test_classical_car_launch_unaffected():
    executables = _launch_executables(CAR_LAUNCH)
    source = _launch_source(CAR_LAUNCH)

    assert "vesc_actuator" in executables
    assert "rl_deadman_gate" in executables
    assert "sensor_racer" not in executables
    assert "rl_current_gate" not in executables
    assert "bringup_launch.py" in source


def test_race_launch_still_uses_classical_car_graph():
    source = _launch_source(RACE_LAUNCH)

    assert "car.launch.py" in source
    assert "sensor_policy.launch.py" not in source

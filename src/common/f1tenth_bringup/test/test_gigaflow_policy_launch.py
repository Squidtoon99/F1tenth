import ast
from pathlib import Path

BRINGUP = Path(__file__).resolve().parents[1]
GIGAFLOW_POLICY_LAUNCH = BRINGUP / "launch" / "gigaflow_policy.launch.py"
CAR_OVERLAY = BRINGUP.parents[2] / "deploy" / "cars" / "car01" / "params.yaml"
CAR_LAUNCH = BRINGUP / "launch" / "car.launch.py"
RACE_LAUNCH = BRINGUP / "launch" / "race.launch.py"
GIGAFLOW_RACER_NODE = (
    Path(__file__).resolve().parents[3]
    / "racing_rl"
    / "f1tenth_rl_agent"
    / "f1tenth_rl_agent"
    / "gigaflow_racer_node.py"
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


def _launch_source(launch_file: Path = GIGAFLOW_POLICY_LAUNCH) -> str:
    return launch_file.read_text()


def _overlay_node_block(node_name: str) -> str:
    source = CAR_OVERLAY.read_text()
    return source.split(f"\n{node_name}:\n", 1)[1].split("\n\n", 1)[0]


def test_gigaflow_policy_launch_graph_composition():
    executables = _launch_executables(GIGAFLOW_POLICY_LAUNCH)

    assert "vesc_driver_node" in executables
    assert "vesc_to_odom_node" in executables
    assert "joy_node" in executables
    assert "joy_teleop" in executables
    assert "rl_deadman_gate" in executables
    assert "urg_node_driver" in executables
    assert "gigaflow_racer" in executables
    assert "rl_current_gate" in executables
    assert "safety" in executables
    assert "sensor_racer" not in executables
    assert "policy_inference" not in executables


def test_gigaflow_policy_launch_no_unsafe_bypass():
    executables = _launch_executables(GIGAFLOW_POLICY_LAUNCH)
    source = _launch_source()

    assert "vesc_actuator" not in executables
    assert "ackermann_to_vesc_node" not in executables
    assert "ackermann_mux" not in executables
    assert "direct_policy" not in source
    assert "desired_action" not in source


def test_gigaflow_policy_launch_single_vesc_command_owner():
    source = _launch_source()

    assert "gate_remappings" in source
    assert "commands/motor/current_dryrun" in source
    assert "remappings=gate_remappings" in source


def test_gigaflow_policy_launch_staged_enable_flags():
    source = _launch_source()

    assert '"enable_drivers"' in source
    assert '"enable_deadman"' in source
    assert '"enable_racer"' in source
    assert '"enable_gate"' in source
    assert "IfCondition(enable_racer)" in source
    assert "IfCondition(enable_deadman)" in source
    assert '"enable_deadman",\n                default_value="true"' in source


def test_car_overlay_keeps_gigaflow_racer_and_gate_at_60a_drive_20a_brake():
    racer = _overlay_node_block("gigaflow_racer")
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


def test_gigaflow_racer_node_does_not_publish_vesc_topics():
    source = GIGAFLOW_RACER_NODE.read_text()

    assert "commands/motor/current" not in source
    assert "commands/motor/brake" not in source
    assert "commands/servo/position" not in source


def test_gigaflow_launch_declares_online_ppo_defaults():
    source = _launch_source()
    racer = GIGAFLOW_RACER_NODE.read_text()

    assert '"online_ppo_enable",\n                default_value="false"' in source
    assert '"online_ppo_rollout_steps",\n                default_value="128"' in source
    assert '"online_ppo_lr",\n                default_value="3e-5"' in source
    assert '"online_ppo_kl_coef",\n                default_value="0.1"' in source
    assert '"online_ppo_kl_clip",\n                default_value="0.05"' in source
    assert (
        '"online_ppo_output_dir",\n                default_value="/policies/online_ppo"'
        in source
    )
    assert "self.declare_parameter(\"online_ppo_enable\", False)" in racer
    assert "self._online_ppo.discard()" in racer
    assert "ActuatorCommand.SOURCE_RL" in racer.split("def _on_timer", 1)[1]


def test_gigaflow_racer_blocks_non_rl_applied_before_pack():
    timer = GIGAFLOW_RACER_NODE.read_text().split("def _on_timer", 1)[1]
    block_at = timer.find("if self._applied_blocks_rl_publish():")
    pack_at = timer.find("pack_observation(")
    infer_at = timer.find("infer_host_obs(")

    assert 0 <= block_at < pack_at < infer_at
    assert "self._reset_runtime(" in timer[block_at:pack_at]
    assert "applied_long = 0.0" in timer
    assert "vesc_current = 0.0" in timer
    assert "VESC_CURRENT_SCALE_A * si.applied_current_fraction" in timer


def test_gigaflow_launch_does_not_replace_classical_car_graph():
    executables = _launch_executables(CAR_LAUNCH)
    source = _launch_source(RACE_LAUNCH)

    assert "vesc_actuator" in executables
    assert "gigaflow_racer" not in executables
    assert "gigaflow_policy.launch.py" not in source

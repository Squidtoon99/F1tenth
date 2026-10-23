"""Deadman L1/R1 accepts both linux js0 and SDL GameController indices."""

from pathlib import Path

NODE = (
    Path(__file__).resolve().parents[1]
    / "f1tenth_control"
    / "rl_deadman_gate_node.py"
)


def test_deadman_joy_cb_accepts_js0_and_sdl_l1_r1():
    joy_cb = NODE.read_text().split("def _joy_cb", 1)[1].split(
        "def _deadman_held", 1
    )[0]
    assert "(self.manual_button, 4, 9)" in joy_cb
    assert "(self.autonomous_button, 5, 10)" in joy_cb

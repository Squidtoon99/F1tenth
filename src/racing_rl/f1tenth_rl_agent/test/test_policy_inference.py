"""rclpy integration test for policy_inference_node (runs in the container)."""

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from std_msgs.msg import Float32MultiArray  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_rl_agent.policy_inference_node import PolicyInferenceNode  # noqa: E402


def test_policy_inference_publishes_valid_action():
    rclpy.init()
    node = None
    pub = None
    try:
        node = PolicyInferenceNode()  # no checkpoint -> random-init actor
        pub = rclpy.create_node("obs_pub")
        obs_pub = pub.create_publisher(Float32MultiArray, ifc.TOPIC_OBSERVATION, 10)

        received = {}
        # Subscribe on the separate publisher node: subscribing on `node` itself
        # (which also has a busy observation subscription) starves this callback
        # under single-threaded spin_once.
        pub.create_subscription(
            Float32MultiArray, ifc.TOPIC_ACTION,
            lambda m: received.__setitem__("a", m), 10)

        obs = Float32MultiArray()
        obs.data = [0.0] * ifc.NUM_OBS

        end = node.get_clock().now().nanoseconds + int(5e9)
        while node.get_clock().now().nanoseconds < end and "a" not in received:
            obs_pub.publish(obs)
            rclpy.spin_once(pub, timeout_sec=0.02)
            rclpy.spin_once(node, timeout_sec=0.05)

        assert "a" in received
        a = np.asarray(received["a"].data, dtype=np.float32)
        assert a.shape == (ifc.NUM_ACTIONS,)
        assert bool(np.all(np.abs(a) <= ifc.CLIP_ACTIONS + 1e-5))
    finally:
        if node is not None:
            node.destroy_node()
        if pub is not None:
            pub.destroy_node()
        rclpy.shutdown()

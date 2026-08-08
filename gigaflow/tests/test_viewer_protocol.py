from __future__ import annotations

import json

import numpy as np
import pytest

from gigaflow_f1tenth.viewer.protocol import (
    CAR_POSE_DIM,
    PROTOCOL_VERSION,
    decode_client_message,
    encode_hello,
    encode_tick,
    encode_track,
)


def test_encode_hello_includes_track_catalog():
    hello = json.loads(
        encode_hello(
            track="Austin",
            track_id=0,
            suite="solo",
            checkpoint="actor.pt",
            seed=0,
            device="cpu",
            control_hz=10.0,
            car_length=0.568,
            car_width=0.296,
            car_height=0.24992,
            speed_axis_max_mps=16.2,
            num_cars=1,
            num_obstacles=2,
            obstacle_preset="light",
            obstacle_nonce=3,
            obstacle_presets=["off", "light", "heavy"],
            obstacles=[[4.0, 5.0, 0.2], [8.0, 9.0, 0.4]],
            num_environments=1,
            cars_per_environment=1,
            min_dense_environments=1,
            max_dense_environments=4,
            tracks=["Austin", "Catalunya"],
            suites=["solo", "head_to_head", "dense"],
        )
    )
    assert hello["v"] == PROTOCOL_VERSION
    assert hello["type"] == "hello"
    assert hello["tracks"] == ["Austin", "Catalunya"]
    assert hello["suites"] == ["solo", "head_to_head", "dense"]
    assert hello["num_environments"] == 1
    assert hello["cars_per_environment"] == 1
    assert hello["max_dense_environments"] == 4
    assert hello["speed_axis_max_mps"] == pytest.approx(16.2)
    assert hello["num_obstacles"] == 2
    assert hello["obstacle_preset"] == "light"
    assert hello["obstacle_nonce"] == 3


def test_encode_track_and_tick_with_velocity():
    center = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    left = center + np.asarray([[0.0, 1.0]], dtype=np.float32)
    right = center - np.asarray([[0.0, 1.0]], dtype=np.float32)
    track = json.loads(encode_track(center=center, left=left, right=right))
    assert track["type"] == "track"
    assert track["center"] == [[0.0, 0.0], [1.0, 0.0]]

    tick = json.loads(
        encode_tick(
            step=4,
            sim_fps=9.5,
            cars=np.asarray(
                [[1.0, 2.0, 0.3, 1.0, 1.5, 1.2, 0.9, 0.05, 0.1, -0.4, 0.2]],
                dtype=np.float32,
            ),
            paused=True,
            t=0.4,
            control_dt=0.1,
        )
    )
    assert tick["t"] == pytest.approx(0.4)
    assert tick["control_dt"] == pytest.approx(0.1)
    assert tick["cars"][0][4] == pytest.approx(1.5)
    assert tick["cars"][0][5] == pytest.approx(1.2)
    assert tick["cars"][0][6] == pytest.approx(0.9)
    assert tick["cars"][0][7] == pytest.approx(0.05)
    assert tick["cars"][0][8:] == pytest.approx([0.1, -0.4, 0.2, 0.0])
    assert len(tick["cars"][0]) == CAR_POSE_DIM
    assert tick["cars"][0][11] == 0.0

    # Short rows pad speed/vx/vy/yaw_rate with zeros; t defaults from step*dt.
    tick4 = json.loads(
        encode_tick(
            step=2,
            sim_fps=1.0,
            cars=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            control_dt=0.1,
        )
    )
    assert tick4["t"] == pytest.approx(0.2)
    assert tick4["cars"][0] == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    tick5 = json.loads(
        encode_tick(
            step=1,
            sim_fps=1.0,
            cars=np.asarray([[0.0, 0.0, 0.0, 1.0, 2.0]], dtype=np.float32),
        )
    )
    assert tick5["cars"][0][4] == pytest.approx(2.0)
    assert tick5["cars"][0][5:] == pytest.approx([0.0] * 7)


def test_decode_client_setters_and_controls():
    assert decode_client_message('{"v":4,"type":"reset"}') == {
        "v": 4,
        "type": "reset",
    }
    assert decode_client_message(
        '{"v":4,"type":"set_suite","suite":"dense"}'
    ) == {"v": 4, "type": "set_suite", "suite": "dense"}
    assert decode_client_message(
        '{"v":4,"type":"set_track","track":"Austin"}'
    ) == {"v": 4, "type": "set_track", "track": "Austin"}
    assert decode_client_message(
        '{"v":4,"type":"set_environment_count","environment_count":4}'
    ) == {
        "v": 4,
        "type": "set_environment_count",
        "environment_count": 4,
    }
    with pytest.raises(ValueError, match="unsupported"):
        decode_client_message('{"v":4,"type":"explode"}')
    with pytest.raises(ValueError, match="unsupported suite"):
        decode_client_message('{"v":4,"type":"set_suite","suite":"nope"}')
    with pytest.raises(ValueError, match="environment_count"):
        decode_client_message(
            '{"v":4,"type":"set_environment_count","environment_count":5}'
        )
    assert decode_client_message(
        '{"v":4,"type":"set_obstacle_preset","obstacle_preset":"heavy"}'
    )["obstacle_preset"] == "heavy"
    assert decode_client_message(
        '{"v":4,"type":"respawn_obstacles"}'
    )["type"] == "respawn_obstacles"
    assert decode_client_message(
        '{"v":4,"type":"set_checkpoint","checkpoint":"/runs/a/actor_000100.pt"}'
    )["checkpoint"] == "/runs/a/actor_000100.pt"
    with pytest.raises(ValueError, match="set_checkpoint requires"):
        decode_client_message('{"v":4,"type":"set_checkpoint"}')

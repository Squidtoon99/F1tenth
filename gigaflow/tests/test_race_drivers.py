"""Each trained actor family can be loaded and stepped on the shared observation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "libs" / "f1tenth_policy"))

from f1tenth_policy.actor import actor_architecture_from_module, make_actor  # noqa: E402
from f1tenth_policy.layout import (  # noqa: E402
    ACTOR_ARCHITECTURE_NAME,
    LIDAR_DREAMER_ARCHITECTURE_NAME,
    TINYLIDARNET_ARCHITECTURE_NAME,
)
from gigaflow_f1tenth.artifacts import export_actor_artifact  # noqa: E402
from gigaflow_f1tenth.config import config_from_dict, config_to_dict, load_config  # noqa: E402
from gigaflow_f1tenth.model import build_actor  # noqa: E402
from gigaflow_f1tenth.viewer.drivers import (  # noqa: E402
    describe_checkpoint,
    load_race_driver,
)

SMOKE = ROOT / "configs" / "smoke.yaml"


def _sensor(path: Path, actor) -> None:
    torch.save(
        {
            "actor": actor.state_dict(),
            "actor_architecture": actor_architecture_from_module(actor),
            "obs_norm": {
                "mean": torch.zeros(1097),
                "var": torch.ones(1097),
                "count": 1.0,
            },
        },
        path,
    )


def _act(cfg, path: Path) -> torch.Tensor:
    driver = load_race_driver(cfg, path, "cpu")
    driver.reset(1, "cpu")
    obs = torch.zeros(1, 1097)
    slots = torch.zeros(1, dtype=torch.long)
    live = torch.zeros(1, int(cfg.agents.condition_dim))
    return driver.act(obs, slots, live)


def test_every_trained_kind_steps(tmp_path: Path):
    cfg = load_config(SMOKE)
    cases = []

    gru = tmp_path / "actor_gru.pt"
    export_actor_artifact(cfg, build_actor(cfg), str(gru))
    cases.append((gru, "gigaflow-gru"))

    feedforward = tmp_path / "actor_ff.pt"
    export_actor_artifact(
        cfg, build_actor(cfg, variant="feedforward"), str(feedforward)
    )
    cases.append((feedforward, "gigaflow-feedforward"))

    stacked = tmp_path / "actor_stack.pt"
    export_actor_artifact(
        cfg,
        build_actor(cfg, variant="frame_stack", frame_stack=4),
        str(stacked),
    )
    cases.append((stacked, "gigaflow-frame-stack"))

    cnn = tmp_path / "policy_cnn.pt"
    _sensor(
        cnn,
        make_actor(
            1097, 2, [32], gru_hidden_dim=32, lidar_projection_dim=32
        ),
    )
    cases.append((cnn, "cnn-gru"))

    chunk = tmp_path / "policy_chunk.pt"
    _sensor(
        chunk,
        make_actor(
            1097,
            4,
            [32],
            gru_hidden_dim=32,
            lidar_projection_dim=32,
            actor_type=ACTOR_ARCHITECTURE_NAME,
        ),
    )
    cases.append((chunk, "action-chunk"))

    tiny = tmp_path / "policy_tiny.pt"
    _sensor(
        tiny,
        make_actor(
            1097,
            2,
            [32],
            gru_hidden_dim=32,
            lidar_projection_dim=32,
            actor_type=TINYLIDARNET_ARCHITECTURE_NAME,
        ),
    )
    cases.append((tiny, "tinylidarnet"))

    dreamer = tmp_path / "policy_dreamer.pt"
    _sensor(
        dreamer,
        make_actor(
            1097,
            2,
            [32],
            actor_type=LIDAR_DREAMER_ARCHITECTURE_NAME,
            embed_dim=32,
            deter_dim=32,
            stoch_dim=8,
        ),
    )
    cases.append((dreamer, "lidar-dreamer"))

    for path, kind in cases:
        described = describe_checkpoint(path, cfg)
        assert described is not None
        assert described["kind"] == kind
        actions = _act(cfg, path)
        assert actions.shape == (1, 2)
        assert torch.isfinite(actions).all()


def test_strided_history_uses_the_20hz_window():
    from f1tenth_policy.layout import (  # noqa: E402
        STEER_DELTA0,
        STEER_DELTA2,
        STEER_T,
        STEER_T1,
        STEER_T2,
        THROTTLE_CURRENT,
        THROTTLE_PRED,
    )
    from gigaflow_f1tenth.viewer.drivers import StridedCommandHistory  # noqa: E402

    history = StridedCommandHistory()
    history.reset(1, "cpu")
    for index, angle in enumerate((0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
        history.push(torch.tensor([angle]), torch.tensor([float(index) / 10.0]))
    out = history.project(torch.zeros(1, 1097), torch.zeros(1, dtype=torch.long))
    assert float(out[0, STEER_T]) == pytest.approx(0.30)
    assert float(out[0, STEER_T1]) == pytest.approx(0.20)
    assert float(out[0, STEER_T2]) == pytest.approx(0.10)
    assert float(out[0, STEER_DELTA0]) == pytest.approx(0.10)
    assert float(out[0, STEER_DELTA2]) == pytest.approx(0.10)
    assert float(out[0, THROTTLE_CURRENT]) == pytest.approx(0.6)
    assert float(out[0, THROTTLE_PRED]) == pytest.approx(0.4)


def test_sensor_command_plus_one_is_the_training_delta():
    import math

    import warp as wp

    from gigaflow_f1tenth.sim.stages import physics_control_kernel  # noqa: E402
    from gigaflow_f1tenth.sim.state import SimulatorBuffers  # noqa: E402
    from gigaflow_f1tenth.sim.vehicle import VehicleParams  # noqa: E402
    from gigaflow_f1tenth.viewer.drivers import sensor_control_spec  # noqa: E402
    from gigaflow_f1tenth.viewer.replay import _viewer_suite_config  # noqa: E402

    hz, interval, delta = sensor_control_spec()
    assert hz == pytest.approx(20.0)
    assert interval == 10
    assert delta == pytest.approx(math.pi / 120.0)
    cfg = load_config(SMOKE)
    native = _viewer_suite_config(cfg, "solo", 0)
    sensor = _viewer_suite_config(cfg, "solo", 0, sensor_cadence=True)
    assert native.agents.control_hz == pytest.approx(10.0)
    assert sensor.agents.control_hz == pytest.approx(hz)
    assert sensor.agents.control_interval == interval
    assert sensor.agents.control_interval * sensor.agents.sim_dt == pytest.approx(
        1.0 / hz
    )

    params = VehicleParams(steering_action_mode="delta", steering_delta_max=delta)
    buffers = SimulatorBuffers(cfg, "cpu")
    buffers.seed_defaults(params)
    sim_params = params.to_warp(sim_dt=0.005, control_dt=1.0 / hz)
    arrays = buffers.torch_arrays
    n = int(arrays.x.shape[0])
    arrays.active[0] = 1
    arrays.steer_scale[0] = 1.0
    actions = torch.zeros(n, 2)
    actions[0, 1] = 1.0
    wp.launch(
        physics_control_kernel,
        dim=n,
        inputs=[
            buffers.vehicle_buffers(),
            buffers.active,
            wp.from_torch(actions, dtype=wp.vec2f),
            buffers.prev_x,
            buffers.prev_y,
            buffers.prev_yaw,
            sim_params,
            interval,
        ],
        device=buffers.wp_device,
    )
    assert float(arrays.steer[0]) == pytest.approx(delta, rel=1e-5)


def test_narrow_gigaflow_trunk_stays_out(tmp_path: Path):
    cfg = load_config(SMOKE)
    raw = config_to_dict(cfg)
    agents = dict(raw["agents"])
    agents["gru_hidden_dim"] = 128
    raw["agents"] = agents
    other = config_from_dict(raw, check_training_budget=False)
    path = tmp_path / "actor_narrow.pt"
    export_actor_artifact(other, build_actor(other), str(path))
    assert describe_checkpoint(path, cfg) is None

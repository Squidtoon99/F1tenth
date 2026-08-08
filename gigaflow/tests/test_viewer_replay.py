from __future__ import annotations

import shutil
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.artifacts import (
    build_actor_artifact_payload,
    export_actor_artifact,
)
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.model import build_actor
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import LIDAR_DIM
from gigaflow_f1tenth.viewer.replay import (
    CheckpointReplay,
    ViewerError,
    ViewerLaunchArgs,
    resolve_track_index,
    select_track_view,
    validate_view_args,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tracks" / "oval_centerline.csv"


@pytest.fixture()
def smoke_cache(tmp_path: Path) -> Path:
    local = tmp_path / "local"
    local.mkdir()
    shutil.copy(FIXTURE, local / "oval_centerline.csv")
    from gigaflow_f1tenth.cli import main

    rc = main(
        [
            "prepare-tracks",
            "--config",
            str(SMOKE),
            "--cache-dir",
            str(tmp_path),
            "--skip-download",
            "--lut-resolution",
            "0.5",
            "--edt-resolution",
            "0.25",
        ]
    )
    assert rc == 0
    return tmp_path


@pytest.fixture()
def smoke_actor(tmp_path: Path) -> Path:
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    path = tmp_path / "actor.pt"
    export_actor_artifact(cfg, actor, str(path))
    return path


def test_validate_view_args_missing_atlas(tmp_path: Path, smoke_actor: Path):
    args = ViewerLaunchArgs(
        checkpoint=smoke_actor,
        config=SMOKE,
        cache_dir=tmp_path / "missing",
        suite="solo",
        seed=0,
        device="cpu",
    )
    with pytest.raises(ViewerError, match="track cache directory not found"):
        validate_view_args(args)


def test_validate_view_args_bad_suite(tmp_path: Path, smoke_actor: Path):
    (tmp_path / "atlas.npz").write_bytes(b"x")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    args = ViewerLaunchArgs(
        checkpoint=smoke_actor,
        config=SMOKE,
        cache_dir=tmp_path,
        suite="surprise_braking",
        seed=0,
        device="cpu",
    )
    with pytest.raises(ViewerError, match="unsupported suite"):
        validate_view_args(args)


def test_select_track_and_resolve_name():
    atlas = make_synthetic_oval_atlas()
    view = select_track_view(atlas, 0)
    assert view.num_tracks == 1
    assert view.track_ids == (atlas.track_ids[0],)
    assert resolve_track_index(atlas, track=atlas.track_ids[0], track_id=None) == 0
    with pytest.raises(ViewerError, match="not found"):
        resolve_track_index(atlas, track="nope", track_id=None)


def test_deterministic_replay_startup_and_step(smoke_cache: Path, smoke_actor: Path):
    args = ViewerLaunchArgs(
        checkpoint=smoke_actor,
        config=SMOKE,
        cache_dir=smoke_cache,
        suite="solo",
        seed=0,
        device="cpu",
        track="oval",
        open_browser=False,
    )
    replay_a = CheckpointReplay.from_launch_args(args)
    p0 = replay_a.poses().copy()
    p1 = replay_a.step().copy()
    assert p0.shape[1] == 12
    assert p1.shape == p0.shape
    assert not np.allclose(p0[:, :4], p1[:, :4])
    assert replay_a.speed_axis_max_mps == pytest.approx((320.0 / 0.075) ** (1.0 / 3.0))
    assert replay_a.hello_fields()["speed_axis_max_mps"] == pytest.approx(
        replay_a.speed_axis_max_mps
    )

    replay_b = CheckpointReplay.from_launch_args(args)
    q0 = replay_b.poses()
    q1 = replay_b.step()
    assert np.allclose(p0, q0, atol=1e-5)
    assert np.allclose(p1, q1, atol=1e-5)

    replay_a.set_paused(True)
    held = replay_a.step()
    assert np.allclose(held, p1, atol=1e-5)
    replay_a.reset()
    assert replay_a.step_index == 0


def test_runtime_suite_switch(smoke_cache: Path, smoke_actor: Path):
    args = ViewerLaunchArgs(
        checkpoint=smoke_actor,
        config=SMOKE,
        cache_dir=smoke_cache,
        suite="solo",
        seed=0,
        device="cpu",
        track="oval",
        open_browser=False,
    )
    replay = CheckpointReplay.from_launch_args(args)
    assert replay.num_cars == 1
    replay.set_suite("head_to_head")
    assert replay.suite == "head_to_head"
    assert replay.num_cars == 2
    for seed in (0, 1, 7, 13):
        replay.seed = seed
        poses = replay.reset()
        assert poses.shape[0] == 2
        assert int(np.sum(poses[:, 3] > 0.5)) == 2
    assert "oval" in replay.available_tracks


def test_dense_environment_count_is_distinct_from_cars_per_environment(
    smoke_cache: Path, smoke_actor: Path
):
    replay = CheckpointReplay.from_launch_args(
        ViewerLaunchArgs(
            checkpoint=smoke_actor,
            config=SMOKE,
            cache_dir=smoke_cache,
            suite="dense",
            seed=0,
            device="cpu",
            track="oval",
            open_browser=False,
        )
    )
    assert replay.num_environments == 1
    assert replay.cars_per_environment == 2
    replay.set_environment_count(3)
    assert replay.num_environments == 3
    assert replay.cars_per_environment == 2
    assert replay.num_cars == 6
    assert replay.poses().shape == (6, 12)
    with pytest.raises(ViewerError, match="between 1 and 4"):
        replay.set_environment_count(5)
    replay.set_suite("solo")
    assert replay.num_environments == 1
    assert replay.num_cars == 1
    with pytest.raises(ViewerError, match="only be changed in dense"):
        replay.set_environment_count(2)


def test_checkpoint_config_mismatch(smoke_cache: Path, tmp_path: Path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    payload = build_actor_artifact_payload(cfg, actor)
    payload["actor_architecture"]["gru_hidden_dim"] = 7
    bad = tmp_path / "bad_actor.pt"
    torch.save(payload, bad)
    args = ViewerLaunchArgs(
        checkpoint=bad,
        config=SMOKE,
        cache_dir=smoke_cache,
        suite="solo",
        seed=0,
        device="cpu",
        open_browser=False,
    )
    with pytest.raises(ViewerError, match="mismatch on gru_hidden_dim"):
        CheckpointReplay.from_launch_args(args)


def test_stationary_obstacle_placement_and_policy_exclusion(
    smoke_cache: Path, smoke_actor: Path
):
    replay = CheckpointReplay.from_launch_args(
        ViewerLaunchArgs(
            checkpoint=smoke_actor,
            config=SMOKE,
            cache_dir=smoke_cache,
            suite="head_to_head",
            seed=7,
            device="cpu",
            track="oval",
            open_browser=False,
        )
    )
    replay.set_obstacle_preset("heavy")
    first = replay.obstacle_poses().copy()
    racers = replay.poses()[:, :3].copy()
    assert first.shape == (4, 3)
    assert replay.hello_fields()["num_cars"] == 2
    assert replay.hello_fields()["num_obstacles"] == 4
    t = replay.sim.buffers.torch_arrays
    assert int(t.active.sum()) == 6
    assert int(t.trainable.sum()) == 2
    actions = replay._policy_actions()
    assert torch.all(
        actions.index_select(0, replay._obstacle_indices())
        == torch.tensor([-1.0, 0.0])
    )

    diagonal = math.hypot(replay.car_length, replay.car_width)
    for i, pose in enumerate(first):
        assert all(
            np.linalg.norm(pose[:2] - other[:2]) >= 3.0 * diagonal
            for other in first[:i]
        )
        assert all(
            np.linalg.norm(pose[:2] - racer[:2]) >= 4.0 * diagonal
            for racer in racers
        )
    obstacle_slots = replay._obstacle_indices()
    assert torch.all(t.boundary_distance.index_select(0, obstacle_slots) > 0.14)
    obstacle_specs = [pose for _, pose in replay._fixed_obstacles]
    assert any(abs(float(pose["ey"])) < 1e-5 for pose in obstacle_specs)
    track_length = float(replay.sim.geom.track_length[0])
    min_arc = max(
        4.0 * replay.car_length,
        track_length / (2.5 * replay.obstacles_per_environment),
    )
    for i, pose in enumerate(obstacle_specs):
        for other in obstacle_specs[:i]:
            arc = abs(float(pose["s"]) - float(other["s"]))
            assert min(arc, track_length - arc) >= min_arc

    replay.reset()
    assert np.allclose(replay.obstacle_poses(), first)
    replay.respawn_obstacles()
    second = replay.obstacle_poses().copy()
    assert replay.obstacle_nonce == 1
    assert second.shape == first.shape
    assert not np.array_equal(second, first)
    replay.reset()
    assert np.array_equal(replay.obstacle_poses(), second)
    for _ in range(4):
        replay.step()
    assert np.array_equal(replay.obstacle_poses(), second)


def test_stationary_obstacle_lidar_and_contact(
    smoke_cache: Path, smoke_actor: Path
):
    replay = CheckpointReplay.from_launch_args(
        ViewerLaunchArgs(
            checkpoint=smoke_actor,
            config=SMOKE,
            cache_dir=smoke_cache,
            suite="solo",
            seed=3,
            device="cpu",
            track="oval",
            open_browser=False,
        )
    )
    replay.set_obstacle_preset("light")
    t = replay.sim.buffers.torch_arrays
    racer = int(replay._racer_indices()[0])
    obstacle = int(replay._obstacle_indices()[0])
    ox = float(t.x[obstacle])
    oy = float(t.y[obstacle])
    yaw = float(t.yaw[obstacle])
    t.x[racer] = ox - 2.0 * math.cos(yaw)
    t.y[racer] = oy - 2.0 * math.sin(yaw)
    t.yaw[racer] = yaw
    t.vx[racer] = 0.0
    t.vy[racer] = 0.0
    t.lidar_range_noise_std[racer] = 0.0
    t.lidar_dropout_prob[racer] = 0.0
    t.lidar_far_dropout_prob[racer] = 0.0
    t.lidar_angle_bias[racer] = 0.0
    t.lidar_extrinsic_x[racer] = 0.0
    t.lidar_extrinsic_y[racer] = 0.0
    t.lidar_extrinsic_yaw[racer] = 0.0
    t.lidar_sector_width[racer] = 0
    replay.sim.rebuild_sensors()
    assert 1.0 < float(t.sensor_obs[racer, LIDAR_DIM // 2]) < 2.0

    static_before = replay.obstacle_poses().copy()
    t.x[racer] = ox - 0.4 * replay.car_length * math.cos(yaw)
    t.y[racer] = oy - 0.4 * replay.car_length * math.sin(yaw)
    t.prev_x[racer] = t.x[racer]
    t.prev_y[racer] = t.y[racer]
    t.yaw[racer] = yaw
    t.vx[racer] = 1.0
    actions = replay._policy_actions()
    out = replay._step_simulator(actions)
    assert bool(out["contact"][racer])
    assert np.array_equal(replay.obstacle_poses(), static_before)
    for _ in range(20):
        replay._step_simulator(replay._policy_actions())
    assert np.array_equal(replay.obstacle_poses(), static_before)
    assert torch.count_nonzero(t.vx.index_select(0, replay._obstacle_indices())) == 0
    assert torch.all(t.applied_effort.index_select(0, replay._obstacle_indices()) == -1.0)


def test_collision_contact_channel_and_hot_swap_rollback(
    smoke_cache: Path, smoke_actor: Path, tmp_path: Path
):
    replay = CheckpointReplay.from_launch_args(
        ViewerLaunchArgs(
            checkpoint=smoke_actor,
            config=SMOKE,
            cache_dir=smoke_cache,
            suite="solo",
            seed=0,
            device="cpu",
            track="oval",
            open_browser=False,
        )
    )
    racer = int(replay._racer_indices()[0])
    t = replay.sim.buffers.torch_arrays
    t.contact[racer] = 1
    assert replay.poses()[0, 11] == 1.0
    t.contact[racer] = 0
    t.wall_contact[racer] = 1
    assert replay.poses()[0, 11] == 1.0

    cfg = load_config(SMOKE)
    replacement = tmp_path / "replacement.pt"
    export_actor_artifact(cfg, build_actor(cfg), str(replacement))
    old_actor = replay.actor
    assert replay._hidden is not None
    replay._hidden.fill_(1.0)
    actor, condition, resolved = replay.load_actor_candidate(replacement)
    replay.install_actor(actor, condition, resolved)
    assert condition is None
    assert replay.actor is actor
    assert replay.actor is not old_actor
    assert replay.checkpoint_label == replacement.name
    assert replay.checkpoint_path == str(replacement.resolve())
    assert replay._hidden is not None
    assert torch.count_nonzero(replay._hidden) == 0

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    installed = replay.actor
    with pytest.raises(Exception):
        replay.load_actor_candidate(corrupt)
    assert replay.actor is installed

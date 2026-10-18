"""CPU tests for GT Sophy multi-scenario spawn, ids, and speed classes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from f1tenth_env.kernel import MAX_OPPONENTS, OBS_DIM, OBS_OPPONENT_START
from f1tenth_env.utils import (
    EGO_PACE_MPS,
    GIGAFLOW_BLOCKED_TRACKS,
    GIGAFLOW_PUBLIC_TRACKS,
    SOPHY_SCENARIO_IDS,
    SOPHY_SCENARIO_NAMES,
    SPEED_CLASS_FASTER,
    SPEED_CLASS_SAME,
    SPEED_CLASS_SLOWER,
    SPEED_CLASS_WEIGHTS,
    draw_opponent_speed_class,
    hardest_corner_s,
)
from ppo import stratified_env_permutation
from standalone_trainer import (
    build_config,
    build_env_cfg,
    load_config_patch,
    parse_args,
    resolve_training_scenarios,
    validate_config_patch,
)

try:
    import warp  # noqa: F401
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

_TRAINING_ROOT = Path(__file__).resolve().parents[1]
_SOPHY_CFG = _TRAINING_ROOT / "configs" / "sophy_multi_scenario.json"
_SMOKE_TRACK = "Oschersleben"
_V2_DRIVE = 21.46877492171339
_V2_BRAKE = 9.139188067139935
_V2_STEER = 1.0511988660159353
_V2_COAST = 4.804781294740983
_V2_LAT = 5.329306089973615


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return args, build_config(args, patch=patch, explicit=explicit)


def test_sophy_config_uses_v2_plant_and_public_tracks():
    path = _SOPHY_CFG
    patch = json.loads(path.read_text())
    validate_config_patch(patch)
    args, explicit = parse_args([])
    cfg = build_config(args, patch=patch, explicit=explicit)
    env = cfg["env"]
    assert env["gigaflow_track_sampling"] is True
    assert env["episode_length"] == pytest.approx(150.0)
    assert env["f_drive_max"] == pytest.approx(_V2_DRIVE)
    assert env["f_brake_max"] == pytest.approx(_V2_BRAKE)
    assert env["steer_gain"] == pytest.approx(_V2_STEER)
    assert env["warp_sim"]["c_coast"] == pytest.approx(_V2_COAST)
    assert env["warp_sim"]["lat_stiffness_scale"] == pytest.approx(_V2_LAT)
    assert env["ego_pace_mps"] == pytest.approx(EGO_PACE_MPS)
    assert env["v2_policy_ckpt"] is None
    assert cfg["obs"]["enable_opponent_obs"] is True
    assert cfg["obs"]["opponent_obs_dim"] == 8
    assert OBS_DIM - OBS_OPPONENT_START == 8
    names = [scenario["name"] for scenario in cfg["scenarios"]]
    assert tuple(names) == SOPHY_SCENARIO_NAMES
    blob = path.read_text()
    for blocked in GIGAFLOW_BLOCKED_TRACKS:
        assert blocked not in blob
        assert blocked not in json.dumps(cfg["scenarios"])
    assert len(GIGAFLOW_PUBLIC_TRACKS) == 23


def test_sophy_scenarios_resolve_ids_and_counts():
    patch, _meta = load_config_patch(str(_SOPHY_CFG))
    _args, cfg = _resolve(["--num-envs", "7", "--config", str(_SOPHY_CFG)], patch=patch)
    resolved = resolve_training_scenarios(cfg, 7, str(_TRAINING_ROOT))
    assert [spec["name"] for spec in resolved] == list(SOPHY_SCENARIO_NAMES)
    assert [spec["num_envs"] for spec in resolved] == [1] * 7
    for spec in resolved:
        name = spec["name"]
        env = spec["cfg"]["env"]
        assert env["scenario_id"] == SOPHY_SCENARIO_IDS[name]
        if name == "1v0":
            assert env["opponent_strategy"] is None
            assert env["num_opponents"] == 0
            assert env["spawn_scene"] == "nearby"
        elif name == "grid":
            assert env["opponent_strategy"] == "scripted"
            assert env["num_opponents"] == 7
            assert env["spawn_scene"] == "grid"
        elif name == "corner":
            assert env["opponent_strategy"] == "scripted"
            assert env["num_opponents"] == 2
            assert env["spawn_scene"] == "corner"
        else:
            assert env["opponent_strategy"] == "scripted"
            assert env["num_opponents"] == int(name.split("v")[1])
            assert env["spawn_scene"] == "nearby"


def test_stratified_env_permutation_keeps_scenario_share():
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.int32)
    torch.manual_seed(0)
    perm = stratified_env_permutation(ids)
    assert sorted(perm.tolist()) == list(range(8))
    mix = ids[perm].tolist()
    first = mix[:4]
    second = mix[4:]
    assert first.count(0) == 2 and second.count(0) == 2
    assert first.count(1) == 1 and second.count(1) == 1
    assert first.count(2) == 1 and second.count(2) == 1


def test_draw_opponent_speed_class_mix():
    rng = np.random.default_rng(0)
    counts = {SPEED_CLASS_SLOWER: 0, SPEED_CLASS_SAME: 0, SPEED_CLASS_FASTER: 0}
    hold = 4.0
    pace = EGO_PACE_MPS
    n = 2000
    for _ in range(n):
        cls, speed, mode = draw_opponent_speed_class(rng, pace, hold)
        counts[cls] += 1
        if cls == SPEED_CLASS_SLOWER:
            assert 0.55 * pace <= speed <= 0.85 * pace
        elif cls == SPEED_CLASS_SAME:
            assert 0.90 * pace <= speed <= 1.10 * pace
        else:
            assert 1.15 * pace <= speed <= 1.40 * pace
        assert mode == int(speed > hold)
    assert counts[SPEED_CLASS_SLOWER] / n == pytest.approx(SPEED_CLASS_WEIGHTS[0], abs=0.05)
    assert counts[SPEED_CLASS_SAME] / n == pytest.approx(SPEED_CLASS_WEIGHTS[1], abs=0.05)
    assert counts[SPEED_CLASS_FASTER] / n == pytest.approx(SPEED_CLASS_WEIGHTS[2], abs=0.05)


def test_hardest_corner_s_is_on_loop():
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    circle = np.stack((np.cos(t), np.sin(t)), axis=1)
    s = hardest_corner_s(circle)
    assert 0.0 <= s < 2.0 * np.pi + 1.0e-6


def _smoke_cfg():
    patch, _meta = load_config_patch(str(_SOPHY_CFG))
    _args, cfg = _resolve(
        ["--num-envs", "7", "--device", "cpu", "--config", str(_SOPHY_CFG)],
        patch=patch,
    )
    cfg["env"]["gigaflow_track_sampling"] = False
    cfg["env"]["track"] = _SMOKE_TRACK
    cfg["env"]["episode_length"] = 8.0
    cfg["env"]["domain_randomization"]["enabled"] = False
    return cfg


def _open_scenario(cfg, spec, num_envs=1):
    from f1tenth_env import F1tenthEnv

    env_cfg = build_env_cfg(
        spec["cfg"],
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": num_envs},
    )
    env_cfg["gigaflow_track_sampling"] = False
    env_cfg["track"] = _SMOKE_TRACK
    env_cfg["episode_length"] = 8.0
    env_cfg["domain_randomization"] = {
        **env_cfg["domain_randomization"],
        "enabled": False,
    }
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_sophy_cpu_smoke_resets_each_scene(warp_runtime):
    del warp_runtime
    from f1tenth_env import runtime as rt

    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = _smoke_cfg()
    resolved = resolve_training_scenarios(cfg, 7, str(_TRAINING_ROOT))
    expected_count = {
        "1v0": 0,
        "1v1": 1,
        "1v2": 2,
        "1v3": 3,
        "1v7": 7,
        "grid": 7,
        "corner": 2,
    }
    sampled_classes = []
    try:
        for spec in resolved:
            env = _open_scenario(cfg, spec)
            try:
                obs, extras = env.reset(seed=0)
                name = spec["name"]
                sid = SOPHY_SCENARIO_IDS[name]
                assert extras["scenario_id"].tolist() == [sid]
                assert int(extras["opponent_count"][0]) == expected_count[name]
                assert obs.shape[-1] == OBS_DIM
                opp = obs[0, OBS_OPPONENT_START:]
                assert opp.numel() == 8
                if name == "1v0":
                    assert torch.all(opp == 0)
                    assert not env.has_opponent
                else:
                    assert env.has_opponent
                    classes = extras["slot_class"][0, : expected_count[name]]
                    speeds = extras["slot_target_speed"][0, : expected_count[name]]
                    modes = extras["slot_mode"][0, : expected_count[name]]
                    sampled_classes.extend(int(v) for v in classes.tolist())
                    pace = max(float(env._reset_params.ego_pace), 3.0)
                    hold = float(env._reset_params.scripted_hold_speed)
                    for cls, speed, mode in zip(
                        classes.tolist(), speeds.tolist(), modes.tolist()
                    ):
                        assert cls in (
                            SPEED_CLASS_SLOWER,
                            SPEED_CLASS_SAME,
                            SPEED_CLASS_FASTER,
                        ), (name, cls, speed)
                        if cls == SPEED_CLASS_SLOWER:
                            lo, hi = 0.55, 0.85
                        elif cls == SPEED_CLASS_SAME:
                            lo, hi = 0.90, 1.10
                        else:
                            lo, hi = 1.15, 1.40
                        assert lo * pace <= float(speed) <= hi * pace + 1.0e-5, (
                            name,
                            cls,
                            speed,
                            pace,
                            hold,
                            float(env._reset_params.use_speed_classes),
                            float(env._reset_params.spawn_scene),
                            float(env._reset_params.num_opponents),
                        )
                        assert int(mode) == int(float(speed) > hold), (
                            name,
                            mode,
                            speed,
                            hold,
                        )
                    assert extras["v2_policy_ckpt"] is None
            finally:
                env.close()
    finally:
        pass
    assert sampled_classes
    assert set(sampled_classes) <= {
        SPEED_CLASS_SLOWER,
        SPEED_CLASS_SAME,
        SPEED_CLASS_FASTER,
    }


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_sophy_speed_class_mix_on_reset(warp_runtime):
    del warp_runtime
    from f1tenth_env import runtime as rt

    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = _smoke_cfg()
    resolved = resolve_training_scenarios(cfg, 7, str(_TRAINING_ROOT))
    spec = next(row for row in resolved if row["name"] == "1v1")
    env = _open_scenario(cfg, spec, num_envs=32)
    try:
        _obs, extras = env.reset(seed=1)
        classes = extras["slot_class"][:, 0]
        counts = torch.bincount(classes.clamp(min=0), minlength=3).tolist()
        total = sum(counts)
        assert total == 32
        assert counts[SPEED_CLASS_SLOWER] >= 1
        assert counts[SPEED_CLASS_SAME] >= 1
        assert counts[SPEED_CLASS_FASTER] >= 1
        assert extras["scenario_id"].unique().tolist() == [SOPHY_SCENARIO_IDS["1v1"]]
        assert extras["slot_class"].shape == (32, MAX_OPPONENTS)
    finally:
        env.close()

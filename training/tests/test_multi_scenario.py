"""Real tests for GT Sophy-style multi-scenario mixing in the standalone trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env.utils import episode_length_for_track
from standalone_trainer import (
    RunningStats,
    accumulate_scenario_diagnostics,
    allocate_env_counts,
    apply_scenario_patch,
    build_config,
    build_env_cfg,
    cat_nested,
    collect_sensor_step,
    load_config_patch,
    parse_args,
    resolve_training_scenarios,
    update_episode_lap_counts,
)

try:
    import warp  # noqa: F401
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

_MULTI_CFG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "courtyard_2_e2e_ppo_multi_scenario.json"
)
_TRAINING_ROOT = str(Path(__file__).resolve().parents[1])


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return args, build_config(args, patch=patch, explicit=explicit)


def test_allocate_env_counts_largest_remainder():
    assert allocate_env_counts([0.5, 0.25, 0.25], 4) == [2, 1, 1]
    assert allocate_env_counts([0.5, 0.25, 0.25], 8) == [4, 2, 2]
    assert allocate_env_counts([0.5, 0.25, 0.25], 1024) == [512, 256, 256]
    assert allocate_env_counts([0.5, 0.25, 0.25], 3) == [1, 1, 1]


def test_allocate_env_counts_rejects_too_few_rows():
    with pytest.raises(ValueError, match="must be >= number of scenarios"):
        allocate_env_counts([1.0, 1.0], 1)


def test_cat_nested_concatenates_extras_along_env_dim():
    extras_a = {
        "termination": {
            "out_of_bounds": torch.tensor([True, False]),
            "time_out": torch.tensor([False, False]),
        },
        "metrics": {"lap_count": torch.tensor([1, 0], dtype=torch.int32)},
    }
    extras_b = {
        "termination": {
            "out_of_bounds": torch.tensor([False]),
            "time_out": torch.tensor([True]),
        },
        "metrics": {"lap_count": torch.tensor([2], dtype=torch.int32)},
    }
    cat = cat_nested([extras_a, extras_b])
    assert cat["metrics"]["lap_count"].tolist() == [1, 0, 2]
    assert cat["termination"]["out_of_bounds"].tolist() == [True, False, False]
    assert cat["termination"]["time_out"].tolist() == [False, False, True]


def test_update_episode_lap_counts_keeps_pre_reset_value_on_done():
    prev = torch.tensor([0.0, 2.0, 1.0])
    lap_count = torch.tensor([1, 0, 3], dtype=torch.int32)
    done = torch.tensor([False, True, False])
    out = update_episode_lap_counts(prev, lap_count, done)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_accumulate_scenario_diagnostics_rates():
    diag = RunningStats()
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    laps = torch.tensor([1.0, 0.0, 2.0, 0.0])
    done = torch.tensor([True, True, False, True])
    extras = {
        "termination": {
            "out_of_bounds": torch.tensor([True, False, False, False]),
            "not_moving": torch.tensor([False, False, False, True]),
            "invalid_state": torch.tensor([False, False, False, False]),
            "collision": torch.tensor([False, False, False, False]),
        }
    }
    accumulate_scenario_diagnostics(
        diag,
        ["flying", "standing"],
        [2, 2],
        rewards,
        laps,
        done,
        extras,
    )
    assert diag.total("scn/flying/episodes") == 2.0
    assert diag.mean("scn/flying/episode_return") == pytest.approx(1.5)
    assert diag.mean("scn/flying/lap_completion") == pytest.approx(0.5)
    assert diag.mean("scn/flying/crash") == pytest.approx(0.5)
    assert diag.total("scn/standing/episodes") == 1.0
    assert diag.mean("scn/standing/episode_return") == pytest.approx(4.0)
    assert diag.mean("scn/standing/lap_completion") == pytest.approx(0.0)
    assert diag.mean("scn/standing/crash") == pytest.approx(1.0)


def test_shipped_multi_scenario_config_resolves():
    patch, _meta = load_config_patch(str(_MULTI_CFG))
    args, cfg = _resolve(["--num-envs", "8", "--config", str(_MULTI_CFG)], patch=patch)
    del args
    assert cfg["algorithm"] == "ppo"
    assert cfg["env"]["opponent_strategy"] is None
    names = [scenario["name"] for scenario in cfg["scenarios"]]
    assert names == ["solo_flying", "solo_standing", "solo_iv_2026"]
    resolved = resolve_training_scenarios(cfg, 8, _TRAINING_ROOT)
    assert [spec["num_envs"] for spec in resolved] == [4, 2, 2]
    assert resolved[0]["cfg"]["env"]["reset_stationary_probability"] == 0.0
    assert resolved[1]["cfg"]["env"]["reset_stationary_probability"] == 1.0
    assert resolved[2]["cfg"]["env"]["track"] == "IV_2026_SIM"
    assert resolved[2]["cfg"]["sensor"]["lidar_map_yaml"] is None
    assert resolved[0]["cfg"]["sensor"]["lidar_map_yaml"] == "assets/courtyard_2.yaml"
    assert resolved[0]["cfg"]["env"]["episode_length"] == pytest.approx(
        episode_length_for_track("courtyard_2", _TRAINING_ROOT)
    )
    assert resolved[2]["cfg"]["env"]["episode_length"] == pytest.approx(
        episode_length_for_track("IV_2026_SIM", _TRAINING_ROOT)
    )


def test_scenario_cannot_change_control_interval():
    cfg = dict(DEFAULT_CONFIG)
    scenario = {
        "name": "bad_dt",
        "weight": 1.0,
        "env": {"control_interval": 20},
    }
    with pytest.raises(ValueError, match="cannot change env.control_interval"):
        apply_scenario_patch(cfg, scenario, _TRAINING_ROOT)


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_scenario_envs_apply_standing_and_flying_resets(warp_runtime):
    del warp_runtime
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt

    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    patch, _meta = load_config_patch(str(_MULTI_CFG))
    args, cfg = _resolve(
        ["--num-envs", "8", "--device", "cpu"],
        patch=patch,
    )
    resolved = resolve_training_scenarios(cfg, args.num_envs, _TRAINING_ROOT)
    flying = resolved[0]
    standing = resolved[1]
    envs = []
    try:
        for spec, n_envs in ((flying, 4), (standing, 4)):
            env = F1tenthEnv(
                num_envs=n_envs,
                env_cfg=build_env_cfg(
                    spec["cfg"],
                    launch_strategy="uniform_jittered",
                    launch_strategy_data={"num_cars": n_envs},
                ),
                obs_cfg=cfg["obs"],
                reward_cfg=cfg["reward"],
            )
            envs.append(env)
        envs[0].reset(seed=0)
        envs[1].reset(seed=0)
        standing_speed = envs[1].read_state()["base_lin_vel"][:, 0]
        assert float(envs[0]._reset_params.stationary_probability) == 0.0
        assert float(envs[1]._reset_params.stationary_probability) == 1.0
        assert torch.all(standing_speed == 0.0)
        assert not envs[0].has_opponent
        assert not envs[1].has_opponent
    finally:
        for env in envs:
            env.close()


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_collect_sensor_step_concatenates_two_real_envs(warp_runtime):
    del warp_runtime
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt

    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    patch = {
        "env": {
            "episode_length": 8.0,
            "opponent_strategy": None,
            "domain_randomization": {"enabled": False},
        }
    }
    args, cfg = _resolve(["--num-envs", "3", "--device", "cpu"], patch=patch)
    del args
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": 2},
    )
    env_cfg_b = copy_env_cfg_with_cars(cfg, 1)
    envs = [
        F1tenthEnv(
            num_envs=2,
            env_cfg=env_cfg,
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        ),
        F1tenthEnv(
            num_envs=1,
            env_cfg=env_cfg_b,
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        ),
    ]
    try:
        for env in envs:
            env.reset(seed=0, with_sensors=True)
        actions = torch.zeros(3, 2, dtype=torch.float32)
        actor_obs, critic_obs, reward, done, extras = collect_sensor_step(
            envs, actions, [2, 1], envs[0].control_interval
        )
        assert actor_obs.shape == (3, cfg["obs"]["num_actor_obs"])
        assert critic_obs.shape == (3, cfg["obs"]["num_obs"])
        assert reward.shape == (3,)
        assert done.shape == (3,)
        assert extras["metrics"]["lap_count"].shape == (3,)
        assert extras["termination"]["out_of_bounds"].shape == (3,)
        assert extras["observations"]["terminal_critic"].shape[0] == 3
    finally:
        for env in envs:
            env.close()


def copy_env_cfg_with_cars(cfg, num_cars):
    return build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": num_cars},
    )

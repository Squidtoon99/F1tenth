"""Config load/validation tests (stdlib + PyYAML only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gigaflow_f1tenth.config import (
    ACTION_DIM,
    LIDAR_DIM,
    PROPRIO_DIM,
    SENSOR_OBS_DIM,
    ConfigError,
    config_from_dict,
    episode_seconds_for_track_length,
    episode_steps,
    estimate_memory_bytes,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_layout_constants():
    assert SENSOR_OBS_DIM == LIDAR_DIM + PROPRIO_DIM == 1097
    assert ACTION_DIM == 2


def test_load_default_and_smoke():
    default_cfg = load_config(DEFAULT)
    smoke_cfg = load_config(SMOKE)
    assert default_cfg.config_version == 1
    assert smoke_cfg.worlds.max_agents_per_world == 2
    assert smoke_cfg.tracks.num_tracks == 1
    assert smoke_cfg.evaluation.viz_enabled is True
    assert smoke_cfg.evaluation.viz_fps > 0
    assert smoke_cfg.evaluation.viz_max_frames > 0
    assert smoke_cfg.evaluation.device is None
    assert smoke_cfg.evaluation.num_worlds is None
    assert estimate_memory_bytes(smoke_cfg) <= smoke_cfg.profiling.estimate_bytes_budget


def test_evaluation_device_and_num_worlds_validation():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["evaluation"]["device"] = "tpu"
    with pytest.raises(ConfigError, match="evaluation.device"):
        config_from_dict(raw)
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["evaluation"]["num_worlds"] = 0
    with pytest.raises(ConfigError, match="evaluation.num_worlds"):
        config_from_dict(raw)


def test_control_cadence_invariant():
    cfg = load_config(SMOKE)
    assert cfg.agents.control_interval * cfg.agents.sim_dt == pytest.approx(
        1.0 / cfg.agents.control_hz
    )


def test_episode_horizon_clamp():
    cfg = load_config(SMOKE)
    # Short track hits the configured floor.
    assert episode_seconds_for_track_length(cfg, 10.0) == cfg.agents.episode_seconds_min
    # Long track hits the ceiling.
    assert episode_seconds_for_track_length(cfg, 1.0e6) == cfg.agents.episode_seconds_max
    steps = episode_steps(cfg, 10.0)
    assert steps == int(cfg.agents.episode_seconds_min * cfg.agents.control_hz)


def test_rejects_sensor_layout_drift():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["agents"]["lidar_dim"] = 1080
    with pytest.raises(ConfigError, match="sensor layout"):
        config_from_dict(raw)


def test_rejects_value_clip():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["ppo"]["value_clip"] = True
    with pytest.raises(ConfigError, match="value_clip"):
        config_from_dict(raw)


def test_rejects_memory_over_budget():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["worlds"]["num_worlds"] = 10_000
    raw["worlds"]["max_agents_per_world"] = 32
    raw["ppo"]["rollout_length"] = 256
    raw["ppo"]["minibatch_size"] = 128
    raw["profiling"]["estimate_bytes_budget"] = 1_000_000
    with pytest.raises(ConfigError, match="estimated memory"):
        config_from_dict(raw)


def test_rejects_minibatch_larger_than_rollout():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["ppo"]["minibatch_size"] = 10_000_000
    with pytest.raises(ConfigError, match="minibatch_size"):
        config_from_dict(raw)


def test_rejects_unknown_keys():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["ppo"]["not_a_real_key"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_dict(raw)


@pytest.mark.parametrize(
    "name", ["production_a100", "production_h100", "production_rtx4080"]
)
def test_production_configs_use_scale_adjusted_ppo(name: str):
    """Production runs must not reintroduce KL early-stop or the paper's 5e-4.

    The KL check runs after ``optimizer.step()``, so a nonzero target reports a
    destructive update rather than preventing one; 5e-4 collapsed this recurrent
    setup at both 32768 and 2048 minibatches.
    """
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.ppo.target_kl == 0.0
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-4)
    assert cfg.ppo.num_epochs == 3
    assert cfg.ppo.rollout_length == 128
    assert cfg.ppo.clip_ratio == pytest.approx(0.2)
    assert cfg.ppo.ent_coef == pytest.approx(0.01)
    assert cfg.ppo.vf_coef == pytest.approx(0.5)
    assert cfg.ppo.max_grad_norm == pytest.approx(0.5)
    assert cfg.ppo.gamma == pytest.approx(0.999)
    assert cfg.ppo.gae_lambda == pytest.approx(0.95)
    assert cfg.ppo.adaptive_filter_enabled is True
    assert cfg.ppo.adaptive_filter_beta == pytest.approx(0.25)
    assert cfg.ppo.adaptive_filter_eta_scale == pytest.approx(0.01)
    assert cfg.ppo.value_clip is False
    assert cfg.evaluation.device == "cpu"


def test_ppo_defaults_disable_kl_early_stop():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["ppo"].pop("target_kl", None)
    raw["ppo"].pop("learning_rate", None)
    cfg = config_from_dict(raw)
    assert cfg.ppo.target_kl == 0.0
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-4)


def test_entropy_schedule_rejects_legacy_coefficient_conflict():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["ppo"].update(
        {
            "ent_coef_initial": 0.005,
            "ent_coef_final": 0.0005,
            "ent_anneal_updates": 2000,
        }
    )
    with pytest.raises(ConfigError, match="legacy fixed coefficient"):
        config_from_dict(raw)


def test_robustness_config_declares_sentinel_and_actor_safety():
    cfg = load_config(ROOT / "configs" / "robustness_rtxpro6000_6h.yaml")
    assert cfg.ppo.ent_coef is None
    assert cfg.ppo.ent_coef_initial == pytest.approx(0.005)
    assert cfg.ppo.ent_coef_final == pytest.approx(0.0005)
    assert cfg.ppo.ent_anneal_updates == 2000
    assert cfg.ppo.actor_kl_soft == pytest.approx(0.02)
    assert cfg.ppo.actor_kl_hard == pytest.approx(0.05)
    assert cfg.ppo.actor_lr_backoff == pytest.approx(0.5)
    assert cfg.ppo.actor_lr_scale_min == pytest.approx(0.125)
    assert cfg.ppo.actor_checkpoint_interval_updates == 100
    assert cfg.ppo.full_checkpoint_interval_updates == 200
    assert cfg.ppo.total_updates == 3200
    assert cfg.evaluation.device == "cpu"
    assert cfg.evaluation.num_worlds == 32
    assert cfg.evaluation.seeds == (0,)
    assert cfg.evaluation.suite == ("solo", "head_to_head", "dense")
    assert cfg.evaluation.soak_steps == 2000
    assert cfg.evaluation.viz_enabled is False
    assert cfg.wandb.eval_interval_updates == 200
    assert cfg.wandb.resume == "never"


def test_smoke_configs_still_exercise_kl_early_stop():
    """Disabling KL early-stop in production must not delete the mechanism."""
    for name in ("smoke", "gpu_smoke", "gpu_reduced_gate"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.ppo.target_kl > 0.0, name


def test_a100_matches_h100_world_shape():
    a100 = load_config(ROOT / "configs" / "production_a100.yaml")
    h100 = load_config(ROOT / "configs" / "production_h100.yaml")
    assert a100.worlds.num_worlds == h100.worlds.num_worlds
    assert a100.worlds.max_agents_per_world == h100.worlds.max_agents_per_world
    assert a100.worlds.static_opponents_per_world == (
        h100.worlds.static_opponents_per_world
    )
    assert a100.ppo.minibatch_size == h100.ppo.minibatch_size
    assert estimate_memory_bytes(a100) == estimate_memory_bytes(h100)
    assert a100.wandb.group == "prod_a100"
    assert "a100" in a100.wandb.tags


def test_wandb_section_defaults_and_optional():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw.pop("wandb", None)
    cfg = config_from_dict(raw)
    assert cfg.wandb.enabled is False
    assert cfg.wandb.project == "f1tenth-gigaflow"
    assert cfg.wandb.resume == "allow"
    smoke = load_config(SMOKE)
    assert smoke.wandb.mode == "offline"
    assert smoke.wandb.tags == ("smoke",)


def test_recalibrated_config_declares_recalibrated_gates():
    cfg = load_config(ROOT / "configs" / "recalibrated_rtxpro6000_6h.yaml")
    assert cfg.ppo.actor_kl_soft == pytest.approx(0.05)
    assert cfg.ppo.actor_kl_hard == pytest.approx(0.20)
    assert cfg.ppo.actor_kl_warmup_updates == 200
    assert cfg.ppo.actor_kl_stop_window_updates == 500
    assert cfg.ppo.actor_kl_stop_window_count == 5
    assert cfg.ppo.actor_lr_recovery_interval_updates == 100
    assert cfg.evaluation.behavior_feasibility_passes == 2
    assert cfg.evaluation.behavior_no_feasibility_stop_updates == 1000
    assert cfg.ppo.total_updates == 2400
    assert cfg.wandb.group == "recalibrated-robust6h"


def test_clean_grace_production_disables_stops_and_keeps_filter():
    cfg = load_config(ROOT / "configs" / "production_rtxpro6000_clean_grace.yaml")
    h100 = load_config(ROOT / "configs" / "production_h100.yaml")
    assert cfg.worlds.num_worlds == h100.worlds.num_worlds
    assert cfg.ppo.minibatch_size == h100.ppo.minibatch_size
    assert cfg.ppo.target_kl == 0.0
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-4)
    assert cfg.ppo.adaptive_filter_eta_scale == pytest.approx(0.005)
    assert cfg.ppo.ent_coef is None
    assert cfg.ppo.ent_coef_initial == pytest.approx(0.005)
    assert cfg.ppo.ent_coef_final == pytest.approx(0.0005)
    assert cfg.ppo.actor_kl_stop_enabled is False
    assert cfg.ppo.actor_kl_soft == 0.0
    assert cfg.ppo.actor_kl_hard == 0.0
    assert cfg.evaluation.quality_stop_enabled is False
    assert cfg.evaluation.device == "cpu"
    assert cfg.wandb.eval_interval_updates == 200
    assert cfg.wandb.group == "clean-grace-long"
    assert cfg.wandb.resume == "never"

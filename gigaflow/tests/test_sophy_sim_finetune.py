"""Bake IRL/Sophy defaults and wrap a Gigaflow actor for Sophy-sim FT."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.artifacts import (
    bake_actor_artifact,
    build_actor_artifact_payload,
    export_baked_actor,
    validate_actor_artifact,
)
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.evaluation import load_actor_from_checkpoint
from gigaflow_f1tenth.irl_identity import (
    IRL_DYNAMICS,
    IRL_MASS_KG,
    IRL_TIRE_MU,
    baked_normalized_condition,
    baked_raw_style,
)
from gigaflow_f1tenth.model import ConditionedLidarGRUActor, actor_shapes, build_actor
from gigaflow_f1tenth.sim.vehicle import F_BRAKE_MAX, F_DRIVE_MAX, VehicleParams
from gigaflow_f1tenth.sophy_compat import (
    BakedConditionActor,
    load_baked_actor,
    refuse_ppo_resume,
)
from gigaflow_f1tenth.sophy_sim_finetune import sophy_env_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_irl_identity_matches_live_plant():
    params = VehicleParams()
    assert IRL_MASS_KG == pytest.approx(params.mass)
    assert IRL_TIRE_MU == pytest.approx(params.tire_mu)
    assert IRL_DYNAMICS["vmax_scale"] == 1.0
    assert IRL_DYNAMICS["drive_scale"] == 1.0
    assert IRL_DYNAMICS["mass_kg"] == pytest.approx(3.444)
    assert F_DRIVE_MAX == pytest.approx(21.46877492171339)
    assert F_BRAKE_MAX == pytest.approx(9.139188067139935)
    style = baked_raw_style()
    assert style.vmax_scale == 1.0
    assert style.mass_kg == pytest.approx(3.444)
    vec = baked_normalized_condition()
    assert len(vec) == 10
    assert vec[8] == pytest.approx(0.0)
    assert vec[9] == pytest.approx(-0.112, abs=1e-3)


def test_bake_actor_artifact_hardcodes_identity_and_strips_adaptation():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    payload = build_actor_artifact_payload(cfg, actor)
    payload["adaptation"] = {"vmax_scale": 1.7}
    payload["condition_schema"]["conservative_raw"]["vmax_scale"] = 1.7
    baked = bake_actor_artifact(payload, actor=actor)
    validate_actor_artifact(baked)
    assert baked["irl_baked"] is True
    assert "adaptation" not in baked
    raw = baked["condition_schema"]["conservative_raw"]
    assert raw["vmax_scale"] == 1.0
    assert raw["drive_scale"] == 1.0
    assert raw["mass_kg"] == pytest.approx(3.444)
    assert baked["deployment_condition_normalized"][8] == pytest.approx(0.0)


def test_load_maxpool64_artifact_from_smoke_cfg(tmp_path):
    cfg = load_config(SMOKE)
    actor = ConditionedLidarGRUActor(
        actor_shapes(cfg), pool_bins=64, pool_mode="max"
    )
    path = tmp_path / "max64.pt"
    export_baked_actor(cfg, actor, str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["actor_architecture"]["pool_bins"] == 64
    assert payload["actor_architecture"]["pool_mode"] == "max"
    restored = load_actor_from_checkpoint(cfg, path, device="cpu")
    assert restored.pool_bins == 64
    assert restored.pool_mode == "max"
    assert restored.encoder.projection.weight.shape[-1] == 64 * 64


def test_refuse_ppo_resume():
    with pytest.raises(ValueError, match="never PPO-resume"):
        refuse_ppo_resume("ppo")
    with pytest.raises(ValueError, match="never PPO-resume"):
        refuse_ppo_resume("ppo-resume")
    refuse_ppo_resume("sophy-kl")


def test_baked_adapter_matches_constant_condition_and_freezes_encoder():
    cfg = load_config(SMOKE)
    actor = ConditionedLidarGRUActor(
        actor_shapes(cfg), pool_bins=64, pool_mode="max"
    )
    wrapped = BakedConditionActor(actor)
    assert wrapped.obs_dim == 1097
    assert wrapped.condition.numel() == 10
    assert wrapped.condition[8].item() == pytest.approx(0.0)
    for param in actor.condition_encoder.parameters():
        assert param.requires_grad is False
    obs = torch.randn(3, 1097)
    hidden = wrapped.initial_hidden(3, device="cpu", dtype=torch.float32)
    actions, logp, next_h = wrapped.step(obs, hidden, deterministic=True)
    cond = wrapped.condition.unsqueeze(0).expand(3, -1)
    out = actor.forward(obs, cond, hidden, deterministic=True)
    assert torch.allclose(actions, out.actions, atol=1e-5)
    assert torch.allclose(logp, out.log_prob, atol=1e-5)
    assert torch.allclose(next_h, out.hidden, atol=1e-5)


def test_load_baked_actor_roundtrip(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    path = tmp_path / "baked.pt"
    export_baked_actor(cfg, actor, str(path))
    restored, payload = load_baked_actor(path, device="cpu")
    assert payload["irl_baked"] is True
    assert payload["irl_identity"]["vmax_scale"] == 1.0
    assert restored.shapes().sensor_obs_dim == 1097


def test_sophy_env_config_disables_condition_sampling():
    cfg = sophy_env_config(track="IV_2026_SIM", smoke=True)
    env_cfg = cfg["env"]
    assert env_cfg["opponent_strategy"] is None
    assert env_cfg["domain_randomization"]["enabled"] is False
    assert env_cfg["f_drive_max"] == pytest.approx(21.46877492171339)
    assert env_cfg["f_brake_max"] == pytest.approx(9.139188067139935)
    assert env_cfg["tire_friction"] == pytest.approx(0.71)
    assert env_cfg["episode_length"] <= 8.0

"""IRL bake-in: freeze embodied condition so on-car inference never samples it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gigaflow_f1tenth.artifacts import (
    IRL_DEFAULT_STYLE,
    bake_irl_artifact,
    bake_irl_condition,
    build_actor_artifact_payload,
    load_irl_actor,
    validate_actor_artifact,
)
from gigaflow_f1tenth.cli import main
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.model import ActorShapes, ConditionedLidarGRUActor, build_actor
from gigaflow_f1tenth.rewards import (
    CONDITION_DIM,
    deployment_style,
    normalize_condition_vector,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_bake_irl_pins_centered_high_collision(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    src = tmp_path / "actor.pt"
    dest = tmp_path / "baked.pt"
    torch.save(build_actor_artifact_payload(cfg, actor), src)
    payload = bake_irl_artifact(str(src), str(dest))
    validate_actor_artifact(payload)
    expected = normalize_condition_vector(
        deployment_style(IRL_DEFAULT_STYLE).raw_vector()
    )
    got = np.asarray(payload["deployment_condition_normalized"], dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
    assert payload["irl_condition_raw"]["vmax_scale"] == 1.0
    assert payload["deployment_style"] == IRL_DEFAULT_STYLE


def test_bake_irl_field_override_vmax_scale():
    cfg = load_config(SMOKE)
    payload = build_actor_artifact_payload(cfg, build_actor(cfg))
    baked = bake_irl_condition(payload, field_overrides={"vmax_scale": 1.0})
    assert baked["irl_condition_raw"]["vmax_scale"] == 1.0
    other = bake_irl_condition(payload, field_overrides={"vmax_scale": 1.5})
    assert not np.allclose(
        baked["deployment_condition_normalized"],
        other["deployment_condition_normalized"],
    )


def test_load_irl_actor_uses_frozen_condition_not_samples(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    src = tmp_path / "src.pt"
    path = tmp_path / "baked.pt"
    torch.save(build_actor_artifact_payload(cfg, actor), src)
    bake_irl_artifact(str(src), str(path))
    loaded, condition, payload = load_irl_actor(str(path), device="cpu")
    assert condition.shape == (1, CONDITION_DIM)
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.random((1, 1097), dtype=np.float32))
    hidden = loaded.initial_hidden(1, "cpu")
    out_a = loaded.forward(obs, condition, hidden, deterministic=True)
    out_b = loaded.forward(obs, condition, hidden, deterministic=True)
    assert torch.allclose(out_a.actions, out_b.actions, atol=1e-6)
    assert float(out_a.actions.abs().max()) <= 1.0 + 1e-5
    expected = torch.as_tensor(
        payload["deployment_condition_normalized"], dtype=torch.float32
    ).view(1, CONDITION_DIM)
    assert torch.allclose(condition, expected)


def test_actor_from_architecture_loads_maxpool64_512(tmp_path):
    shapes = ActorShapes(
        cnn_projection_dim=512,
        cnn_pool_bins=64,
        cnn_pool_mode="max",
    )
    actor = ConditionedLidarGRUActor(shapes)
    cfg = load_config(SMOKE)
    payload = build_actor_artifact_payload(cfg, actor)
    assert payload["actor_architecture"]["pool_mode"] == "max"
    assert payload["actor_architecture"]["pool_bins"] == 64
    assert payload["actor_architecture"]["cnn_projection_dim"] == 512
    path = tmp_path / "maxpool.pt"
    torch.save(payload, path)
    loaded, condition, _ = load_irl_actor(str(path), device="cpu")
    assert loaded.encoder.pool_mode == "max"
    assert loaded.encoder.pool_bins == 64
    rng = np.random.default_rng(1)
    obs = torch.as_tensor(rng.random((1, 1097), dtype=np.float32))
    out = loaded.forward(
        obs, condition, loaded.initial_hidden(1, "cpu"), deterministic=True
    )
    assert out.actions.shape == (1, 2)
    assert torch.isfinite(out.actions).all()


def test_load_irl_actor_infers_maxpool64_from_weights(tmp_path):
    shapes = ActorShapes(
        cnn_projection_dim=512,
        cnn_pool_bins=64,
        cnn_pool_mode="max",
    )
    actor = ConditionedLidarGRUActor(shapes)
    cfg = load_config(SMOKE)
    payload = build_actor_artifact_payload(cfg, actor)
    payload["actor_architecture"] = {
        k: v
        for k, v in payload["actor_architecture"].items()
        if k not in ("pool_bins", "pool_mode", "cnn_pool_bins", "cnn_pool_mode")
    }
    path = tmp_path / "inferred.pt"
    torch.save(payload, path)
    loaded, _, _ = load_irl_actor(str(path), device="cpu")
    assert loaded.encoder.pool_mode == "max"
    assert loaded.encoder.pool_bins == 64
    assert loaded.encoder.projection_dim == 512


def test_bake_irl_cli(tmp_path, capsys):
    cfg = load_config(SMOKE)
    src = tmp_path / "actor.pt"
    dest = tmp_path / "baked.pt"
    torch.save(build_actor_artifact_payload(cfg, build_actor(cfg)), src)
    rc = main(
        [
            "bake-irl",
            "--checkpoint",
            str(src),
            "--output",
            str(dest),
            "--set",
            "vmax_scale=1.0",
        ]
    )
    assert rc == 0
    assert dest.is_file()
    out = capsys.readouterr().out
    assert "centered_high_collision" in out
    assert '"ok": true' in out

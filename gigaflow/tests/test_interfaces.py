"""Stable interface / shape contract tests for parallel phase ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gigaflow_f1tenth import artifacts, buffers, critic, evaluation, kernels, model, ppo
from gigaflow_f1tenth import tracks, trainer
from gigaflow_f1tenth.artifacts import ARTIFACT_FORMAT_VERSION, build_manifest
from gigaflow_f1tenth.buffers import buffer_shapes
from gigaflow_f1tenth.config import PINNED_UPSTREAM_TRACK_COUNT, load_config
from gigaflow_f1tenth.kernels import slot_index, world_slot_layout
from gigaflow_f1tenth.model import architecture_metadata, actor_shapes
from gigaflow_f1tenth.ppo import initial_filter_state

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
PKG = ROOT / "src" / "gigaflow_f1tenth"

FORBIDDEN = (
    "training",
    "f1tenth_env",
    "f1tenth_sim",
    "qrsac",
    "f1tenth_policy",
    "f1tenth_contract",
)


def test_flat_modules_present():
    expected = {
        "config.py",
        "tracks.py",
        "buffers.py",
        "kernels.py",
        "model.py",
        "critic.py",
        "ppo.py",
        "trainer.py",
        "artifacts.py",
        "evaluation.py",
        "cli.py",
    }
    present = {p.name for p in PKG.glob("*.py")}
    assert expected <= present


def test_no_forbidden_runtime_imports():
    hits = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == p or name.startswith(p + ".") for p in FORBIDDEN):
                    hits.append(f"{path.name}:{node.lineno}:{name}")
    assert hits == []


def test_buffer_and_world_shapes():
    cfg = load_config(SMOKE)
    shapes = buffer_shapes(cfg)
    layout = world_slot_layout(cfg)
    assert shapes.num_slots == layout.num_slots == 8
    assert shapes.action_dim == 2
    assert shapes.gru_hidden_dim == cfg.agents.gru_hidden_dim
    assert shapes.state_dim > 0
    assert slot_index(1, 1, 2) == 3


def test_actor_critic_metadata_boundaries():
    cfg = load_config(SMOKE)
    a = architecture_metadata(cfg)
    c = critic.architecture_metadata(cfg)
    assert a["sensor_obs_dim"] == 1097
    assert a["condition_dim"] == cfg.agents.condition_dim
    assert a["gru_hidden_dim"] == cfg.agents.gru_hidden_dim
    assert a["mlp_sizes"] == list(cfg.agents.actor_mlp_sizes)
    assert c["training_only"] is True
    assert c["max_other_agents"] == 1
    assert actor_shapes(cfg).lidar_dim + actor_shapes(cfg).proprio_dim == 1097
    default_cfg = load_config(ROOT / "configs" / "default.yaml")
    da = architecture_metadata(default_cfg)
    assert da["gru_hidden_dim"] == 512
    assert da["mlp_sizes"] == [1024, 1024, 1024]


def test_artifact_manifest_excludes_critic():
    cfg = load_config(SMOKE)
    manifest = build_manifest(cfg)
    assert manifest.format_version == ARTIFACT_FORMAT_VERSION
    assert "critic" not in manifest.actor_architecture
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "actor_architecture": dict(manifest.actor_architecture),
        "critic": {"weights": []},
    }
    with pytest.raises(ValueError, match="must not contain critic"):
        artifacts.validate_actor_artifact(payload)


def test_integrated_builders_are_live():
    cfg = load_config(SMOKE)
    assert callable(tracks.prepare_tracks)
    assert callable(tracks.load_atlas)
    assert callable(tracks.validate_centerline_table)
    sim = kernels.build_simulator(cfg, None, "cpu")
    assert isinstance(sim, kernels.Simulator)
    assert sim.state().layout.num_slots == 8
    assert kernels.simulator_interface_gaps() == ()
    actor = model.build_actor(cfg)
    value_critic = critic.build_critic(cfg)
    assert actor.shapes().sensor_obs_dim == 1097
    assert actor.shapes().action_dim == 2
    assert value_critic.shapes().condition_dim == cfg.agents.condition_dim
    with pytest.raises(TypeError):
        ppo.build_ppo(cfg, None, None)  # type: ignore[arg-type]
    learner = ppo.build_ppo(cfg, actor, value_critic, device="cpu")
    assert learner.state().update_index == 0
    tr = trainer.build_trainer(cfg, device="cpu")
    assert isinstance(tr, trainer.Trainer)
    ev = evaluation.build_evaluator(cfg, device="cpu")
    assert isinstance(ev, evaluation.Evaluator)
    filt = initial_filter_state(cfg)
    assert filt.beta == cfg.ppo.adaptive_filter_beta
    assert filt.eta == 0.0
    assert PINNED_UPSTREAM_TRACK_COUNT == 23


def test_protocol_names_stable():
    assert tracks.TrackAtlas.__name__ == "TrackAtlas"
    assert buffers.RolloutBuffer.__name__ == "RolloutBuffer"
    assert kernels.Simulator.__name__ == "Simulator"
    assert model.ConditionedActor.__name__ == "ConditionedActor"
    assert critic.CentralValueCritic.__name__ == "CentralValueCritic"
    assert ppo.PPOLearner.__name__ == "PPOLearner"
    assert trainer.Trainer.__name__ == "Trainer"
    assert artifacts.ArtifactStore.__name__ == "ArtifactStore"
    assert evaluation.Evaluator.__name__ == "Evaluator"

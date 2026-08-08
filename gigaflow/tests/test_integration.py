"""End-to-end integration: collect → reconstruct → PPO → checkpoint → eval."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gigaflow_f1tenth import tracks as T
from gigaflow_f1tenth.buffers import STATE_INDEX
from gigaflow_f1tenth.config import config_from_dict, config_to_dict, load_config
from gigaflow_f1tenth.critic import critic_values_over_time, pack_critic_features
from gigaflow_f1tenth.evaluation import (
    ablation_config,
    build_evaluator,
    run_evaluation,
    run_soak,
)
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.trainer import (
    CHECKPOINT_VERSION,
    build_trainer,
    run_training,
    track_manifest_hash,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
FIXTURE_TRACKS = ROOT / "tests" / "fixtures" / "tracks"


def _local_track_manifest(cfg, cache_dir: Path, name: str) -> Path:
    """Prepare a real single-track atlas cache from a checked-in fixture."""
    local = cache_dir / "local"
    local.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        FIXTURE_TRACKS / f"{name}_centerline.csv", local / f"{name}_centerline.csv"
    )
    T.prepare_tracks(
        cfg,
        str(cache_dir),
        pin_path=ROOT / "configs" / "track_pin.json",
        lut_resolution=0.5,
        edt_resolution=0.25,
        skip_download=True,
    )
    return cache_dir / T.MANIFEST_FILENAME


def _with_manifest(cfg, manifest: Path):
    raw = config_to_dict(cfg)
    raw["tracks"]["manifest_path"] = str(manifest)
    return config_from_dict(raw)


def test_train_update_and_resume(tmp_path):
    cfg = load_config(SMOKE)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    trainer = build_trainer(cfg, atlas=atlas, device="cpu", run_dir=tmp_path / "run")
    trainer.setup()
    p0 = trainer.train_update()
    assert p0.update_index == 1
    assert p0.transitions > 0
    assert torch.isfinite(
        torch.tensor(
            [
                p0.metrics["policy_loss"],
                p0.metrics["value_loss"],
                p0.metrics["entropy"],
            ]
        )
    ).all()
    normalizer_count = float(trainer.actor.sensor_normalizer.count.item())
    normalizer_mean = trainer.actor.sensor_normalizer.mean.clone()
    assert normalizer_count > 0.0, "collect_rollout must have fed the normalizer"
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt))
    idx = trainer.ppo.update_index
    filt = float(trainer.ppo.filter_state.ewma_max_abs_adv)
    trainer2 = build_trainer(cfg, atlas=atlas, device="cpu")
    trainer2.setup()
    trainer2.load_checkpoint(str(ckpt))
    assert trainer2.ppo.update_index == idx
    assert abs(float(trainer2.ppo.filter_state.ewma_max_abs_adv) - filt) < 1e-8
    # Resume must not silently reset the sensor normalizer's running stats.
    assert float(trainer2.actor.sensor_normalizer.count.item()) == normalizer_count
    assert torch.allclose(trainer2.actor.sensor_normalizer.mean, normalizer_mean)
    p1 = trainer2.train_update()
    assert p1.update_index == idx + 1
    assert (
        float(trainer2.actor.sensor_normalizer.count.item()) > normalizer_count
    ), "a second update must advance statistics for the next rollout"


def test_reconstruction_parity_and_compact_state():
    cfg = load_config(SMOKE)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    trainer = build_trainer(cfg, atlas=atlas, device="cpu")
    trainer.setup()
    batch = trainer.collect_rollout()
    assert batch.state.shape[-1] == trainer.sim.state_dim
    prepared = trainer.reconstruct_prepared(batch, verify_parity=True)
    assert prepared.sensor_obs.shape[-1] == 1097
    # ego_state concatenates the ordered track preview onto the compact state.
    assert (
        prepared.ego_state.shape[-1]
        == trainer.sim.state_dim + T.TRACK_PREVIEW_DIM
    )
    assert trainer.profile["reconstruction_digest_mismatches"] == 0.0


def test_next_state_is_the_pre_respawn_transition_state():
    """Fail-before: bootstrap values were read after respawn overwrote the slot."""
    cfg = load_config(SMOKE)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    trainer = build_trainer(cfg, atlas=atlas, device="cpu")
    trainer.setup()
    batch = trainer.collect_rollout()
    assert batch.next_state.shape == batch.state.shape

    # For a slot that did not respawn, s'_t is exactly the next action-time state.
    t_steps = batch.state.shape[0]
    for step in range(t_steps - 1):
        kept = ~batch.reset_mask[step + 1]
        assert torch.equal(
            batch.next_state[step][kept], batch.state[step + 1][kept]
        )

    prepared = trainer.reconstruct_prepared(batch)
    assert prepared.bootstrap_values is not None
    assert prepared.bootstrap_values.shape == batch.rewards.shape
    num_slots = batch.state.shape[1]
    world_id = torch.arange(num_slots) // cfg.worlds.max_agents_per_world
    world_id = world_id.view(1, num_slots).expand(t_steps, num_slots)
    with torch.no_grad():
        expected = critic_values_over_time(
            trainer.critic,
            *pack_critic_features(
                batch.next_state,
                world_id=world_id,
                active=batch.next_state[..., STATE_INDEX["active"]] > 0.5,
                max_agents_per_world=cfg.worlds.max_agents_per_world,
                track_id=batch.track_id,
                atlas=atlas,
                centralized=bool(cfg.ablations.centralized_critic),
            ),
            batch.condition,
        )
        action_time = critic_values_over_time(
            trainer.critic,
            prepared.ego_state,
            prepared.other_agents,
            prepared.other_mask,
            batch.condition,
        )
    assert torch.allclose(prepared.bootstrap_values, expected, atol=1e-6)
    # The final bootstrap is a next-state value, not the action-time value.
    assert not torch.allclose(
        prepared.bootstrap_values[-1], action_time[-1], atol=1e-4
    )


def test_checkpoint_saves_device_styles_and_rejects_atlas_mismatch(tmp_path):
    """Fail-before: the stale host style list was serialized and reapplied."""
    base = load_config(SMOKE)
    manifest_a = _local_track_manifest(base, tmp_path / "tracks_a", "oval")
    manifest_b = _local_track_manifest(base, tmp_path / "tracks_b", "stadium")
    cfg_a = _with_manifest(base, manifest_a)

    trainer = build_trainer(cfg_a, device="cpu")
    trainer.setup()
    host_styles = np.stack([s.raw_vector() for s in trainer.sim.styles], axis=0)
    device_styles = trainer.sim.raw_styles().detach().cpu().numpy().copy()
    assert not np.allclose(device_styles, host_styles), "host list must be stale"

    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt))
    expected_draw = float(np.random.random())
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert int(payload["checkpoint_version"]) == CHECKPOINT_VERSION
    assert np.allclose(payload["styles_raw"], device_styles)
    assert payload["track_manifest_hash"] == track_manifest_hash(cfg_a)

    resumed = build_trainer(cfg_a, device="cpu")
    resumed.setup()
    resumed.load_checkpoint(str(ckpt))
    assert np.allclose(
        resumed.sim.raw_styles().detach().cpu().numpy(), device_styles
    )
    assert float(np.random.random()) == expected_draw
    progress = resumed.train_update()
    assert 0.0 <= progress.metrics["retention"] <= 1.0
    for key in ("policy_loss", "value_loss", "approx_kl"):
        assert math.isfinite(progress.metrics[key])

    mismatched = build_trainer(_with_manifest(base, manifest_b), device="cpu")
    mismatched.setup()
    with pytest.raises(ValueError, match="track manifest mismatch"):
        mismatched.load_checkpoint(str(ckpt))


def test_run_training_smoke(tmp_path):
    cfg = load_config(SMOKE)
    progress = run_training(
        cfg,
        num_updates=1,
        atlas=make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world),
        device="cpu",
        run_dir=tmp_path / "train",
    )
    assert progress.update_index >= 1
    assert (tmp_path / "train" / "ckpt_final.pt").is_file()
    assert (tmp_path / "train" / "actor_final.pt").is_file()


@pytest.mark.parametrize(
    ("signum", "reason"),
    (
        (signal.SIGINT, "requested_sigint"),
        (signal.SIGTERM, "requested_sigterm"),
    ),
)
def test_signal_writes_final_checkpoint_and_finishes_wandb(
    tmp_path, signum, reason
):
    pytest.importorskip("wandb")
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["ppo"]["total_updates"] = 100
    raw["wandb"].update(
        {
            "enabled": True,
            "mode": "offline",
            "run_id": f"signal-{int(signum)}",
            "resume": "never",
        }
    )
    config_path = tmp_path / f"signal_{int(signum)}.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run_dir = tmp_path / f"run_{int(signum)}"
    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gigaflow_f1tenth",
            "train",
            "--config",
            str(config_path),
            "--num-updates",
            "100",
            "--device",
            "cpu",
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if (run_dir / "metrics_000001.json").is_file():
            break
        if proc.poll() is not None:
            output = proc.communicate()[0]
            pytest.fail(f"trainer exited before signal: {output}")
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("trainer did not commit its first update")
    os.kill(proc.pid, signum)
    output = proc.communicate(timeout=180)[0]
    assert proc.returncode == 0, output
    checkpoint = torch.load(
        run_dir / "ckpt_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["shutdown_reason"] == reason
    assert (run_dir / "actor_final.pt").is_file()
    assert not list(run_dir.glob("ckpt_emergency_*.pt"))
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["shutdown_reason"] == reason
    assert manifest["finished_at"] is not None
    assert list((run_dir / "wandb").glob("offline-run-*"))


def test_evaluation_and_promotion_gates(tmp_path):
    cfg = load_config(SMOKE)
    reports = run_evaluation(
        cfg,
        suites=("solo",),
        atlas=make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world),
        device="cpu",
        output_dir=tmp_path / "eval",
    )
    assert len(reports) == 1
    assert (tmp_path / "eval" / "eval_report.json").is_file()
    assert (tmp_path / "eval" / "solo_seed0_frame.png").is_file()
    assert (tmp_path / "eval" / "solo_seed0.mp4").is_file()
    assert (tmp_path / "eval" / "solo_seed0.gif").is_file()
    gates = build_evaluator(cfg).promotion_gates(reports)
    assert gates["has_reports"]
    assert gates["finite_metrics"]


def test_soak_and_ablation_seam(tmp_path):
    cfg = load_config(SMOKE)
    report = run_soak(
        cfg,
        steps=8,
        atlas=make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world),
        device="cpu",
        output_dir=tmp_path / "soak",
    )
    assert report["ok"] is True
    ab = ablation_config(cfg, adaptive_filter_enabled=False, actor_variant="feedforward")
    assert ab.ppo.adaptive_filter_enabled is False
    assert ab.ablations.actor_variant == "feedforward"

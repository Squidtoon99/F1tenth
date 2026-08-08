"""W&B integration tests using the real wandb library (offline/disabled only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from gigaflow_f1tenth.config import (
    ConfigError,
    config_from_dict,
    load_config,
    replace_wandb_config,
)
from gigaflow_f1tenth.evaluation import EvalMetrics, EvalReport
from gigaflow_f1tenth.trainer import build_trainer, run_training
from gigaflow_f1tenth.wandb_log import (
    EVAL_METRIC_GLOB,
    EVAL_SOURCE_STEP_METRIC,
    RUN_META_FILENAME,
    WandbAuthError,
    WandbSession,
    assert_wandb_auth_for_online,
    build_wandb_session,
    flatten_train_metrics,
    has_wandb_auth,
    read_run_meta,
    summarize_eval_lap_times,
    summarize_rollout_aux,
    wandb_is_active,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"

wandb = pytest.importorskip("wandb")


@pytest.fixture
def smoke_cfg():
    return load_config(SMOKE)


@pytest.fixture
def wandb_home(tmp_path, monkeypatch):
    """Isolate wandb state under tmp_path; never touch network or real credentials."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_dir"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    return home


def test_wandb_defaults_disabled(smoke_cfg):
    assert smoke_cfg.wandb.enabled is False
    assert wandb_is_active(smoke_cfg.wandb) is False


def test_wandb_config_validation_rejects_bad_mode():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["wandb"] = {
        "enabled": True,
        "mode": "cloud",
        "project": "f1tenth-gigaflow",
    }
    with pytest.raises(ConfigError, match="wandb.mode"):
        config_from_dict(raw)


def test_wandb_config_requires_project_when_enabled():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["wandb"] = {"enabled": True, "mode": "offline", "project": "  "}
    with pytest.raises(ConfigError, match="wandb.project"):
        config_from_dict(raw)


def test_replace_wandb_cli_overrides(smoke_cfg):
    cfg = replace_wandb_config(
        smoke_cfg,
        enabled=True,
        mode="offline",
        project="cli-proj",
        entity="ent",
        group="grp",
        name="run-a",
        tags=("a", "b"),
        notes="n",
        run_id="abc123",
        resume="must",
    )
    assert cfg.wandb.enabled is True
    assert cfg.wandb.mode == "offline"
    assert cfg.wandb.project == "cli-proj"
    assert cfg.wandb.entity == "ent"
    assert cfg.wandb.group == "grp"
    assert cfg.wandb.name == "run-a"
    assert cfg.wandb.tags == ("a", "b")
    assert cfg.wandb.notes == "n"
    assert cfg.wandb.run_id == "abc123"
    assert cfg.wandb.resume == "must"


def test_online_auth_missing_fails_clearly(wandb_home):
    assert has_wandb_auth() is False
    with pytest.raises(WandbAuthError, match="no local W&B credentials"):
        assert_wandb_auth_for_online("online")
    assert_wandb_auth_for_online("offline")  # no raise
    assert_wandb_auth_for_online("disabled")


def test_disabled_session_never_inits_run(smoke_cfg, tmp_path, wandb_home):
    cfg = replace_wandb_config(smoke_cfg, enabled=False, mode="online")
    session = build_wandb_session(cfg, run_dir=tmp_path / "run")
    assert session.start() is None
    assert session.active is False
    assert session.run_id is None
    meta = read_run_meta(tmp_path / "run")
    assert meta["wandb"]["enabled"] is False
    session.log_metrics({"ppo/policy_loss": 1.0}, step=1)
    session.finish()


def test_offline_session_logs_and_persists_run_id(smoke_cfg, tmp_path, wandb_home):
    run_dir = tmp_path / "run_offline"
    cfg = replace_wandb_config(
        smoke_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow",
        name="offline_unit",
        tags=("unit",),
        run_id="gigaflow_offline_unit_001",
        resume="allow",
    )
    session = build_wandb_session(cfg, run_dir=run_dir)
    run_id = session.start()
    assert run_id == "gigaflow_offline_unit_001"
    assert session.active is True
    assert session._eval_axis_defined is True
    assert EVAL_SOURCE_STEP_METRIC == "eval/source_step"
    assert EVAL_METRIC_GLOB == "eval/*"
    session.log_metrics(
        flatten_train_metrics(
            progress_metrics={"policy_loss": 0.5, "transitions_per_s": 10.0},
            profile={"collect_s": 0.1, "ppo_s": 0.2},
            update_index=1,
            transitions=32,
            reward_term_means={"total_mean": 0.01},
            track_stats={"unique_count": 1.0},
            density_stats={"active_frac": 0.5},
            device="cpu",
        ),
        step=1,
    )
    session.finish()
    meta = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["wandb"]["run_id"] == "gigaflow_offline_unit_001"
    assert "provenance" in meta
    assert "hardware" in meta["provenance"]


def test_offline_resume_reuses_same_run_id(smoke_cfg, tmp_path, wandb_home):
    run_dir = tmp_path / "run_resume"
    cfg = replace_wandb_config(
        smoke_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow",
        run_id="gigaflow_resume_unit_001",
        resume="allow",
    )
    s1 = build_wandb_session(cfg, run_dir=run_dir)
    assert s1.start() == "gigaflow_resume_unit_001"
    s1.log_metrics({"train/update_index": 1.0}, step=1)
    s1.finish()

    # Second init with same id + resume must not allocate a new id.
    cfg2 = replace_wandb_config(cfg, run_id=None, resume="must")
    s2 = build_wandb_session(cfg2, run_dir=run_dir)
    assert s2.start() == "gigaflow_resume_unit_001"
    s2.log_metrics({"train/update_index": 2.0}, step=2)
    s2.finish()


def test_run_training_offline_writes_checkpoint_run_id(smoke_cfg, tmp_path, wandb_home):
    cfg = replace_wandb_config(
        smoke_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow",
        run_id="gigaflow_train_offline_001",
        resume="allow",
    )
    run_dir = tmp_path / "train"
    progress = run_training(
        cfg,
        num_updates=1,
        device="cpu",
        run_dir=run_dir,
        checkpoint_interval=0,
    )
    assert progress.update_index >= 1
    assert (run_dir / "metrics_000001.json").is_file()
    assert (run_dir / RUN_META_FILENAME).is_file()
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ckpt["wandb_run_id"] == "gigaflow_train_offline_001"


def test_recoverable_log_failure_keeps_local_authority(smoke_cfg, tmp_path, wandb_home):
    cfg = replace_wandb_config(
        smoke_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow",
        run_id="gigaflow_fail_unit_001",
    )
    session = WandbSession(cfg.wandb, experiment=cfg, run_dir=tmp_path / "run")
    session.start()

    class _Boom:
        def log(self, *args, **kwargs):
            raise RuntimeError("synthetic wandb queue failure")

        def finish(self):
            return None

    session._run = _Boom()
    with pytest.warns(RuntimeWarning, match="metric logging failed"):
        session.log_metrics({"ppo/policy_loss": 1.0}, step=1)
    assert session.active is False
    # Local path still writable independently of W&B.
    trainer = build_trainer(cfg, device="cpu", run_dir=tmp_path / "local")
    trainer.setup()
    progress = trainer.train_update()
    assert (tmp_path / "local" / f"metrics_{progress.update_index:06d}.json").is_file()
    session.finish()


def test_summarize_rollout_aux_shapes():
    rewards = torch.arange(8, dtype=torch.float32).view(2, 4)
    valid = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 0, 0]],
        dtype=torch.bool,
    )
    track_id = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    active_end = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    terms = {
        "progress": torch.tensor(
            [[0.1, 0.2, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
    }
    done = torch.zeros_like(valid)
    done[0, 0] = True
    wall = torch.zeros_like(valid)
    wall[0, 0] = True
    reward_means, track_stats, density_stats, sim_stats = summarize_rollout_aux(
        rewards=rewards,
        valid=valid,
        track_id=track_id,
        active_end=active_end,
        max_agents_per_world=2,
        reward_terms=terms,
        done=done,
        timeout=torch.zeros_like(valid),
        reset_mask=done,
        wall_contact=wall,
        contact=torch.zeros_like(valid),
    )
    assert reward_means["total_mean"] == pytest.approx(rewards[valid].float().mean().item())
    assert reward_means["progress_mean"] == pytest.approx(
        terms["progress"][valid].float().mean().item()
    )
    assert track_stats["unique_count"] == 2.0
    assert density_stats["agents_per_world_mean"] == pytest.approx(1.0)
    assert sim_stats["done_frac"] > 0.0
    assert "reset_frac" in sim_stats
    assert sim_stats["oob_count"] == pytest.approx(1.0)


def _lap_report(suite: str, seed: int, lap_time_s, completers: float, participants: float):
    return EvalReport(
        suite=suite,
        seed=seed,
        metrics=EvalMetrics(
            lap_time_s=lap_time_s,
            completion_rate=completers / participants,
            progress_rate_mps=1.0,
            collision_per_km=0.0,
            oob_per_km=0.0,
            clean_overtakes=0.0,
            stall_rate=0.0,
            return_mean=0.0,
        ),
        extras={
            "num_lap_completers": completers,
            "num_participants": participants,
        },
    )


def test_summarize_eval_lap_times_aggregates_over_reports_that_timed_a_lap():
    reports = [
        _lap_report("solo", 0, 30.0, 4.0, 4.0),
        _lap_report("solo", 1, 20.0, 2.0, 4.0),
        _lap_report("dense", 0, None, 0.0, 8.0),
    ]
    summary = summarize_eval_lap_times(reports)
    assert summary["eval/lap_time_s_mean"] == pytest.approx(25.0)
    assert summary["eval/lap_time_s_best"] == pytest.approx(20.0)
    assert summary["eval/solo/lap_time_s_mean"] == pytest.approx(25.0)
    # The suite where nobody finished contributes no lap-time series.
    assert "eval/dense/lap_time_s_mean" not in summary
    assert summary["eval/lap_completers_frac"] == pytest.approx(6.0 / 16.0)


def test_summarize_eval_lap_times_reports_zero_completers_without_a_lap_time():
    reports = [
        _lap_report("solo", 0, None, 0.0, 4.0),
        _lap_report("dense", 0, None, 0.0, 8.0),
    ]
    summary = summarize_eval_lap_times(reports)
    # An eval that ran but finished no lap is a countable 0, not a missing point.
    assert summary == {"eval/lap_completers_frac": 0.0}
    assert summarize_eval_lap_times([]) == {}


def test_log_evaluation_publishes_lap_aggregates(smoke_cfg, tmp_path, wandb_home):
    cfg = replace_wandb_config(
        smoke_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow",
        run_id="gigaflow_eval_lap_unit_001",
    )
    session = build_wandb_session(cfg, run_dir=tmp_path / "run_eval")
    session.start()
    session.log_evaluation(
        [
            _lap_report("solo", 0, 30.0, 4.0, 4.0),
            _lap_report("dense", 0, None, 0.0, 8.0),
        ],
        source_step=1500,
        global_step=1500,
    )
    # wandb buffers ``log(..., step=N)`` into row N and commits it only once the
    # history step advances, so the row above is not readable until a later log.
    session.log_evaluation(
        [_lap_report("solo", 0, 25.0, 4.0, 4.0)],
        source_step=1600,
        global_step=1600,
    )
    summary = dict(session._run.summary)
    session.finish()
    assert summary[EVAL_SOURCE_STEP_METRIC] == pytest.approx(1500.0)
    assert summary["eval/solo/seed0/lap_time_s"] == pytest.approx(30.0)
    assert summary["eval/lap_time_s_mean"] == pytest.approx(30.0)
    assert summary["eval/lap_time_s_best"] == pytest.approx(30.0)
    assert summary["eval/lap_completers_frac"] == pytest.approx(4.0 / 12.0)
    # Per-suite/seed keys are untouched by the aggregates.
    assert summary["eval/dense/seed0/extra/num_lap_completers"] == pytest.approx(0.0)

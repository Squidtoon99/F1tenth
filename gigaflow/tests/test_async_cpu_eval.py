"""Real subprocess / concurrency tests for isolated CPU cadenced eval."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from gigaflow_f1tenth.async_cpu_eval import (
    REPORT_FILENAME,
    SKIP_REASON_INFLIGHT,
    UPLOADED_FILENAME,
    AsyncCpuEvalManager,
    _InflightEval,
    assert_no_cuda,
    build_cpu_eval_config,
    cpu_hidden_env,
    export_cpu_actor_snapshot,
    read_status,
    read_uploaded,
    resolve_eval_global_step,
    run_cpu_eval_worker,
    upload_completed_eval,
)
from gigaflow_f1tenth.wandb_log import EVAL_SOURCE_STEP_METRIC
from gigaflow_f1tenth.config import (
    config_from_dict,
    config_to_dict,
    load_config,
    replace_wandb_config,
)
from gigaflow_f1tenth.evaluation import (
    EvalMetrics,
    FixedSeedEvaluator,
    resolve_eval_device,
    resolve_eval_num_worlds,
)
from gigaflow_f1tenth.model import build_actor
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth import tracks as T
from gigaflow_f1tenth.wandb_log import build_wandb_session

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


@pytest.fixture
def cpu_eval_cfg():
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["worlds"]["num_worlds"] = 4
    raw["worlds"]["max_agents_per_world"] = 2
    raw["worlds"]["device"] = "cuda"
    raw["evaluation"]["device"] = "cpu"
    raw["evaluation"]["num_worlds"] = 1
    raw["evaluation"]["soak_steps"] = 8
    raw["evaluation"]["seeds"] = [0]
    raw["evaluation"]["suite"] = ["solo"]
    raw["evaluation"]["viz_enabled"] = False
    raw["wandb"] = {
        "enabled": True,
        "mode": "offline",
        "project": "f1tenth-gigaflow-test",
        "eval_interval_updates": 2,
        "resume": "never",
        "log_artifacts": True,
        "tags": ["test"],
    }
    return config_from_dict(raw)


def test_resolve_eval_scale_distinct_from_training():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=64, device="cuda"),
        evaluation=replace(cfg.evaluation, device="cpu", num_worlds=23),
    )
    assert resolve_eval_device(cfg) == "cpu"
    assert resolve_eval_num_worlds(cfg) == 23
    assert cfg.worlds.num_worlds == 64


def test_build_cpu_eval_config_forces_cpu_and_eval_worlds(cpu_eval_cfg):
    frozen = build_cpu_eval_config(cpu_eval_cfg)
    assert frozen.worlds.device == "cpu"
    # Training world count preserved for PPO validation; eval scale is separate.
    assert frozen.worlds.num_worlds == cpu_eval_cfg.worlds.num_worlds
    assert frozen.evaluation.num_worlds == 1
    assert frozen.evaluation.device == "cpu"
    assert frozen.wandb.enabled is False


def test_export_cpu_actor_snapshot_has_no_cuda_tensors(cpu_eval_cfg, tmp_path):
    actor = build_actor(cpu_eval_cfg)
    if torch.cuda.is_available():
        actor = actor.to("cuda")
        assert next(actor.parameters()).is_cuda
    path = tmp_path / "actor_snapshot.pt"
    export_cpu_actor_snapshot(cpu_eval_cfg, actor, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, tensor in payload["actor_state_dict"].items():
        assert not (hasattr(tensor, "is_cuda") and tensor.is_cuda), key
        assert tensor.device.type == "cpu"


def test_assert_no_cuda_under_hidden_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Torch caches CUDA init in-process; only assert when the runtime agrees.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        pytest.skip("CUDA still visible despite empty CUDA_VISIBLE_DEVICES")
    info = assert_no_cuda("unit")
    assert info["cuda_device_count"] == 0
    assert info["torch_cuda_available"] is False


def test_worker_subprocess_allocates_no_cuda(cpu_eval_cfg, tmp_path):
    run_dir = tmp_path / "run"
    mgr = AsyncCpuEvalManager(cpu_eval_cfg, run_dir=run_dir)
    actor = build_actor(cpu_eval_cfg)
    if torch.cuda.is_available():
        actor = actor.cuda()
    result = mgr.try_launch(step=1, actor=actor, session=None)
    assert result == "launched"
    assert mgr.inflight_pid is not None
    pid = mgr.inflight_pid
    status = None
    deadline = time.time() + 180
    while time.time() < deadline:
        status = mgr.poll_and_upload(None)
        if not mgr.inflight:
            break
        time.sleep(0.2)
    assert not mgr.inflight, "eval child did not finish"
    assert status is not None
    assert status.get("state") == "completed", status
    assert status.get("cuda_device_count") == 0
    assert status.get("torch_cuda_available") is False
    assert (run_dir / "eval_000001" / REPORT_FILENAME).is_file()
    zpath = Path(f"/proc/{pid}/stat")
    if zpath.is_file():
        fields = zpath.read_text(encoding="utf-8").split()
        assert fields[2] != "Z", f"zombie left behind pid={pid}"


def test_skip_coalesce_when_inflight(cpu_eval_cfg, tmp_path):
    run_dir = tmp_path / "run"
    mgr = AsyncCpuEvalManager(cpu_eval_cfg, run_dir=run_dir)
    actor = build_actor(cpu_eval_cfg)

    hold = tmp_path / "hold.py"
    hold.write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "out = Path(os.environ['HOLD_OUT'])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'eval_status.json').write_text(\n"
        "    json.dumps({\n"
        "        'state': 'running',\n"
        "        'pid': os.getpid(),\n"
        "        'step': 1,\n"
        "        'cuda_device_count': 0,\n"
        "        'torch_cuda_available': False,\n"
        "    }) + '\\n'\n"
        ")\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    out1 = run_dir / "eval_000001"
    env = cpu_hidden_env()
    env["HOLD_OUT"] = str(out1)
    log_path = tmp_path / "hold.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, str(hold)],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    mgr._inflight = _InflightEval(
        step=1, out_dir=out1, proc=proc, launched_at=time.perf_counter()
    )
    try:
        result = mgr.try_launch(step=2, actor=actor, session=None)
        assert result == "skipped_inflight"
        skip = read_status(run_dir / "eval_000002")
        assert skip is not None
        assert skip["state"] == "skipped"
        assert skip["reason"] == SKIP_REASON_INFLIGHT
        assert skip["inflight_step"] == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        mgr._inflight = None


def test_shutdown_reaps_real_process_group(cpu_eval_cfg, tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "eval_000001"
    out_dir.mkdir(parents=True)
    script = tmp_path / "process_tree.py"
    child_pid_path = tmp_path / "child.pid"
    script.write_text(
        "import os, signal, subprocess, sys, time\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c',\n"
        "    'import signal,time; signal.signal(signal.SIGTERM, "
        "signal.SIG_IGN); time.sleep(60)'\n"
        "])\n"
        f"open({str(child_pid_path)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "tree.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.time() + 10
    while time.time() < deadline and not child_pid_path.is_file():
        time.sleep(0.05)
    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    mgr = AsyncCpuEvalManager(cpu_eval_cfg, run_dir=run_dir)
    mgr._inflight = _InflightEval(
        step=1,
        out_dir=out_dir,
        proc=proc,
        launched_at=time.perf_counter(),
    )
    mgr.shutdown(
        None,
        grace_s=0.05,
        term_wait_s=0.1,
        kill_wait_s=2.0,
    )
    assert mgr.inflight is False
    assert proc.poll() is not None
    deadline = time.time() + 5
    while time.time() < deadline and Path(f"/proc/{child_pid}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_parent_uploads_completed_eval_to_wandb(cpu_eval_cfg, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_dir"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")

    cfg = replace_wandb_config(
        cpu_eval_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow-test",
        resume="never",
    )
    run_dir = tmp_path / "run"
    session = build_wandb_session(cfg, run_dir=run_dir)
    session.start()
    try:
        mgr = AsyncCpuEvalManager(cfg, run_dir=run_dir)
        actor = build_actor(cfg)
        assert mgr.try_launch(step=3, actor=actor, session=session) == "launched"
        deadline = time.time() + 180
        while time.time() < deadline and mgr.inflight:
            mgr.poll_and_upload(session, train_step=3)
            time.sleep(0.2)
        assert not mgr.inflight
        assert 3 in mgr.completed_uploads
        assert (run_dir / "eval_000003" / REPORT_FILENAME).is_file()
        uploaded = read_uploaded(run_dir / "eval_000003")
        assert uploaded is not None
        assert uploaded["source_step"] == 3
        assert uploaded.get("global_step") is None or uploaded["global_step"] >= 3
    finally:
        session.finish()


def _write_completed_eval_fixture(out_dir: Path, *, source_step: int) -> None:
    """Local completed-eval tree with real report JSON + PNG/MP4 media."""
    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = EvalMetrics(
        lap_time_s=12.0,
        completion_rate=1.0,
        progress_rate_mps=0.5,
        collision_per_km=0.0,
        oob_per_km=0.0,
        clean_overtakes=0.0,
        stall_rate=0.0,
        return_mean=0.1,
    )
    report = {
        "reports": [
            {
                "suite": "solo",
                "seed": 0,
                "metrics": metrics.__dict__,
                "extras": {"horizon_steps": 8.0},
            }
        ]
    }
    (out_dir / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "eval_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "step": int(source_step),
                "elapsed_s": 1.5,
                "num_worlds": 1,
                "training_num_worlds": 4,
                "cuda_device_count": 0,
                "media_count": 2,
                "exit_code": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    png = out_dir / "solo_seed0_frame.png"
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(png)
    mp4 = out_dir / "solo_seed0.mp4"
    try:
        import imageio.v2 as imageio

        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
        imageio.mimsave(
            mp4, frames, fps=2, format="FFMPEG", codec="libx264"
        )
    except Exception:
        # Still exercise Video(format="mp4") path when ffmpeg encode is unavailable.
        mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)


def test_async_eval_upload_uses_current_train_step_not_source(
    cpu_eval_cfg, tmp_path, monkeypatch
):
    """Training can advance past cadence before async eval finishes.

    Reproduces the live failure mode: train logged global step 210, then parent
    reaped source step 100. Eval must log ``eval/source_step=100`` on the custom
    axis while global history step stays >=210 (never ``step=100``).
    """
    wandb = pytest.importorskip("wandb")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_dir"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    cfg = replace_wandb_config(
        cpu_eval_cfg,
        enabled=True,
        mode="offline",
        project="f1tenth-gigaflow-test",
        name="async_step_order",
        resume="never",
        run_id="gigaflow_async_step_order_001",
    )
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval_000100"
    _write_completed_eval_fixture(eval_dir, source_step=100)

    session = build_wandb_session(cfg, run_dir=run_dir)
    session.start()
    assert session._eval_axis_defined is True
    try:
        # Train metrics advance while the child (source_step=100) is still "inflight".
        session.log_metrics({"train/update_index": 210.0, "ppo/policy_loss": 0.1}, step=210)
        assert session.last_step == 210
        assert resolve_eval_global_step(train_step=210, session=session) == 210
        # Cadence must never be selected as the global history step.
        assert resolve_eval_global_step(source_step=100, session=session) == 210

        hold = tmp_path / "completed_hold.py"
        hold.write_text(
            "import os, time\n"
            "time.sleep(0.05)\n",
            encoding="utf-8",
        )
        with (tmp_path / "hold.log").open("w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                [sys.executable, str(hold)],
                env=cpu_hidden_env(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        mgr = AsyncCpuEvalManager(cfg, run_dir=run_dir)
        mgr._inflight = _InflightEval(
            step=100,
            out_dir=eval_dir,
            proc=proc,
            launched_at=time.perf_counter(),
        )
        status = None
        deadline = time.time() + 30
        while time.time() < deadline:
            status = mgr.poll_and_upload(session, train_step=210)
            if not mgr.inflight:
                break
            time.sleep(0.05)
        assert not mgr.inflight
        assert status is not None
        assert status.get("state") == "completed"
        assert 100 in mgr.completed_uploads
        assert mgr.completed_upload_records
        record = mgr.completed_upload_records[-1]
        assert record["source_step"] == 100
        assert record["global_step"] >= 210
        assert session.last_step >= 210

        uploaded = read_uploaded(eval_dir)
        assert uploaded is not None
        assert uploaded["source_step"] == 100
        assert uploaded["global_step"] >= 210
        assert (eval_dir / UPLOADED_FILENAME).is_file()

        # Duplicate upload is a no-op (marker short-circuit).
        again = upload_completed_eval(
            session,
            source_step=100,
            out_dir=eval_dir,
            train_step=220,
            mark_backfill=True,
        )
        assert again is not None
        assert again["global_step"] == uploaded["global_step"]
    finally:
        session.finish()

    offline = next((run_dir / "wandb").glob("offline-run-*"), None)
    assert offline is not None
    debug = (offline / "logs" / "debug.log").read_text(encoding="utf-8", errors="replace")
    assert "non monotonic" not in debug.lower()
    assert "non-monotonic" not in debug.lower()
    media_root = offline / "files" / "media"
    assert media_root.is_dir(), offline
    media_files = [p for p in media_root.rglob("*") if p.is_file()]
    assert media_files, offline
    assert any("eval/media" in str(p) for p in media_files), media_files
    stepped = []
    for path in media_files:
        parts = path.name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            stepped.append((path, int(parts[1])))
    assert stepped, [p.name for p in media_files]
    assert all(step >= 210 for _, step in stepped), stepped
    # Prove the rejected pattern (global step=100) is not what we staged.
    assert all(step != 100 for _, step in stepped), stepped
    _ = (wandb, EVAL_SOURCE_STEP_METRIC)


def test_inprocess_worker_rejects_visible_cuda(cpu_eval_cfg, tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to prove rejection path")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.device_count() < 1:
        pytest.skip("CUDA_VISIBLE_DEVICES=0 still hides devices in this env")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(config_to_dict(cpu_eval_cfg), indent=2) + "\n", encoding="utf-8"
    )
    actor_path = tmp_path / "actor.pt"
    export_cpu_actor_snapshot(cpu_eval_cfg, build_actor(cpu_eval_cfg), actor_path)
    out = tmp_path / "out"
    rc = run_cpu_eval_worker(
        config_path=cfg_path,
        actor_path=actor_path,
        output_dir=out,
        step=9,
    )
    assert rc == 1
    status = read_status(out)
    assert status is not None
    assert status["state"] == "failed"
    assert "CUDA must be hidden" in str(status.get("error"))


def _local_manifest(cfg, tmp_path: Path) -> Path:
    """Build a real one-track atlas cache from the checked-in fixture."""
    local = tmp_path / "local"
    local.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        ROOT / "tests" / "fixtures" / "tracks" / "oval_centerline.csv",
        local / "oval_centerline.csv",
    )
    T.prepare_tracks(
        cfg,
        str(tmp_path),
        pin_path=ROOT / "configs" / "track_pin.json",
        lut_resolution=0.5,
        edt_resolution=0.25,
        skip_download=True,
    )
    return tmp_path / T.MANIFEST_FILENAME


def test_sync_and_async_eval_load_the_configured_atlas(tmp_path):
    """Fail-before: evaluation silently scored the policy on a synthetic oval."""
    cfg = load_config(SMOKE)
    manifest = _local_manifest(cfg, tmp_path)
    raw = config_to_dict(cfg)
    raw["tracks"]["manifest_path"] = str(manifest)
    cfg = config_from_dict(raw)

    expected = T.load_atlas(str(manifest)).view()
    synthetic = make_synthetic_oval_atlas(
        max_agents=cfg.worlds.max_agents_per_world
    )
    sync_atlas = FixedSeedEvaluator(cfg, device="cpu").atlas
    async_atlas = FixedSeedEvaluator(
        build_cpu_eval_config(cfg), device="cpu"
    ).atlas

    assert sync_atlas.track_ids == expected.track_ids
    assert async_atlas.track_ids == expected.track_ids
    assert sync_atlas.track_ids != synthetic.track_ids
    for atlas in (sync_atlas, async_atlas):
        assert atlas.centerline_xy.shape == expected.centerline_xy.shape
        assert (atlas.centerline_xy == expected.centerline_xy).all()

    # A configured manifest that cannot be loaded must fail, not fall back.
    raw["tracks"]["manifest_path"] = str(tmp_path / "absent")
    with pytest.raises(T.TrackError):
        FixedSeedEvaluator(config_from_dict(raw), device="cpu")


def test_production_config_declares_cpu_eval_device():
    prod = ROOT / "configs" / "production_h100.yaml"
    cfg = load_config(prod)
    assert cfg.evaluation.device == "cpu"
    assert cfg.evaluation.num_worlds is None
    assert resolve_eval_device(cfg) == "cpu"

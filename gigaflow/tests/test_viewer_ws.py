from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth import model as model_mod
from gigaflow_f1tenth.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_SCOPE_SIM_TRAINING,
    export_actor_artifact,
)
from gigaflow_f1tenth.config import config_from_dict, config_to_dict, load_config
from gigaflow_f1tenth.model import (
    ConditionedLidarGRUActor,
    actor_architecture_from_module,
    actor_shapes,
    build_actor,
)
from gigaflow_f1tenth.viewer.protocol import decode_client_message
from gigaflow_f1tenth.viewer.replay import (
    CheckpointReplay,
    ViewerError,
    ViewerLaunchArgs,
)
from gigaflow_f1tenth.viewer.server import (
    _client_loop,
    _run_ws_server,
    _start_http_server,
    advance_deadline,
    discover_checkpoints,
    resolve_checkpoint_request,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tracks" / "oval_centerline.csv"

pytest.importorskip("websockets")


@pytest.fixture()
def smoke_replay(tmp_path: Path) -> CheckpointReplay:
    local = tmp_path / "local"
    local.mkdir()
    shutil.copy(FIXTURE, local / "oval_centerline.csv")
    from gigaflow_f1tenth.cli import main

    assert (
        main(
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
        == 0
    )
    cfg = load_config(SMOKE)
    actor_path = tmp_path / "actor.pt"
    export_actor_artifact(cfg, build_actor(cfg), str(actor_path))
    return CheckpointReplay.from_launch_args(
        ViewerLaunchArgs(
            checkpoint=actor_path,
            config=SMOKE,
            cache_dir=tmp_path,
            suite="solo",
            seed=0,
            device="cpu",
            track="oval",
            open_browser=False,
        )
    )


def _export_wide_condition_artifact(cfg, path: Path, width: int = 16) -> Path:
    """Write an older-generation artifact: wider condition, no normalizer buffers."""
    shapes = replace(actor_shapes(cfg), condition_dim=width)
    live_schema = model_mod.CONDITION_DIM
    model_mod.CONDITION_DIM = width
    try:
        actor = ConditionedLidarGRUActor(shapes)
    finally:
        model_mod.CONDITION_DIM = live_schema
    torch.save(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "scope": ARTIFACT_SCOPE_SIM_TRAINING,
            "actor_architecture": actor_architecture_from_module(actor),
            "condition_schema": {"condition_dim": width},
            "deployment_style": cfg.evaluation.conservative_deployment_style,
            "deployment_condition_normalized": [1.0] * width,
            "actor_state_dict": {
                key: value.detach().cpu()
                for key, value in actor.state_dict().items()
                if not key.startswith("sensor_normalizer.")
            },
            "sensor_normalizer": {},
            "condition_normalizer": {"mode": "schema_ranges"},
            "track_manifest_hash": None,
        },
        path,
    )
    return path


def _export_narrow_trunk_artifact(cfg, path: Path) -> Path:
    raw = config_to_dict(cfg)
    agents = dict(raw["agents"])
    agents["gru_hidden_dim"] = 128
    raw["agents"] = agents
    other_cfg = config_from_dict(raw, check_training_budget=False)
    export_actor_artifact(other_cfg, build_actor(other_cfg), str(path))
    return path


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._inbox: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        return await self._inbox.get()

    def push(self, data: str) -> None:
        self._inbox.put_nowait(data)


def test_websocket_smoke_hello_track_tick(smoke_replay: CheckpointReplay):
    ws = _FakeWS()

    async def run_briefly() -> None:
        task = asyncio.create_task(_client_loop(ws, smoke_replay, {ws}, ()))
        await asyncio.sleep(0.05)
        ws.push(json.dumps({"v": 4, "type": "pause"}))
        await asyncio.sleep(0.05)
        ws.push(json.dumps({"v": 4, "type": "set_suite", "suite": "head_to_head"}))
        await asyncio.sleep(0.08)
        ws.push(json.dumps({"v": 4, "type": "set_suite", "suite": "dense"}))
        await asyncio.sleep(0.08)
        ws.push(
            json.dumps(
                {
                    "v": 4,
                    "type": "set_environment_count",
                    "environment_count": 2,
                }
            )
        )
        await asyncio.sleep(0.15)
        ws.push(
            json.dumps(
                {
                    "v": 4,
                    "type": "set_obstacle_preset",
                    "obstacle_preset": "light",
                }
            )
        )
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_briefly())
    types = [json.loads(m)["type"] for m in ws.sent]
    assert "hello" in types
    assert "track" in types
    assert "tick" in types
    hellos = [json.loads(m) for m in ws.sent if json.loads(m)["type"] == "hello"]
    assert any(h["suite"] == "head_to_head" and h["num_cars"] == 2 for h in hellos)
    assert hellos[-1]["suite"] == "dense"
    assert hellos[-1]["num_environments"] == 2
    assert hellos[-1]["cars_per_environment"] == 2
    assert hellos[-1]["obstacle_preset"] == "light"
    assert hellos[-1]["num_obstacles"] == 4
    ticks = [json.loads(m) for m in ws.sent if json.loads(m)["type"] == "tick"]
    assert ticks
    assert "t" in ticks[-1]
    assert "control_dt" in ticks[-1]
    assert len(ticks[-1]["cars"][0]) == 12
    h2h_ticks = [
        t
        for t in ticks
        if len(t["cars"]) == 2
    ]
    assert h2h_ticks
    assert any([c[3] for c in tick["cars"]] == [1.0, 1.0] for tick in h2h_ticks)


def test_cli_view_help_lists_flags():
    from gigaflow_f1tenth.cli import build_parser

    parser = build_parser()
    view = None
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        if hasattr(action, "choices") and action.choices and "view" in action.choices:
            view = action.choices["view"]
            break
    assert view is not None
    help_text = view.format_help()
    assert "--checkpoint" in help_text
    assert "--cache-dir" in help_text
    assert "--suite" in help_text


def test_decode_set_track_id():
    msg = decode_client_message('{"v":4,"type":"set_track","track_id":2}')
    assert msg["track_id"] == 2


def test_advance_deadline_skips_missed_slots_without_burst():
    period = 0.1
    deadline = 1.0
    # Emit finished 50 ms late: next slot stays on the absolute grid.
    assert advance_deadline(deadline, period, now=1.05) == pytest.approx(1.1)
    # Overran by more than one period: skip missed slots (no catch-up burst).
    assert advance_deadline(deadline, period, now=1.25) == pytest.approx(1.3)
    # Exactly on a boundary: move to the following slot.
    assert advance_deadline(deadline, period, now=1.1) == pytest.approx(1.2)
    # Irregular 70–180 ms compute costs still land on the period grid.
    d = 0.0
    costs = [0.07, 0.18, 0.11, 0.09, 0.16, 0.075, 0.14]
    scheduled = []
    now = 0.0
    for cost in costs:
        # Emit at current deadline (or immediately if already late).
        emit_at = max(d, now)
        scheduled.append(emit_at)
        now = emit_at + cost
        d = advance_deadline(d, period, now)
    for t in scheduled[1:]:
        # Every scheduled emit aligns to n*period from the first.
        assert abs((t - scheduled[0]) / period - round((t - scheduled[0]) / period)) < 1e-9
    gaps = [b - a for a, b in zip(scheduled, scheduled[1:])]
    assert all(g + 1e-12 >= period for g in gaps)
    # No zero-gap burst pair.
    assert min(gaps) > period * 0.5


async def _next_message(ws: object, kind: str) -> dict:
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        if message["type"] == kind:
            return message


def test_checkpoint_index_endpoint_groups_runs(
    smoke_replay: CheckpointReplay, tmp_path: Path
):
    cfg = load_config(SMOKE)
    root = (tmp_path / "outputs").resolve()
    preserved = root / "preserved" / "kj65" / "checkpoints"
    latest = root / "runs" / "rtx4080"
    older = root / "remote_backups" / "o391" / "checkpoints"
    preserved.mkdir(parents=True)
    latest.mkdir(parents=True)
    older.mkdir(parents=True)
    export_actor_artifact(cfg, build_actor(cfg), str(preserved / "actor_000100.pt"))
    shutil.copy(preserved / "actor_000100.pt", preserved / "actor_000200.pt")
    shutil.copy(preserved / "actor_000100.pt", latest / "actor_004300.pt")
    outside = (tmp_path / "actor_outside.pt").resolve()
    shutil.copy(preserved / "actor_000100.pt", outside)
    (preserved / "SHA256SUMS").write_text("checksums only\n", encoding="utf-8")
    wide = older / "actor_010000.pt"
    _export_wide_condition_artifact(cfg, wide)
    narrow_trunk = _export_narrow_trunk_artifact(cfg, latest / "actor_004400.pt")

    index = discover_checkpoints([root], cfg)
    runs = index["runs"]
    assert [run["run"] for run in runs] == [
        "preserved/kj65/checkpoints",
        "remote_backups/o391/checkpoints",
        "runs/rtx4080",
    ]
    assert [entry["name"] for entry in runs[0]["checkpoints"]] == [
        "actor_000100.pt",
        "actor_000200.pt",
    ]
    assert runs[1]["checkpoints"] == [
        {"name": "actor_010000.pt", "path": str(wide), "condition_dim": 16}
    ]
    assert runs[2]["checkpoints"] == [
        {
            "name": "actor_004300.pt",
            "path": str(latest / "actor_004300.pt"),
            "condition_dim": int(cfg.agents.condition_dim),
        }
    ]
    assert index["condition_dim"] == int(cfg.agents.condition_dim)
    assert index["incompatible"] == {
        "count": 1,
        "reasons": [
            "checkpoint/config mismatch on gru_hidden_dim: checkpoint=128 "
            f"config={cfg.agents.gru_hidden_dim}"
        ],
    }

    wanted = latest / "actor_004300.pt"
    assert resolve_checkpoint_request([root], str(wanted)) == wanted
    # Left out of the listing, still resolvable: the load path rejects it.
    assert resolve_checkpoint_request([root], str(narrow_trunk)) == narrow_trunk
    with pytest.raises(ViewerError, match="gru_hidden_dim"):
        smoke_replay.load_actor_candidate(narrow_trunk)
    with pytest.raises(ViewerError, match="not available"):
        resolve_checkpoint_request([root], str(outside))

    server = _start_http_server(tmp_path, "127.0.0.1", 0, smoke_replay, (root,))
    try:
        port = int(server.server_address[1])
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/checkpoints", timeout=20
        ) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
    assert payload["active"] == {
        "name": smoke_replay.checkpoint_label,
        "path": smoke_replay.checkpoint_path,
    }
    listed = {
        entry["path"] for run in payload["runs"] for entry in run["checkpoints"]
    }
    assert str(latest / "actor_004300.pt") in listed
    assert str(wide) in listed
    assert str(narrow_trunk) not in listed
    assert payload["incompatible"]["count"] == 1


def test_set_checkpoint_swaps_policy_and_survives_incompatible_actor(
    smoke_replay: CheckpointReplay, tmp_path: Path
):
    cfg = load_config(SMOKE)
    root = (tmp_path / "browsable").resolve()
    root.mkdir()
    replacement = root / "actor_000900.pt"
    export_actor_artifact(cfg, build_actor(cfg), str(replacement))
    incompatible = _export_narrow_trunk_artifact(cfg, root / "actor_001000.pt")
    wide = _export_wide_condition_artifact(cfg, root / "actor_010000.pt")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])

    async def exercise() -> None:
        from websockets.asyncio.client import connect

        server = asyncio.create_task(
            _run_ws_server(smoke_replay, "127.0.0.1", port, None, (root,))
        )
        try:
            await asyncio.sleep(0.1)
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                await _next_message(ws, "tick")
                await ws.send(
                    json.dumps(
                        {
                            "v": 4,
                            "type": "set_checkpoint",
                            "checkpoint": str(replacement),
                        }
                    )
                )
                update = await _next_message(ws, "actor_update")
                assert update["checkpoint"] == replacement.name
                assert smoke_replay.checkpoint_label == replacement.name
                assert smoke_replay.checkpoint_path == str(replacement)
                loaded_actor = smoke_replay.actor

                await ws.send(
                    json.dumps(
                        {
                            "v": 4,
                            "type": "set_checkpoint",
                            "checkpoint": str(incompatible),
                        }
                    )
                )
                rejected = await _next_message(ws, "error")
                assert rejected["code"] == "checkpoint_error"
                assert "gru_hidden_dim" in rejected["message"]
                assert smoke_replay.actor is loaded_actor
                assert smoke_replay.checkpoint_label == replacement.name

                await ws.send(
                    json.dumps(
                        {
                            "v": 4,
                            "type": "set_checkpoint",
                            "checkpoint": str(tmp_path / "actor_elsewhere.pt"),
                        }
                    )
                )
                unavailable = await _next_message(ws, "error")
                assert unavailable["code"] == "checkpoint_error"
                assert smoke_replay.actor is loaded_actor

                await ws.send(
                    json.dumps(
                        {
                            "v": 4,
                            "type": "set_checkpoint",
                            "checkpoint": str(wide),
                        }
                    )
                )
                widened = await _next_message(ws, "actor_update")
                assert widened["checkpoint"] == wide.name
                assert smoke_replay.actor.shapes().condition_dim == 16
                assert float(smoke_replay.actor.sensor_normalizer.count.item()) == 0.0
                before = await _next_message(ws, "tick")
                after = await _next_message(ws, "tick")
                assert after["step"] != before["step"]

                await ws.send(json.dumps({"v": 4, "type": "ping"}))
                await _next_message(ws, "tick")
        finally:
            server.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server

    asyncio.run(exercise())


def test_hot_reload_keeps_server_and_websocket_alive(
    smoke_replay: CheckpointReplay, tmp_path: Path
):
    cfg = load_config(SMOKE)
    replacement = tmp_path / "replacement.pt"
    export_actor_artifact(cfg, build_actor(cfg), str(replacement))
    sha256 = hashlib.sha256(replacement.read_bytes()).hexdigest()
    request = tmp_path / "policy_reload.request.json"
    result = tmp_path / "policy_reload.result.json"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])

    async def exercise() -> None:
        from websockets.asyncio.client import connect

        server = asyncio.create_task(
            _run_ws_server(smoke_replay, "127.0.0.1", port, request)
        )
        try:
            await asyncio.sleep(0.1)
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                assert json.loads(await ws.recv())["type"] == "hello"
                await ws.recv()
                await ws.recv()
                request.write_text(
                    json.dumps(
                        {
                            "request_id": "replacement",
                            "checkpoint": str(replacement),
                            "sha256": sha256,
                        }
                    )
                )
                while True:
                    message = json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=10)
                    )
                    if message["type"] == "actor_update":
                        break
                loaded = json.loads(result.read_text())
                assert loaded["status"] == "loaded"
                assert loaded["server_pid"] == os.getpid()
                assert loaded["hidden_state_reset"] is True
                assert smoke_replay.checkpoint_label == replacement.name
                await ws.send(json.dumps({"v": 4, "type": "ping"}))
                assert json.loads(await ws.recv())["type"] == "tick"
                installed_actor = smoke_replay.actor
                corrupt = tmp_path / "corrupt.pt"
                corrupt.write_bytes(b"invalid actor")
                request.write_text(
                    json.dumps(
                        {
                            "request_id": "corrupt",
                            "checkpoint": str(corrupt),
                            "sha256": hashlib.sha256(
                                corrupt.read_bytes()
                            ).hexdigest(),
                        }
                    )
                )
                for _ in range(100):
                    await asyncio.sleep(0.05)
                    rejected = json.loads(result.read_text())
                    if rejected["request_id"] == "corrupt":
                        break
                assert rejected["status"] == "rejected"
                assert smoke_replay.actor is installed_actor
                assert smoke_replay.checkpoint_label == replacement.name
        finally:
            server.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server

    asyncio.run(exercise())

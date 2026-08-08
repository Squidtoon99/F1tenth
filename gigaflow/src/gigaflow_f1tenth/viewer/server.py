"""WebSocket + static HTTP server for the local checkpoint viewer."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.viewer.protocol import (
    decode_client_message,
    encode_actor_update,
    encode_error,
    encode_hello,
    encode_tick,
    encode_track,
)
from gigaflow_f1tenth.viewer.replay import (
    CheckpointReplay,
    ViewerError,
    ViewerLaunchArgs,
    checkpoint_architecture,
    checkpoint_condition_dim,
    trunk_mismatch,
    validate_view_args,
)

VIEWER_DIST = Path(__file__).resolve().parents[3] / "viewer" / "dist"
DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[3] / "outputs"
CHECKPOINT_GLOB = "actor_*.pt"


def advance_deadline(deadline: float, period: float, now: float) -> float:
    """Advance an absolute tick deadline past ``now`` without catch-up bursts.

    Missed slots are skipped so emit times stay on a stable wall-clock grid
    rather than compute-time-plus-sleep drift or back-to-back catch-up ticks.
    """
    if period <= 0.0:
        return now
    next_deadline = deadline + period
    while next_deadline <= now:
        next_deadline += period
    return next_deadline


def discover_checkpoints(
    roots: Sequence[Path], cfg: ExperimentConfig
) -> dict[str, Any]:
    """Group the ``actor_*.pt`` files under ``roots`` by the run directory holding them.

    Each entry carries the condition width it was trained at, since the viewer
    runs an actor at its own width. Checkpoints whose trunk cannot run against
    ``cfg`` are excluded from the listing and only counted.
    """
    runs: dict[Path, dict[str, Any]] = {}
    reasons: dict[str, int] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob(CHECKPOINT_GLOB)):
            if not path.is_file():
                continue
            arch = checkpoint_architecture(path)
            mismatch = trunk_mismatch(cfg, arch)
            if mismatch is not None:
                reasons[mismatch] = reasons.get(mismatch, 0) + 1
                continue
            run = runs.get(path.parent)
            if run is None:
                relative = path.parent.relative_to(root)
                run = {
                    "run": root.name if relative == Path(".") else str(relative),
                    "checkpoints": [],
                }
                runs[path.parent] = run
            run["checkpoints"].append(
                {
                    "name": path.name,
                    "path": str(path),
                    "condition_dim": checkpoint_condition_dim(cfg, arch),
                }
            )
    return {
        "runs": [runs[parent] for parent in sorted(runs)],
        "condition_dim": int(cfg.agents.condition_dim),
        "incompatible": {
            "count": sum(reasons.values()),
            "reasons": sorted(reasons),
        },
    }


def resolve_checkpoint_request(roots: Sequence[Path], requested: str) -> Path:
    """Map a client-supplied path onto a discoverable checkpoint under ``roots``."""
    candidate = Path(requested).expanduser().resolve()
    if candidate.match(CHECKPOINT_GLOB) and candidate.is_file():
        for root in roots:
            if candidate.is_relative_to(root.expanduser().resolve()):
                return candidate
    raise ViewerError(f"checkpoint is not available to the viewer: {requested}")


class _ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        replay: CheckpointReplay,
        checkpoint_roots: Sequence[Path],
        **kwargs: Any,
    ) -> None:
        self._replay = replay
        self._checkpoint_roots = checkpoint_roots
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        del fmt, args

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/checkpoints":
            super().do_GET()
            return
        index = discover_checkpoints(
            self._checkpoint_roots, self._replay.base_cfg
        )
        body = json.dumps(
            {
                "active": {
                    "name": self._replay.checkpoint_label,
                    "path": self._replay.checkpoint_path,
                },
                "runs": index["runs"],
                "condition_dim": index["condition_dim"],
                "incompatible": index["incompatible"],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _start_http_server(
    root: Path,
    host: str,
    port: int,
    replay: CheckpointReplay,
    checkpoint_roots: Sequence[Path],
) -> ThreadingHTTPServer:
    handler = functools.partial(
        _ViewerHandler,
        directory=str(root),
        replay=replay,
        checkpoint_roots=checkpoint_roots,
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


async def _broadcast(clients: set[Any], message: str) -> None:
    await asyncio.gather(
        *(client.send(message) for client in tuple(clients)),
        return_exceptions=True,
    )


async def _install_checkpoint(
    replay: CheckpointReplay,
    clients: set[Any],
    checkpoint: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Hot-swap the live actor and tell every connected client about it."""
    sha256 = await asyncio.to_thread(_sha256, checkpoint)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ViewerError("checkpoint hash changed before hot load")
    actor, condition, resolved = await asyncio.to_thread(
        replay.load_actor_candidate, checkpoint
    )
    replay.install_actor(actor, condition, resolved)
    loaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    await _broadcast(
        clients,
        encode_actor_update(
            checkpoint=resolved.name, sha256=sha256, loaded_at=loaded_at
        ),
    )
    print(
        f"[gigaflow view] actor hot-loaded checkpoint={resolved.name} "
        f"sha256={sha256} hidden_state=reset",
        flush=True,
    )
    return {
        "checkpoint": resolved.name,
        "sha256": sha256,
        "loaded_at": loaded_at,
    }


class PolicyUpdateWatcher:
    def __init__(self, request_path: Path) -> None:
        self.request_path = request_path
        name = request_path.name.replace(".request.json", ".result.json")
        self.result_path = request_path.with_name(name)
        self._seen_request = ""

    async def run(
        self, replay: CheckpointReplay, clients: set[Any]
    ) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                request = json.loads(
                    self.request_path.read_text(encoding="utf-8")
                )
                request_id = str(request["request_id"])
                if request_id == self._seen_request:
                    continue
                self._seen_request = request_id
                loaded = await _install_checkpoint(
                    replay,
                    clients,
                    Path(str(request["checkpoint"])),
                    expected_sha256=str(request["sha256"]),
                )
                _atomic_json(
                    self.result_path,
                    {
                        "request_id": request_id,
                        "status": "loaded",
                        "hidden_state_reset": True,
                        "server_pid": os.getpid(),
                        **loaded,
                    },
                )
            except FileNotFoundError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                request_id = locals().get("request_id", "")
                if request_id:
                    _atomic_json(
                        self.result_path,
                        {
                            "request_id": request_id,
                            "status": "rejected",
                            "error": str(exc),
                            "server_pid": os.getpid(),
                        },
                    )
                await _broadcast(
                    clients, encode_error(str(exc), code="checkpoint_error")
                )
                print(
                    f"[gigaflow view] actor hot-load rejected: {exc}",
                    flush=True,
                )


async def _send_scene(websocket: Any, replay: CheckpointReplay) -> None:
    center, left, right = replay.track_geometry()
    await websocket.send(encode_hello(**replay.hello_fields()))
    await websocket.send(encode_track(center=center, left=left, right=right))
    await websocket.send(
        encode_tick(
            step=replay.step_index,
            sim_fps=replay.sim_fps,
            cars=replay.poses(),
            paused=replay.paused,
            t=replay.sim_time,
            control_dt=replay.control_dt,
        )
    )


async def _client_loop(
    websocket: Any,
    replay: CheckpointReplay,
    clients: set[Any],
    checkpoint_roots: Sequence[Path],
) -> None:
    await _send_scene(websocket, replay)
    period = 1.0 / max(replay.control_hz, 1e-6)
    next_tick = time.perf_counter() + period

    while True:
        timeout = max(0.0, next_tick - time.perf_counter())
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            msg = decode_client_message(raw)
            kind = msg["type"]
            if kind == "pause":
                replay.set_paused(True)
            elif kind == "resume":
                replay.set_paused(False)
            elif kind == "reset":
                replay.reset()
            elif kind == "set_track":
                replay.set_track(
                    track=msg.get("track"),
                    track_id=msg.get("track_id"),
                )
                await _send_scene(websocket, replay)
                next_tick = time.perf_counter() + period
                continue
            elif kind == "set_suite":
                replay.set_suite(str(msg["suite"]))
                await _send_scene(websocket, replay)
                next_tick = time.perf_counter() + period
                continue
            elif kind == "set_environment_count":
                replay.set_environment_count(int(msg["environment_count"]))
                await _send_scene(websocket, replay)
                next_tick = time.perf_counter() + period
                continue
            elif kind == "set_obstacle_preset":
                replay.set_obstacle_preset(str(msg["obstacle_preset"]))
                await _send_scene(websocket, replay)
                next_tick = time.perf_counter() + period
                continue
            elif kind == "respawn_obstacles":
                replay.respawn_obstacles()
                await _send_scene(websocket, replay)
                next_tick = time.perf_counter() + period
                continue
            elif kind == "set_checkpoint":
                try:
                    await _install_checkpoint(
                        replay,
                        clients,
                        resolve_checkpoint_request(
                            checkpoint_roots, str(msg["checkpoint"])
                        ),
                    )
                except (ViewerError, OSError) as exc:
                    await websocket.send(
                        encode_error(str(exc), code="checkpoint_error")
                    )
                    print(
                        f"[gigaflow view] actor hot-load rejected: {exc}",
                        flush=True,
                    )
                next_tick = time.perf_counter() + period
                continue
            await websocket.send(
                encode_tick(
                    step=replay.step_index,
                    sim_fps=replay.sim_fps,
                    cars=replay.poses(),
                    paused=replay.paused,
                    t=replay.sim_time,
                    control_dt=replay.control_dt,
                )
            )
            continue
        except (TimeoutError, asyncio.TimeoutError):
            pass

        cars = replay.step() if not replay.paused else replay.poses()
        await websocket.send(
            encode_tick(
                step=replay.step_index,
                sim_fps=replay.sim_fps,
                cars=cars,
                paused=replay.paused,
                t=replay.sim_time,
                control_dt=replay.control_dt,
            )
        )
        now = time.perf_counter()
        next_tick = advance_deadline(next_tick, period, now)


async def _run_ws_server(
    replay: CheckpointReplay,
    host: str,
    port: int,
    policy_update_file: Path | None = None,
    checkpoint_roots: Sequence[Path] = (),
) -> None:
    from websockets.asyncio.server import serve

    clients: set[Any] = set()

    async def handler(websocket: Any) -> None:
        clients.add(websocket)
        try:
            await _client_loop(websocket, replay, clients, checkpoint_roots)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Normal browser tab close / client disconnect.
            name = type(exc).__name__
            if "ConnectionClosed" in name:
                return
            try:
                await websocket.send(
                    encode_error(str(exc), code="replay_error")
                )
            except Exception:
                pass
            raise
        finally:
            clients.discard(websocket)

    async with serve(handler, host, port):
        if policy_update_file is None:
            await asyncio.Future()
        else:
            await PolicyUpdateWatcher(policy_update_file).run(replay, clients)


def run_viewer(args: ViewerLaunchArgs) -> int:
    validated = validate_view_args(args)
    try:
        replay = CheckpointReplay.from_launch_args(validated)
    except ViewerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ViewerError(str(exc)) from exc

    checkpoint_roots = validated.checkpoint_roots or (DEFAULT_CHECKPOINT_ROOT,)
    http_server = None
    if VIEWER_DIST.is_dir() and (VIEWER_DIST / "index.html").is_file():
        http_server = _start_http_server(
            VIEWER_DIST,
            validated.host,
            validated.http_port,
            replay,
            checkpoint_roots,
        )
        url = f"http://{validated.host}:{validated.http_port}/"
        print(
            f"[gigaflow view] serving UI at {url} "
            f"(ws://{validated.host}:{validated.ws_port})",
            flush=True,
        )
        if validated.open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
    else:
        print(
            "[gigaflow view] viewer/dist missing; WebSocket-only mode. "
            "Build UI with: cd viewer && npm install && npm run build",
            flush=True,
        )
        print(
            f"[gigaflow view] ws://{validated.host}:{validated.ws_port}",
            flush=True,
        )

    print(
        f"[gigaflow view] track={replay.track_name} suite={replay.suite} "
        f"seed={replay.seed} device={replay.device} "
        f"checkpoint={replay.checkpoint_label}",
        flush=True,
    )
    try:
        asyncio.run(
            _run_ws_server(
                replay,
                validated.host,
                validated.ws_port,
                validated.policy_update_file,
                checkpoint_roots,
            )
        )
    except KeyboardInterrupt:
        print("[gigaflow view] stopped", flush=True)
    finally:
        if http_server is not None:
            http_server.shutdown()
    return 0

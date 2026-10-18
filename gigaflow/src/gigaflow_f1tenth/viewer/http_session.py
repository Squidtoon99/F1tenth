"""Same replay loop as the WebSocket viewer, polled over HTTP by the garage page."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from gigaflow_f1tenth.viewer.protocol import (
    decode_client_message,
    encode_hello,
    encode_tick,
    encode_track,
)
from gigaflow_f1tenth.viewer.replay import (
    CheckpointReplay,
    ViewerError,
    ViewerLaunchArgs,
)
from gigaflow_f1tenth.viewer.server import advance_deadline


class HttpRaceSession:
    def __init__(self, config: Path, cache_dir: Path, device: str = "cpu") -> None:
        self.config = Path(config)
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.replay: CheckpointReplay | None = None
        self._lock = threading.Lock()
        self._hello: str | None = None
        self._track: str | None = None
        self._tick: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            messages = [
                item
                for item in (self._hello, self._track, self._tick)
                if item is not None
            ]
            return {"ready": self.replay is not None, "messages": messages}

    def apply_raw(self, raw: str) -> None:
        msg = decode_client_message(raw)
        with self._lock:
            self._apply_locked(msg)
            self._publish_locked()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _apply_locked(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "set_lineup":
            paths = [Path(item) for item in msg["checkpoints"]]
            if self.replay is None:
                if not self.config.is_file():
                    raise ViewerError(f"config not found: {self.config}")
                if not self.cache_dir.is_dir():
                    raise ViewerError(
                        f"track cache directory not found: {self.cache_dir}"
                    )
                self.replay = CheckpointReplay.from_launch_args(
                    ViewerLaunchArgs(
                        checkpoint=paths[0],
                        config=self.config,
                        cache_dir=self.cache_dir,
                        suite="solo",
                        seed=0,
                        device=self.device,
                        open_browser=False,
                        track=msg.get("track"),
                    )
                )
                self._ensure_thread()
            self.replay.set_lineup(paths)
            if msg.get("track") and self.replay.track_name != str(msg["track"]):
                self.replay.set_track(track=str(msg["track"]))
            return
        if self.replay is None:
            raise ViewerError("select models to start the race")
        replay = self.replay
        if kind == "pause":
            replay.set_paused(True)
        elif kind == "resume":
            replay.set_paused(False)
        elif kind == "reset":
            replay.reset()
        elif kind == "set_track":
            replay.set_track(track=msg.get("track"), track_id=msg.get("track_id"))
        elif kind == "set_suite":
            replay.set_suite(str(msg["suite"]))
        elif kind == "set_environment_count":
            replay.set_environment_count(int(msg["environment_count"]))
        elif kind == "set_obstacle_preset":
            replay.set_obstacle_preset(str(msg["obstacle_preset"]))
        elif kind == "respawn_obstacles":
            replay.respawn_obstacles()
        elif kind == "set_checkpoint":
            raise ViewerError("use set_lineup to choose the cars in this race")

    def _publish_locked(self) -> None:
        replay = self.replay
        if replay is None:
            return
        self._hello = encode_hello(**replay.hello_fields())
        center, left, right = replay.track_geometry()
        self._track = encode_track(center=center, left=left, right=right)
        cars = replay.poses()
        self._tick = encode_tick(
            step=replay.step_index,
            sim_fps=replay.sim_fps,
            cars=cars,
            paused=replay.paused,
            t=replay.sim_time,
            control_dt=replay.control_dt,
        )

    def _loop(self) -> None:
        period = 0.1
        next_tick = time.perf_counter() + period
        while not self._stop.is_set():
            with self._lock:
                replay = self.replay
                if replay is not None and not replay.paused:
                    replay.step()
                    period = 1.0 / max(replay.control_hz, 1e-6)
                    self._tick = encode_tick(
                        step=replay.step_index,
                        sim_fps=replay.sim_fps,
                        cars=replay.poses(),
                        paused=replay.paused,
                        t=replay.sim_time,
                        control_dt=replay.control_dt,
                    )
            now = time.perf_counter()
            next_tick = advance_deadline(next_tick, period, now)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

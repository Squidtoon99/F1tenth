"""Lean JSON WebSocket protocol for the local checkpoint viewer."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

PROTOCOL_VERSION = 4

# Car row:
# [x, y, yaw, active, speed_mps, vx, vy, yaw_rate, steer_rad,
#  longitudinal_action, steering_action, collision_contact]
CAR_POSE_DIM = 12

_CLIENT_TYPES = frozenset(
    {
        "pause",
        "resume",
        "reset",
        "ping",
        "set_track",
        "set_suite",
        "set_environment_count",
        "set_obstacle_preset",
        "respawn_obstacles",
        "set_checkpoint",
    }
)
_VIEWER_SUITES = ("solo", "head_to_head", "dense")
MIN_DENSE_ENVIRONMENTS = 1
MAX_DENSE_ENVIRONMENTS = 4


def _f32_pairs(points: np.ndarray) -> list[list[float]]:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in arr]


def encode_hello(
    *,
    track: str,
    track_id: int,
    suite: str,
    checkpoint: str,
    seed: int,
    device: str,
    control_hz: float,
    car_length: float,
    car_width: float,
    car_height: float,
    speed_axis_max_mps: float,
    num_cars: int,
    num_obstacles: int,
    obstacle_preset: str,
    obstacle_nonce: int,
    obstacle_presets: Sequence[str],
    obstacles: Sequence[Sequence[float]],
    num_environments: int,
    cars_per_environment: int,
    min_dense_environments: int,
    max_dense_environments: int,
    tracks: Sequence[str] | None = None,
    suites: Sequence[str] | None = None,
) -> str:
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": "hello",
            "track": str(track),
            "track_id": int(track_id),
            "suite": str(suite),
            "checkpoint": str(checkpoint),
            "seed": int(seed),
            "device": str(device),
            "control_hz": float(control_hz),
            "car_length": float(car_length),
            "car_width": float(car_width),
            "car_height": float(car_height),
            "speed_axis_max_mps": float(speed_axis_max_mps),
            "num_cars": int(num_cars),
            "num_obstacles": int(num_obstacles),
            "obstacle_preset": str(obstacle_preset),
            "obstacle_nonce": int(obstacle_nonce),
            "obstacle_presets": [str(p) for p in obstacle_presets],
            "obstacles": [
                [float(row[0]), float(row[1]), float(row[2])] for row in obstacles
            ],
            "num_environments": int(num_environments),
            "cars_per_environment": int(cars_per_environment),
            "min_dense_environments": int(min_dense_environments),
            "max_dense_environments": int(max_dense_environments),
            "tracks": [str(t) for t in (tracks or ())],
            "suites": [str(s) for s in (suites or _VIEWER_SUITES)],
        },
        separators=(",", ":"),
    )


def encode_track(
    *,
    center: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> str:
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": "track",
            "center": _f32_pairs(center),
            "left": _f32_pairs(left),
            "right": _f32_pairs(right),
        },
        separators=(",", ":"),
    )


def _pad_cars(cars: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(cars, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("cars must be shaped [N, >=4]")
    if arr.shape[1] < CAR_POSE_DIM:
        pad = np.zeros((arr.shape[0], CAR_POSE_DIM - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate((arr, pad), axis=1)
    return arr[:, :CAR_POSE_DIM]


def encode_tick(
    *,
    step: int,
    sim_fps: float,
    cars: np.ndarray | Sequence[Sequence[float]],
    paused: bool = False,
    t: float | None = None,
    control_dt: float | None = None,
) -> str:
    arr = _pad_cars(cars)
    dt = 0.1 if control_dt is None else float(control_dt)
    if dt <= 0.0:
        raise ValueError("control_dt must be positive")
    sim_t = float(step) * dt if t is None else float(t)
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": "tick",
            "step": int(step),
            "t": sim_t,
            "control_dt": dt,
            "sim_fps": float(sim_fps),
            "paused": bool(paused),
            "cars": [
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9]),
                    float(row[10]),
                    float(row[11]),
                ]
                for row in arr
            ],
        },
        separators=(",", ":"),
    )


def encode_error(message: str, *, code: str = "error") -> str:
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": "error",
            "code": str(code),
            "message": str(message),
        },
        separators=(",", ":"),
    )


def encode_actor_update(
    *,
    checkpoint: str,
    sha256: str,
    loaded_at: str,
) -> str:
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": "actor_update",
            "checkpoint": str(checkpoint),
            "sha256": str(sha256),
            "loaded_at": str(loaded_at),
            "hidden_state_reset": True,
        },
        separators=(",", ":"),
    )


def decode_client_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("client message must be a JSON object")
    version = int(payload.get("v", -1))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {version}")
    msg_type = payload.get("type")
    if msg_type not in _CLIENT_TYPES:
        raise ValueError(f"unsupported client message type: {msg_type!r}")
    out: dict[str, Any] = {"v": PROTOCOL_VERSION, "type": str(msg_type)}
    if msg_type == "set_track":
        if "track_id" in payload and payload["track_id"] is not None:
            out["track_id"] = int(payload["track_id"])
        elif "track" in payload and payload["track"] is not None:
            out["track"] = str(payload["track"])
        else:
            raise ValueError("set_track requires track or track_id")
    elif msg_type == "set_suite":
        suite = payload.get("suite")
        if suite not in _VIEWER_SUITES:
            raise ValueError(
                f"unsupported suite {suite!r}; expected one of {_VIEWER_SUITES}"
            )
        out["suite"] = str(suite)
    elif msg_type == "set_environment_count":
        count = int(payload.get("environment_count", 0))
        if not MIN_DENSE_ENVIRONMENTS <= count <= MAX_DENSE_ENVIRONMENTS:
            raise ValueError(
                "environment_count must be between "
                f"{MIN_DENSE_ENVIRONMENTS} and {MAX_DENSE_ENVIRONMENTS}"
            )
        out["environment_count"] = count
    elif msg_type == "set_obstacle_preset":
        preset = str(payload.get("obstacle_preset", ""))
        if preset not in {"off", "light", "heavy"}:
            raise ValueError(
                "unsupported obstacle preset; expected off, light, or heavy"
            )
        out["obstacle_preset"] = preset
    elif msg_type == "set_checkpoint":
        checkpoint = str(payload.get("checkpoint", ""))
        if not checkpoint:
            raise ValueError("set_checkpoint requires a checkpoint path")
        out["checkpoint"] = checkpoint
    return out

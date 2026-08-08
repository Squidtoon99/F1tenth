"""Deterministic local checkpoint replay for the live viewer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gigaflow_f1tenth import model as model_mod
from gigaflow_f1tenth.config import ExperimentConfig, config_from_dict, config_to_dict
from gigaflow_f1tenth.evaluation import (
    _suite_world_overrides,
    load_actor_from_checkpoint,
)
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.model import (
    ActorShapes,
    ConditionedLidarGRUActor,
    architecture_metadata,
)
from gigaflow_f1tenth.rewards import deployment_style, styles_to_condition_batch
from gigaflow_f1tenth.sim.spawn import apply_static_pins, place_static_opponents
from gigaflow_f1tenth.tracks import PackedTrackAtlasView, TrackError, load_atlas
from gigaflow_f1tenth.visualization import CUBOID_HEIGHT_FRAC, track_lines
from gigaflow_f1tenth.viewer.protocol import (
    MAX_DENSE_ENVIRONMENTS,
    MIN_DENSE_ENVIRONMENTS,
)

VIEWER_SUITES = ("solo", "head_to_head", "dense")
VIEWER_SUITE_SET = frozenset(VIEWER_SUITES)
OBSTACLE_PRESETS = {"off": 0, "light": 2, "heavy": 4}
TRUNK_COMPAT_KEYS = (
    "sensor_obs_dim",
    "lidar_dim",
    "proprio_dim",
    "action_dim",
    "gru_hidden_dim",
    "cnn_projection_dim",
    "mlp_sizes",
)
SENSOR_NORMALIZER_BUFFERS = frozenset(
    {
        "sensor_normalizer.mean",
        "sensor_normalizer.m2",
        "sensor_normalizer.count",
    }
)
_ARCHITECTURE_CACHE: dict[Path, tuple[tuple[int, int], dict[str, Any] | None]] = {}


class ViewerError(RuntimeError):
    """User-facing viewer configuration or checkpoint error."""


@dataclass(frozen=True)
class ViewerLaunchArgs:
    checkpoint: Path
    config: Path
    cache_dir: Path
    suite: str
    seed: int
    device: str
    track: str | None = None
    track_id: int | None = None
    host: str = "127.0.0.1"
    ws_port: int = 8765
    http_port: int = 8766
    open_browser: bool = True
    policy_update_file: Path | None = None
    checkpoint_roots: tuple[Path, ...] = ()


def validate_view_args(args: ViewerLaunchArgs) -> ViewerLaunchArgs:
    if not args.checkpoint.is_file():
        raise ViewerError(f"checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        raise ViewerError(f"config not found: {args.config}")
    if args.suite not in VIEWER_SUITE_SET:
        raise ViewerError(
            f"unsupported suite {args.suite!r}; expected one of "
            f"{list(VIEWER_SUITES)}"
        )
    if args.seed < 0:
        raise ViewerError("seed must be non-negative")
    device = str(args.device)
    if not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ViewerError(
            f"unsupported device {device!r}; expected cpu|cuda|cuda:N"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ViewerError(
            f"device {device!r} requested but CUDA is not available"
        )
    if args.track is not None and args.track_id is not None:
        raise ViewerError("pass only one of --track or --track-id")
    if args.ws_port <= 0 or args.http_port <= 0:
        raise ViewerError("ws-port and http-port must be positive")
    if args.ws_port == args.http_port:
        raise ViewerError("ws-port and http-port must differ")
    cache = args.cache_dir.expanduser()
    if not cache.exists():
        raise ViewerError(f"track cache directory not found: {cache}")
    roots = tuple(root.expanduser().resolve() for root in args.checkpoint_roots)
    for root in roots:
        if not root.is_dir():
            raise ViewerError(f"checkpoint root not found: {root}")
    atlas_npz = cache / "atlas.npz"
    manifest = cache / "manifest.json"
    if not atlas_npz.is_file() or not manifest.is_file():
        raise ViewerError(
            f"track atlas missing under {cache} "
            "(need atlas.npz and manifest.json; run gigaflow prepare-tracks)"
        )
    return ViewerLaunchArgs(
        checkpoint=args.checkpoint.resolve(),
        config=args.config.resolve(),
        cache_dir=cache.resolve(),
        suite=str(args.suite),
        seed=int(args.seed),
        device=device,
        track=None if args.track is None else str(args.track),
        track_id=None if args.track_id is None else int(args.track_id),
        host=str(args.host),
        ws_port=int(args.ws_port),
        http_port=int(args.http_port),
        open_browser=bool(args.open_browser),
        policy_update_file=(
            None
            if args.policy_update_file is None
            else args.policy_update_file.expanduser().resolve()
        ),
        checkpoint_roots=roots,
    )


def select_track_view(
    view: PackedTrackAtlasView, track_index: int
) -> PackedTrackAtlasView:
    """Return a one-track atlas view so the sim runs on a forced track."""
    n = int(view.num_tracks)
    if track_index < 0 or track_index >= n:
        raise ViewerError(
            f"track_id {track_index} out of range for atlas with {n} tracks"
        )
    start = int(view.offsets[track_index])
    stop = int(view.offsets[track_index + 1])
    lut_a = int(view.lut_offsets[track_index])
    lut_b = int(view.lut_offsets[track_index + 1])
    edt_a = int(view.edt_offsets[track_index])
    edt_b = int(view.edt_offsets[track_index + 1])
    return PackedTrackAtlasView(
        num_tracks=1,
        offsets=np.asarray([0, stop - start], dtype=np.int32),
        centerline_xy=np.asarray(view.centerline_xy[start:stop], dtype=np.float32),
        tangents_xy=np.asarray(view.tangents_xy[start:stop], dtype=np.float32),
        widths_rl=np.asarray(view.widths_rl[start:stop], dtype=np.float32),
        cum_length=np.asarray(view.cum_length[start:stop], dtype=np.float32),
        lengths=np.asarray([view.lengths[track_index]], dtype=np.float32),
        track_ids=(str(view.track_ids[track_index]),),
        lut_offsets=np.asarray([0, lut_b - lut_a], dtype=np.int32),
        nearest_segment_lut=np.asarray(
            view.nearest_segment_lut[lut_a:lut_b], dtype=np.int32
        ),
        lut_width=np.asarray([view.lut_width[track_index]], dtype=np.int32),
        lut_height=np.asarray([view.lut_height[track_index]], dtype=np.int32),
        lut_origin_xy=np.asarray(
            [view.lut_origin_xy[track_index]], dtype=np.float32
        ),
        lut_resolution=np.asarray(
            [view.lut_resolution[track_index]], dtype=np.float32
        ),
        edt_offsets=np.asarray([0, edt_b - edt_a], dtype=np.int32),
        edt_distance=np.asarray(
            view.edt_distance[edt_a:edt_b], dtype=np.float32
        ),
        edt_width=np.asarray([view.edt_width[track_index]], dtype=np.int32),
        edt_height=np.asarray([view.edt_height[track_index]], dtype=np.int32),
        edt_origin_xy=np.asarray(
            [view.edt_origin_xy[track_index]], dtype=np.float32
        ),
        edt_resolution=np.asarray(
            [view.edt_resolution[track_index]], dtype=np.float32
        ),
        capacity=np.asarray([view.capacity[track_index]], dtype=np.int32),
    )


def resolve_track_index(
    view: PackedTrackAtlasView,
    *,
    track: str | None,
    track_id: int | None,
) -> int:
    if track_id is not None:
        idx = int(track_id)
        if idx < 0 or idx >= int(view.num_tracks):
            raise ViewerError(
                f"track_id {idx} out of range for atlas with "
                f"{view.num_tracks} tracks"
            )
        return idx
    if track is None:
        return 0
    names = [str(n) for n in view.track_ids]
    lowered = {n.lower(): i for i, n in enumerate(names)}
    key = str(track).lower()
    if key in lowered:
        return lowered[key]
    if track in names:
        return names.index(track)
    raise ViewerError(
        f"track {track!r} not found in atlas; known tracks: {', '.join(names)}"
    )


def _viewer_suite_config(
    cfg: ExperimentConfig,
    suite: str,
    seed: int,
    dense_environment_count: int = 1,
    obstacles_per_environment: int = 0,
) -> ExperimentConfig:
    raw = _suite_world_overrides(suite, cfg)
    raw["seed"] = int(seed)
    worlds = dict(raw["worlds"])
    worlds["num_worlds"] = (
        int(dense_environment_count) if suite == "dense" else 1
    )
    worlds["max_agents_per_world"] = (
        int(worlds["max_agents_per_world"]) + int(obstacles_per_environment)
    )
    raw["worlds"] = worlds
    evaluation = dict(raw["evaluation"])
    evaluation["num_worlds"] = int(worlds["num_worlds"])
    evaluation["sync_no_respawn"] = True
    raw["evaluation"] = evaluation
    # Viewer builds a simulator only, never a trainer, so the PPO minibatch
    # and rollout-memory budgets (sized for a training update) do not apply.
    return config_from_dict(raw, check_training_budget=False)


def trunk_mismatch(
    cfg: ExperimentConfig, arch: Mapping[str, Any] | None
) -> str | None:
    """Explain why an actor's trunk cannot run against ``cfg``.

    ``condition_dim`` is deliberately excluded: it sets the width of one layer
    that is embedded to a fixed size before the trunk, so an actor recorded at
    another width still loads (see ``load_checkpoint_actor``).
    """
    if not isinstance(arch, Mapping):
        return None
    expected = architecture_metadata(cfg)
    for key in TRUNK_COMPAT_KEYS:
        if key not in arch:
            continue
        if arch[key] != expected.get(key):
            return (
                f"checkpoint/config mismatch on {key}: "
                f"checkpoint={arch[key]!r} config={expected.get(key)!r}"
            )
    return None


def checkpoint_architecture(checkpoint: Path) -> Mapping[str, Any] | None:
    """Read a checkpoint's architecture header without materializing weights.

    Memoized on ``(mtime_ns, size)`` so checkpoints a run writes later are
    still picked up. A file we cannot read reports ``None``; the load path
    remains responsible for rejecting it.
    """
    try:
        stat = checkpoint.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _ARCHITECTURE_CACHE.get(checkpoint)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        payload = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
    except Exception:  # noqa: BLE001
        payload = None
    arch: dict[str, Any] | None = None
    header = payload.get("actor_architecture") if isinstance(payload, Mapping) else None
    if isinstance(header, Mapping):
        arch = dict(header)
    _ARCHITECTURE_CACHE[checkpoint] = (stamp, arch)
    return arch


def checkpoint_condition_dim(
    cfg: ExperimentConfig, arch: Mapping[str, Any] | None
) -> int:
    if isinstance(arch, Mapping) and "condition_dim" in arch:
        return int(arch["condition_dim"])
    return int(cfg.agents.condition_dim)


def load_checkpoint_actor(
    cfg: ExperimentConfig, checkpoint: Path, device: str
) -> tuple[Any, torch.Tensor | None]:
    """Load an actor at the condition width recorded in its own artifact.

    Returns the actor plus, when that width differs from the live schema, the
    conservative deployment condition the checkpoint's own run froze into it.
    """
    payload = _assert_checkpoint_matches_config(cfg, checkpoint)
    arch = payload.get("actor_architecture")
    width = checkpoint_condition_dim(cfg, arch)
    if width == int(cfg.agents.condition_dim):
        try:
            actor = load_actor_from_checkpoint(cfg, checkpoint, device=device)
        except Exception as exc:  # noqa: BLE001
            raise ViewerError(
                f"failed to load actor from {checkpoint}: {exc}"
            ) from exc
        return actor, None
    try:
        return _load_native_condition_actor(payload, arch, width, device)
    except ViewerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ViewerError(
            f"failed to load actor from {checkpoint}: {exc}"
        ) from exc


def _load_native_condition_actor(
    payload: Mapping[str, Any],
    arch: Mapping[str, Any],
    width: int,
    device: str,
) -> tuple[Any, torch.Tensor]:
    schema = payload.get("condition_schema")
    if isinstance(schema, Mapping) and int(schema.get("condition_dim", width)) != width:
        raise ViewerError(
            "checkpoint architecture and condition schema disagree on condition_dim"
        )
    condition = payload.get("deployment_condition_normalized")
    if condition is None or len(condition) != width:
        raise ViewerError(
            f"checkpoint records condition_dim={width} but carries no matching "
            "deployment condition vector, so it cannot be fed faithfully"
        )
    state = payload.get("actor_state_dict")
    if not isinstance(state, Mapping):
        raise ViewerError("checkpoint missing actor weights")
    shapes = ActorShapes(
        sensor_obs_dim=int(arch["sensor_obs_dim"]),
        lidar_dim=int(arch["lidar_dim"]),
        proprio_dim=int(arch["proprio_dim"]),
        condition_dim=width,
        action_dim=int(arch["action_dim"]),
        gru_hidden_dim=int(arch["gru_hidden_dim"]),
        cnn_projection_dim=int(arch["cnn_projection_dim"]),
        mlp_sizes=tuple(int(v) for v in arch["mlp_sizes"]),
    )
    # The actor class asserts the live schema width; this one carries its own.
    live_schema = model_mod.CONDITION_DIM
    model_mod.CONDITION_DIM = width
    try:
        actor = ConditionedLidarGRUActor(shapes)
    finally:
        model_mod.CONDITION_DIM = live_schema
    loaded = actor.load_state_dict(state, strict=False)
    unexpected = sorted(loaded.unexpected_keys)
    # Actors from before the sensor normalizer existed keep the identity
    # transform a freshly built normalizer applies, which is the input pipeline
    # they were trained under.
    missing = sorted(set(loaded.missing_keys) - SENSOR_NORMALIZER_BUFFERS)
    if unexpected or missing:
        raise ViewerError(
            f"checkpoint weights do not fit a condition_dim={width} actor: "
            f"missing={missing} unexpected={unexpected}"
        )
    if loaded.missing_keys and payload.get("sensor_normalizer"):
        raise ViewerError(
            "checkpoint carries sensor normalizer stats but no normalizer buffers"
        )
    return actor.to(device), torch.as_tensor(
        [float(v) for v in condition], dtype=torch.float32, device=device
    ).unsqueeze(0)


def _assert_checkpoint_matches_config(
    cfg: ExperimentConfig, checkpoint: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ViewerError(f"unsupported checkpoint payload: {checkpoint}")
    mismatch = trunk_mismatch(cfg, payload.get("actor_architecture"))
    if mismatch is not None:
        raise ViewerError(mismatch)
    if "actor_state_dict" not in payload and not (
        "ppo" in payload
        and isinstance(payload["ppo"], dict)
        and "actor" in payload["ppo"]
    ):
        raise ViewerError("checkpoint missing actor weights")
    return payload


class CheckpointReplay:
    """Runs deterministic local sim steps and exposes pose frames for streaming."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        atlas_full: PackedTrackAtlasView,
        actor: Any,
        suite: str,
        seed: int,
        device: str,
        checkpoint: Path,
        source_track_id: int,
        condition: torch.Tensor | None = None,
    ) -> None:
        if suite not in VIEWER_SUITE_SET:
            raise ViewerError(f"unsupported suite: {suite}")
        self.base_cfg = cfg
        self.atlas_full = atlas_full
        self.available_tracks = tuple(str(t) for t in atlas_full.track_ids)
        self.suite = suite
        self.seed = int(seed)
        self.device = str(device)
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_label = checkpoint.name
        self.source_track_id = int(source_track_id)
        self.track_name = self.available_tracks[self.source_track_id]
        self.actor = actor
        self._condition = condition
        self.dense_environment_count = 1
        self.obstacle_preset = "off"
        self.obstacle_nonce = 0
        self.paused = False
        self.step_index = 0
        self.sim_fps = 0.0
        self.atlas = select_track_view(atlas_full, self.source_track_id)
        self._suite_cfg = _viewer_suite_config(cfg, suite, seed)
        self.sim = build_simulator(
            self._suite_cfg, self.atlas, self.device, sync_no_respawn=True
        )
        self._fixed_obstacles: list[tuple[int, dict[str, float | int]]] = []
        self._obstacle_mask: torch.Tensor | None = None
        self._obstacle_pin: tuple[torch.Tensor, ...] | None = None
        self._hidden: torch.Tensor | None = None
        self._last_actions: torch.Tensor | None = None
        self.reset()

    @classmethod
    def from_launch_args(
        cls,
        args: ViewerLaunchArgs,
        *,
        cfg: ExperimentConfig | None = None,
        atlas: PackedTrackAtlasView | None = None,
    ) -> "CheckpointReplay":
        validated = validate_view_args(args)
        if cfg is None:
            from gigaflow_f1tenth.config import load_config

            cfg = load_config(validated.config)
        raw = config_to_dict(cfg)
        tracks = dict(raw["tracks"])
        tracks["manifest_path"] = str(validated.cache_dir / "manifest.json")
        raw["tracks"] = tracks
        if "wandb" in raw:
            wb = dict(raw["wandb"])
            wb["enabled"] = False
            wb["mode"] = "disabled"
            raw["wandb"] = wb
        cfg = config_from_dict(raw)

        if atlas is None:
            try:
                atlas_full = load_atlas(str(validated.cache_dir)).view()
            except TrackError as exc:
                raise ViewerError(str(exc)) from exc
        else:
            atlas_full = atlas

        if int(atlas_full.num_tracks) < int(cfg.tracks.num_tracks):
            raise ViewerError(
                f"atlas has {atlas_full.num_tracks} tracks but "
                f"config.tracks.num_tracks={cfg.tracks.num_tracks}"
            )

        track_index = resolve_track_index(
            atlas_full, track=validated.track, track_id=validated.track_id
        )
        actor, condition = load_checkpoint_actor(
            cfg, validated.checkpoint, validated.device
        )
        actor.eval()
        return cls(
            cfg,
            atlas_full=atlas_full,
            actor=actor,
            suite=validated.suite,
            seed=validated.seed,
            device=validated.device,
            checkpoint=validated.checkpoint,
            source_track_id=track_index,
            condition=condition,
        )

    @property
    def control_hz(self) -> float:
        return float(self._suite_cfg.agents.control_hz)

    @property
    def control_dt(self) -> float:
        return 1.0 / max(self.control_hz, 1e-6)

    @property
    def sim_time(self) -> float:
        return float(self.step_index) * self.control_dt

    @property
    def car_length(self) -> float:
        return float(self._suite_cfg.agents.car_length_m)

    @property
    def car_width(self) -> float:
        return float(self._suite_cfg.agents.car_width_m)

    @property
    def car_height(self) -> float:
        return float(CUBOID_HEIGHT_FRAC * self.car_length)

    @property
    def speed_axis_max_mps(self) -> float:
        params = self.sim.vehicle_params
        style = deployment_style(
            self._suite_cfg.evaluation.conservative_deployment_style
        )
        drive_scale = float(style.drive_scale)
        drag = float(params.dragcoeff)
        drive_force = min(
            float(params.f_drive_max),
            float(params.tire_mu * params.mass * params.gravity),
        )
        force_limited_speed = (drive_scale * drive_force / drag) ** 0.5
        power_crossover_speed = float(params.power_max) / drive_force
        if force_limited_speed <= power_crossover_speed:
            return force_limited_speed
        return (drive_scale * float(params.power_max) / drag) ** (1.0 / 3.0)

    @property
    def num_cars(self) -> int:
        return self.num_environments * self.cars_per_environment

    @property
    def num_environments(self) -> int:
        return int(world_slot_layout(self._suite_cfg).num_worlds)

    @property
    def cars_per_environment(self) -> int:
        if self.suite == "solo":
            return 1
        if self.suite == "head_to_head":
            return min(2, int(self.base_cfg.worlds.max_agents_per_world))
        return int(self.base_cfg.worlds.max_agents_per_world)

    @property
    def obstacles_per_environment(self) -> int:
        return int(OBSTACLE_PRESETS[self.obstacle_preset])

    @property
    def num_obstacles(self) -> int:
        return self.num_environments * self.obstacles_per_environment

    def _racer_indices(self) -> torch.Tensor:
        stride = int(self.sim.layout.max_agents_per_world)
        indices = [
            world * stride + slot
            for world in range(self.num_environments)
            for slot in range(self.cars_per_environment)
        ]
        return torch.as_tensor(indices, device=self.device, dtype=torch.long)

    def _obstacle_indices(self) -> torch.Tensor:
        stride = int(self.sim.layout.max_agents_per_world)
        indices = [
            world * stride + self.cars_per_environment + slot
            for world in range(self.num_environments)
            for slot in range(self.obstacles_per_environment)
        ]
        return torch.as_tensor(indices, device=self.device, dtype=torch.long)

    def track_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return track_lines(self.atlas, 0)

    def poses(self) -> np.ndarray:
        t = self.sim.buffers.torch_arrays
        indices = self._racer_indices()
        vx = t.vx.to(dtype=torch.float32)
        vy = t.vy.to(dtype=torch.float32)
        speed = torch.sqrt(vx * vx + vy * vy)
        if self._last_actions is None:
            actions = torch.zeros(
                (self.sim.layout.num_slots, 2), device=vx.device, dtype=torch.float32
            )
        else:
            actions = self._last_actions
        return (
            torch.stack(
                (
                    t.x,
                    t.y,
                    t.yaw,
                    t.active.to(dtype=torch.float32),
                    speed,
                    vx,
                    vy,
                    t.yaw_rate.to(dtype=torch.float32),
                    t.steer.to(dtype=torch.float32),
                    actions[:, 0],
                    actions[:, 1],
                    (
                        (t.contact > 0) | (t.wall_contact > 0)
                    ).to(dtype=torch.float32),
                ),
                dim=1,
            ).index_select(0, indices)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def obstacle_poses(self) -> np.ndarray:
        indices = self._obstacle_indices()
        if indices.numel() == 0:
            return np.empty((0, 3), dtype=np.float32)
        t = self.sim.buffers.torch_arrays
        return (
            torch.stack((t.x, t.y, t.yaw), dim=1)
            .index_select(0, indices)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def _rebuild_sim(self) -> None:
        self.atlas = select_track_view(self.atlas_full, self.source_track_id)
        self.track_name = self.available_tracks[self.source_track_id]
        self._suite_cfg = _viewer_suite_config(
            self.base_cfg,
            self.suite,
            self.seed,
            self.dense_environment_count,
            self.obstacles_per_environment,
        )
        self.sim = build_simulator(
            self._suite_cfg, self.atlas, self.device, sync_no_respawn=True
        )
        self.reset()

    def set_track(
        self, *, track: str | None = None, track_id: int | None = None
    ) -> None:
        idx = resolve_track_index(
            self.atlas_full, track=track, track_id=track_id
        )
        if idx == self.source_track_id:
            self.reset()
            return
        self.source_track_id = idx
        self._rebuild_sim()

    def set_suite(self, suite: str) -> None:
        if suite not in VIEWER_SUITE_SET:
            raise ViewerError(
                f"unsupported suite {suite!r}; expected one of "
                f"{list(VIEWER_SUITES)}"
            )
        if suite == self.suite:
            self.reset()
            return
        self.suite = suite
        self._rebuild_sim()

    def set_environment_count(self, count: int) -> None:
        count = int(count)
        if self.suite != "dense":
            raise ViewerError("environment count can only be changed in dense suite")
        if not MIN_DENSE_ENVIRONMENTS <= count <= MAX_DENSE_ENVIRONMENTS:
            raise ViewerError(
                "environment count must be between "
                f"{MIN_DENSE_ENVIRONMENTS} and {MAX_DENSE_ENVIRONMENTS}"
            )
        if count == self.dense_environment_count:
            self.reset()
            return
        self.dense_environment_count = count
        self._rebuild_sim()

    def set_obstacle_preset(self, preset: str) -> None:
        if preset not in OBSTACLE_PRESETS:
            raise ViewerError(
                f"unsupported obstacle preset {preset!r}; expected one of "
                f"{list(OBSTACLE_PRESETS)}"
            )
        if preset == self.obstacle_preset:
            self.reset()
            return
        self.obstacle_preset = preset
        self.obstacle_nonce = 0
        self._rebuild_sim()

    def respawn_obstacles(self) -> None:
        if self.obstacles_per_environment == 0:
            raise ViewerError("cannot respawn obstacles while preset is off")
        self.obstacle_nonce += 1
        self.reset()

    def load_actor_candidate(
        self, checkpoint: Path
    ) -> tuple[Any, torch.Tensor | None, Path]:
        checkpoint = checkpoint.resolve()
        actor, condition = load_checkpoint_actor(
            self.base_cfg, checkpoint, self.device
        )
        actor.eval()
        actor.initial_hidden(self.sim.layout.num_slots, self.device)
        return actor, condition, checkpoint

    def install_actor(
        self, actor: Any, condition: torch.Tensor | None, checkpoint: Path
    ) -> None:
        hidden = actor.initial_hidden(self.sim.layout.num_slots, self.device)
        self.actor = actor
        self._condition = condition
        self._hidden = hidden
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_label = checkpoint.name

    def reset(self) -> np.ndarray:
        self.sim.reset_all(self.seed)
        t = self.sim.buffers.torch_arrays
        t.active.zero_()
        t.trainable.zero_()
        racer_indices = self._racer_indices()
        racer_mask = torch.zeros(
            self.sim.layout.num_slots,
            device=self.device,
            dtype=torch.bool,
        )
        racer_mask[racer_indices] = True
        self.sim.reset_agents(racer_mask, self.seed)
        self._place_obstacles()
        style = deployment_style(
            self._suite_cfg.evaluation.conservative_deployment_style
        )
        self.sim.apply_styles(
            [style for _ in range(self.sim.layout.num_slots)]
        )
        n = self.sim.layout.num_slots
        self._hidden = self.actor.initial_hidden(n, self.device)
        self._last_actions = torch.zeros(
            (n, 2), device=self.device, dtype=torch.float32
        )
        self.step_index = 0
        self.sim_fps = 0.0
        self.paused = False
        self.sim.rebuild_sensors()
        return self.poses()

    def _place_obstacles(self) -> None:
        self._fixed_obstacles = []
        self._obstacle_mask = None
        self._obstacle_pin = None
        if self.obstacles_per_environment == 0:
            return
        t = self.sim.buffers.torch_arrays
        stride = int(self.sim.layout.max_agents_per_world)
        n = self.sim.layout.num_slots
        device = self.device
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        pin_x = torch.zeros(n, dtype=torch.float32, device=device)
        pin_y = torch.zeros(n, dtype=torch.float32, device=device)
        pin_yaw = torch.zeros(n, dtype=torch.float32, device=device)
        pin_segment = torch.zeros(n, dtype=torch.int32, device=device)
        pin_s = torch.zeros(n, dtype=torch.float32, device=device)
        pin_ey = torch.zeros(n, dtype=torch.float32, device=device)
        pin_boundary = torch.zeros(n, dtype=torch.float32, device=device)
        for world in range(self.num_environments):
            racer_poses = [
                (
                    float(t.x[world * stride + slot]),
                    float(t.y[world * stride + slot]),
                    float(t.yaw[world * stride + slot]),
                )
                for slot in range(self.cars_per_environment)
            ]
            poses = place_static_opponents(
                self.sim.geom,
                int(t.track_id[world * stride]),
                self.obstacles_per_environment,
                seed=(
                    self.seed
                    + 104729 * world
                    + 1009 * self.source_track_id
                    + 15485863 * self.obstacle_nonce
                ),
                racer_poses=racer_poses,
                car_length=self.car_length,
                car_width=self.car_width,
            )
            if len(poses) != self.obstacles_per_environment:
                raise ViewerError(
                    f"could not safely place {self.obstacles_per_environment} "
                    f"{self.obstacle_preset} obstacles on {self.track_name}"
                )
            for local, pose in enumerate(poses):
                flat = world * stride + self.cars_per_environment + local
                mask[flat] = True
                pin_x[flat] = float(pose["x"])
                pin_y[flat] = float(pose["y"])
                pin_yaw[flat] = float(pose["yaw"])
                pin_segment[flat] = int(pose["segment"])
                pin_s[flat] = float(pose["s"])
                pin_ey[flat] = float(pose["ey"])
                pin_boundary[flat] = float(pose["boundary_distance"])
                self._fixed_obstacles.append((flat, pose))
        self._obstacle_mask = mask
        self._obstacle_pin = (pin_x, pin_y, pin_yaw, pin_segment, pin_s, pin_ey, pin_boundary)
        apply_static_pins(t, mask, *self._obstacle_pin, reset_contact=True)

    def _restore_obstacles(self) -> None:
        if self._obstacle_mask is None:
            return
        t = self.sim.buffers.torch_arrays
        assert self._obstacle_pin is not None
        apply_static_pins(t, self._obstacle_mask, *self._obstacle_pin, reset_contact=False)

    def _policy_actions(self) -> torch.Tensor:
        assert self._hidden is not None
        n = self.sim.layout.num_slots
        obs = self.sim.action_observation().to(self.device)
        if self._condition is None:
            cond = torch.as_tensor(
                styles_to_condition_batch(self.sim.styles, normalize=True),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            # Every slot carries the one pinned conservative style, so the actor
            # gets the deployment condition its own run froze at its own width.
            cond = self._condition.expand(n, -1)
        active = (
            (self.sim.buffers.torch_arrays.active > 0)
            & (self.sim.buffers.torch_arrays.trainable > 0)
        ).to(
            device=self.device
        )
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        actions = torch.zeros(n, 2, device=self.device, dtype=torch.float32)
        obstacle_indices = self._obstacle_indices()
        if obstacle_indices.numel() > 0:
            actions[obstacle_indices, 0] = -1.0
        if idx.numel() == 0:
            return actions
        with torch.no_grad():
            out = self.actor.forward(
                obs.index_select(0, idx),
                cond.index_select(0, idx),
                self._hidden.index_select(0, idx),
                reset_mask=None,
                deterministic=True,
            )
        actions.index_copy_(0, idx, out.actions.detach())
        next_hidden = self._hidden.clone()
        next_hidden.index_copy_(0, idx, out.hidden.detach())
        self._hidden = next_hidden
        return actions

    def _step_simulator(self, actions: torch.Tensor) -> dict[str, Any]:
        out = self.sim.step(actions)
        self._restore_obstacles()
        self.sim.rebuild_sensors()
        return out

    def step(self) -> np.ndarray:
        if self.paused:
            return self.poses()
        t0 = time.perf_counter()
        actions = self._policy_actions()
        out = self._step_simulator(actions)
        self._last_actions = actions
        elapsed = max(time.perf_counter() - t0, 1e-9)
        self.sim_fps = 1.0 / elapsed
        self.step_index += 1
        done = out["done"].to(dtype=torch.bool)
        if bool(done.index_select(0, self._racer_indices()).all()):
            self.reset()
            self.paused = False
        return self.poses()

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def hello_fields(self) -> dict[str, Any]:
        return {
            "track": self.track_name,
            "track_id": self.source_track_id,
            "suite": self.suite,
            "checkpoint": self.checkpoint_label,
            "seed": self.seed,
            "device": self.device,
            "control_hz": self.control_hz,
            "car_length": self.car_length,
            "car_width": self.car_width,
            "car_height": self.car_height,
            "speed_axis_max_mps": self.speed_axis_max_mps,
            "num_cars": self.num_cars,
            "num_obstacles": self.num_obstacles,
            "obstacle_preset": self.obstacle_preset,
            "obstacle_nonce": self.obstacle_nonce,
            "obstacle_presets": list(OBSTACLE_PRESETS),
            "obstacles": self.obstacle_poses().tolist(),
            "num_environments": self.num_environments,
            "cars_per_environment": self.cars_per_environment,
            "min_dense_environments": MIN_DENSE_ENVIRONMENTS,
            "max_dense_environments": MAX_DENSE_ENVIRONMENTS,
            "tracks": list(self.available_tracks),
            "suites": list(VIEWER_SUITES),
        }

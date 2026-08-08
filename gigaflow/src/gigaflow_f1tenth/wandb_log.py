"""Optional Weights & Biases logging for gigaflow training runs.

Local JSON / checkpoint files remain the authority. W&B is initialized only when
enabled, never stores API keys, and logging failures are recoverable.
Cadenced ``wandb.log`` is asynchronous — never synchronized on the sim hot path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gigaflow_f1tenth.config import ExperimentConfig, WandbConfig, config_to_dict

RUN_META_FILENAME = "run_meta.json"
EVAL_SOURCE_STEP_METRIC = "eval/source_step"
EVAL_METRIC_GLOB = "eval/*"


def should_report_train_metrics(update_index: int, report_interval_updates: int) -> bool:
    """Cadence for local JSON + W&B train metrics.

    Always report the first completed update so large-scale runs are not silent
    until ``report_interval_updates`` (system metrics alone look like a dead app
    logger). Thereafter follow the configured interval.
    """
    interval = max(1, int(report_interval_updates))
    idx = int(update_index)
    if idx <= 0:
        return False
    return idx == 1 or (idx % interval == 0)


_PPO_METRIC_KEYS = frozenset(
    {
        "retention",
        "policy_loss",
        "ppo_surrogate_loss",
        "entropy_loss",
        "entropy_coef",
        "value_loss",
        "entropy",
        "approx_kl",
        "candidate_kl",
        "full_rollout_kl",
        "pre_update_approx_kl",
        "clip_fraction",
        "grad_norm",
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_update_to_weight_norm",
        "actor_steps",
        "critic_steps",
        "actor_rolled_back",
        "actor_rollback_full_rollout",
        "actor_rollback_count",
        "consecutive_actor_rollbacks",
        "actor_lr_safety_multiplier",
        "rollback_stop_requested",
        "actor_kl_warmup_active",
        "rollback_free_accepted_updates",
        "learning_rate",
        "filter_eta",
        "filter_ewma_max_abs_adv",
        "early_stopped",
        "epochs_completed",
        "logp_delta_max",
        "logp_delta_mean",
        "action_pretanh_max_abs",
    }
)
_SIM_METRIC_KEYS = frozenset(
    {
        "valid_transitions",
        "valid_frac",
        "active_end",
        "active_frac_end",
        "reconstruction_digest_mismatches",
        "collision_frac",
        "oob_frac",
        "reset_frac",
        "done_frac",
        "timeout_frac",
        "progress_mean",
        "spawn_requested",
        "spawn_realized",
        "spawn_rejects",
    }
)


class WandbAuthError(RuntimeError):
    """Raised when online W&B mode is requested without local credentials."""


def wandb_is_active(cfg: WandbConfig) -> bool:
    return bool(cfg.enabled) and str(cfg.mode) != "disabled"


def has_wandb_auth() -> bool:
    """True when a local API key appears available (no network calls)."""
    if os.environ.get("WANDB_API_KEY", "").strip():
        return True
    netrc_path = Path.home() / ".netrc"
    if netrc_path.is_file():
        try:
            text = netrc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if "api.wandb.ai" in text or "wandb.ai" in text:
            return True
    settings = Path.home() / ".config" / "wandb" / "settings"
    if settings.is_file():
        try:
            text = settings.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if "api_key" in text:
            return True
    return False


def assert_wandb_auth_for_online(mode: str) -> None:
    if str(mode) != "online":
        return
    if has_wandb_auth():
        return
    raise WandbAuthError(
        "wandb.mode=online but no local W&B credentials were found. "
        "Set WANDB_API_KEY in the environment, run `wandb login`, "
        "or use --wandb-mode offline / --no-wandb. "
        "API keys are never read from or written into experiment configs."
    )


def collect_provenance(cfg: ExperimentConfig) -> dict[str, Any]:
    """Git SHA, track manifest summary, and hardware facts for run config."""
    git_sha = None
    git_dirty = None
    git_diff_hash = None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_sha = sha or None
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        git_dirty = bool(dirty_out.strip())
        diff = subprocess.check_output(
            ["git", "diff", "--binary"],
            stderr=subprocess.DEVNULL,
        )
        git_diff_hash = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass

    manifest_summary: dict[str, Any] = {
        "path": cfg.tracks.manifest_path,
        "num_tracks_config": int(cfg.tracks.num_tracks),
    }
    if cfg.tracks.manifest_path:
        man_path = Path(cfg.tracks.manifest_path).expanduser()
        if man_path.is_file():
            try:
                payload = json.loads(man_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, Mapping):
                tracks = payload.get("tracks", payload.get("entries"))
                if isinstance(tracks, list):
                    manifest_summary["num_tracks_manifest"] = len(tracks)
                    names = [
                        t.get("name")
                        for t in tracks
                        if isinstance(t, Mapping) and t.get("name")
                    ]
                    if names:
                        manifest_summary["track_names"] = names
                for key in ("pin_revision", "revision", "checksum", "sha256"):
                    if key in payload:
                        manifest_summary[key] = payload[key]

    gpu_names: list[str] = []
    gpu_details: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                gpu_names.append(torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                gpu_details.append(
                    {
                        "name": props.name,
                        "compute_capability": [
                            int(props.major),
                            int(props.minor),
                        ],
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
            except Exception:
                gpu_names.append(f"cuda:{i}")
    try:
        import warp

        warp_version = getattr(warp, "__version__", None)
    except ImportError:
        warp_version = None
    try:
        ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        ram_bytes = None
    config_payload = json.dumps(
        config_to_dict(cfg), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_diff_hash": git_diff_hash,
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "track_manifest": manifest_summary,
        "hardware": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "warp": warp_version,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "gpu_names": gpu_names,
            "gpu_details": gpu_details,
            "cpu_count": os.cpu_count(),
            "ram_bytes": ram_bytes,
        },
    }


def flatten_train_metrics(
    *,
    progress_metrics: Mapping[str, float],
    profile: Mapping[str, float],
    update_index: int,
    transitions: int,
    reward_term_means: Mapping[str, float] | None = None,
    track_stats: Mapping[str, float] | None = None,
    density_stats: Mapping[str, float] | None = None,
    sim_stats: Mapping[str, float] | None = None,
    device: str | None = None,
) -> dict[str, float]:
    """Flatten comprehensive trainer/PPO/sim/reward/track/density/throughput/GPU scalars."""
    out: dict[str, float] = {
        "train/update_index": float(update_index),
        "train/transitions": float(transitions),
    }
    merged = dict(progress_metrics)
    if sim_stats:
        merged.update(sim_stats)
    for key, value in merged.items():
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if key in _PPO_METRIC_KEYS:
            out[f"ppo/{key}"] = fval
        elif key.startswith(
            (
                "advantage_",
                "returns_",
                "values_",
                "residual_",
                "latent_entropy_",
                "log_std_",
                "action_saturation_",
                "explained_variance",
            )
        ):
            out[f"ppo/{key}"] = fval
        elif key in _SIM_METRIC_KEYS:
            out[f"sim/{key}"] = fval
        elif key == "transitions_per_s":
            out["throughput/transitions_per_s"] = fval
        else:
            out[f"train/{key}"] = fval

    for key in (
        "collect_s",
        "reconstruct_s",
        "ppo_s",
        "update_s",
        "transitions_per_s",
        "eval_s",
    ):
        if key in profile:
            try:
                out[f"throughput/{key}"] = float(profile[key])
            except (TypeError, ValueError):
                pass

    if reward_term_means:
        for key, value in reward_term_means.items():
            try:
                out[f"reward/{key}"] = float(value)
            except (TypeError, ValueError):
                pass
    if track_stats:
        for key, value in track_stats.items():
            try:
                out[f"track/{key}"] = float(value)
            except (TypeError, ValueError):
                pass
    if density_stats:
        for key, value in density_stats.items():
            try:
                out[f"density/{key}"] = float(value)
            except (TypeError, ValueError):
                pass

    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            out["gpu/memory_allocated_bytes"] = float(torch.cuda.memory_allocated())
            out["gpu/memory_reserved_bytes"] = float(torch.cuda.memory_reserved())
            out["gpu/max_memory_allocated_bytes"] = float(
                torch.cuda.max_memory_allocated()
            )
            out["gpu/max_memory_reserved_bytes"] = float(
                torch.cuda.max_memory_reserved()
            )
        except Exception:
            pass
        util_fn = getattr(torch.cuda, "utilization", None)
        if callable(util_fn):
            try:
                out["gpu/utilization"] = float(util_fn())
            except Exception:
                pass
    return out


def summarize_rollout_aux(
    *,
    rewards: torch.Tensor | None,
    valid: torch.Tensor | None,
    track_id: torch.Tensor | None,
    active_end: torch.Tensor | None,
    max_agents_per_world: int,
    reward_terms: Mapping[str, torch.Tensor] | None = None,
    done: torch.Tensor | None = None,
    timeout: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
    contact: torch.Tensor | None = None,
    wall_contact: torch.Tensor | None = None,
    sim_arrays: Any | None = None,
    spawn_stats: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """CPU summaries for reward/track/density/sim (called only at log cadence).

    Prefer transition-time ``reward_terms`` / ``contact`` / ``wall_contact`` with
    shape ``[T, S]`` (rollout means/counts). End-of-rollout ``sim_arrays`` is only
    a fallback for progress and legacy callers.
    """
    reward_means: dict[str, float] = {}
    track_stats: dict[str, float] = {}
    density_stats: dict[str, float] = {}
    sim_stats: dict[str, float] = {}

    if rewards is not None and valid is not None and bool(valid.any().item()):
        mask = valid.to(dtype=torch.bool)
        vals = rewards[mask].float()
        reward_means["total_mean"] = float(vals.mean().item())
        reward_means["total_std"] = float(vals.std(unbiased=False).item())
        reward_means["total_min"] = float(vals.min().item())
        reward_means["total_max"] = float(vals.max().item())
    if reward_terms:
        for name, tensor in reward_terms.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if (
                valid is not None
                and tensor.shape == valid.shape
                and bool(valid.any().item())
            ):
                # Rollout [T, S] means over valid transitions (preferred).
                reward_means[f"{name}_mean"] = float(
                    tensor[valid.to(dtype=torch.bool)].float().mean().item()
                )
            elif (
                valid is not None
                and tensor.ndim == 1
                and valid.ndim == 2
                and tensor.shape[0] == valid.shape[1]
                and bool(valid.any().item())
            ):
                # Legacy last-step [S] path — prefer stacking terms in the trainer.
                slot_mask = valid.to(dtype=torch.bool).any(dim=0)
                reward_means[f"{name}_mean"] = float(
                    tensor[slot_mask].float().mean().item()
                )
            elif tensor.numel() > 0:
                reward_means[f"{name}_mean"] = float(tensor.float().mean().item())

    if track_id is not None:
        ids = track_id.detach().flatten().to(dtype=torch.int64)
        if valid is not None and valid.ndim == 2 and ids.numel() == valid.shape[1]:
            slot_mask = valid.to(dtype=torch.bool).any(dim=0)
            if bool(slot_mask.any().item()):
                ids = ids[slot_mask]
        if ids.numel() > 0:
            uniq = torch.unique(ids)
            track_stats["unique_count"] = float(uniq.numel())
            track_stats["id_mean"] = float(ids.float().mean().item())
            track_stats["id_mode"] = float(int(torch.mode(ids).values.item()))

    if active_end is not None and max_agents_per_world > 0:
        active = active_end.to(dtype=torch.float32).detach().flatten()
        n = int(active.numel())
        if n > 0:
            worlds = max(1, n // int(max_agents_per_world))
            per_world = active.view(worlds, int(max_agents_per_world)).sum(dim=1)
            dens = per_world / float(max_agents_per_world)
            density_stats["active_frac"] = float(active.mean().item())
            density_stats["agents_per_world_mean"] = float(per_world.mean().item())
            density_stats["agents_per_world_max"] = float(per_world.max().item())
            density_stats["solo_world_frac"] = float(
                (per_world <= 1.0).float().mean().item()
            )
            density_stats["sparse_world_frac"] = float(
                (dens <= 0.34).float().mean().item()
            )
            density_stats["medium_world_frac"] = float(
                ((dens > 0.34) & (dens <= 0.67)).float().mean().item()
            )
            density_stats["dense_world_frac"] = float(
                (dens > 0.67).float().mean().item()
            )

    if done is not None and valid is not None and bool(valid.any().item()):
        mask = valid.to(dtype=torch.bool)
        sim_stats["done_frac"] = float(done.to(dtype=torch.bool)[mask].float().mean().item())
    if timeout is not None and valid is not None and bool(valid.any().item()):
        mask = valid.to(dtype=torch.bool)
        sim_stats["timeout_frac"] = float(
            timeout.to(dtype=torch.bool)[mask].float().mean().item()
        )
    if reset_mask is not None and valid is not None and bool(valid.any().item()):
        mask = valid.to(dtype=torch.bool)
        sim_stats["reset_frac"] = float(
            reset_mask.to(dtype=torch.bool)[mask].float().mean().item()
        )

    if (
        contact is not None
        and valid is not None
        and contact.shape == valid.shape
        and bool(valid.any().item())
    ):
        mask = valid.to(dtype=torch.bool)
        sim_stats["collision_frac"] = float(
            contact.to(dtype=torch.bool)[mask].float().mean().item()
        )
        sim_stats["collision_count"] = float(
            contact.to(dtype=torch.bool)[mask].float().sum().item()
        )
    if (
        wall_contact is not None
        and valid is not None
        and wall_contact.shape == valid.shape
        and bool(valid.any().item())
    ):
        mask = valid.to(dtype=torch.bool)
        sim_stats["oob_frac"] = float(
            wall_contact.to(dtype=torch.bool)[mask].float().mean().item()
        )
        sim_stats["oob_count"] = float(
            wall_contact.to(dtype=torch.bool)[mask].float().sum().item()
        )

    if sim_arrays is not None:
        active = (sim_arrays.active > 0).to(dtype=torch.bool)
        denom = float(max(1, int(active.sum().item())))
        # Fallback only when transition-time event tensors were not provided.
        if "collision_frac" not in sim_stats and hasattr(sim_arrays, "contact"):
            sim_stats["collision_frac"] = float(
                ((sim_arrays.contact > 0) & active).float().sum().item() / denom
            )
        if "oob_frac" not in sim_stats and hasattr(sim_arrays, "wall_contact"):
            sim_stats["oob_frac"] = float(
                ((sim_arrays.wall_contact > 0) & active).float().sum().item() / denom
            )
        if hasattr(sim_arrays, "progress_s") and bool(active.any().item()):
            sim_stats["progress_mean"] = float(
                sim_arrays.progress_s.to(dtype=torch.float32)[active].mean().item()
            )
    if spawn_stats:
        for key in ("requested", "realized", "rejects"):
            if key in spawn_stats:
                try:
                    sim_stats[f"spawn_{key}"] = float(spawn_stats[key])
                except (TypeError, ValueError):
                    pass
    return reward_means, track_stats, density_stats, sim_stats


def summarize_eval_lap_times(reports: Sequence[Any]) -> dict[str, float]:
    """Aggregate lap timing across an eval sweep's per-suite/seed reports.

    ``lap_time_s`` is ``None`` for a report where nobody finished a lap, so the
    mean/best keys cover only the reports that timed one. ``lap_completers_frac``
    is always emitted (0.0 when no lap finished anywhere) so the series
    distinguishes "no lap completed" from "evaluation did not run".
    """
    per_suite: dict[str, list[float]] = {}
    lap_times: list[float] = []
    completers = 0.0
    participants = 0.0
    for report in reports:
        metrics = getattr(report, "metrics", None)
        lap_time = getattr(metrics, "lap_time_s", None)
        if lap_time is not None:
            value = float(lap_time)
            lap_times.append(value)
            per_suite.setdefault(str(getattr(report, "suite", "eval")), []).append(value)
        extras = getattr(report, "extras", None) or {}
        completers += float(extras.get("num_lap_completers", 0.0))
        participants += float(extras.get("num_participants", 0.0))
    if not reports:
        return {}
    summary: dict[str, float] = {
        "eval/lap_completers_frac": (
            completers / participants if participants > 0 else 0.0
        )
    }
    if lap_times:
        summary["eval/lap_time_s_mean"] = sum(lap_times) / len(lap_times)
        summary["eval/lap_time_s_best"] = min(lap_times)
    for suite, values in per_suite.items():
        summary[f"eval/{suite}/lap_time_s_mean"] = sum(values) / len(values)
    return summary


def read_run_meta(run_dir: str | Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    path = Path(run_dir) / RUN_META_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_run_meta(run_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / RUN_META_FILENAME
    out.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def resolve_wandb_run_id(
    cfg: WandbConfig,
    *,
    run_dir: str | Path | None = None,
    checkpoint_wandb_run_id: str | None = None,
) -> str | None:
    """Prefer explicit config id, then checkpoint, then local run_meta."""
    if cfg.run_id:
        return str(cfg.run_id)
    if checkpoint_wandb_run_id:
        return str(checkpoint_wandb_run_id)
    meta = read_run_meta(run_dir)
    wb = meta.get("wandb")
    if isinstance(wb, Mapping) and wb.get("run_id"):
        return str(wb["run_id"])
    return None


class WandbSession:
    """Cadenced W&B logger. No-ops when disabled; never blocks the train hot path."""

    def __init__(
        self,
        cfg: WandbConfig,
        *,
        experiment: ExperimentConfig,
        run_dir: str | Path | None = None,
        checkpoint_wandb_run_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.experiment = experiment
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._checkpoint_wandb_run_id = checkpoint_wandb_run_id
        self._run: Any | None = None
        self._run_id: str | None = None
        self._started = False
        self._logging_disabled = False
        self._last_step: int | None = None
        self._eval_axis_defined = False
        self.provenance: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self._run is not None and not self._logging_disabled

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def last_step(self) -> int | None:
        """Highest explicit global step successfully passed to ``wandb.Run.log``."""
        return self._last_step

    def start(self) -> str | None:
        if self._started:
            return self._run_id
        self._started = True
        self.provenance = collect_provenance(self.experiment)
        if not wandb_is_active(self.cfg):
            if self.run_dir is not None:
                self._write_meta(enabled=False)
            return None

        assert_wandb_auth_for_online(self.cfg.mode)
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is enabled but the wandb package is not installed. "
                "Install requirements-gpu.txt / `pip install wandb`, or disable "
                "with --no-wandb."
            ) from exc

        run_id = resolve_wandb_run_id(
            self.cfg,
            run_dir=self.run_dir,
            checkpoint_wandb_run_id=self._checkpoint_wandb_run_id,
        )
        init_kwargs: dict[str, Any] = {
            "project": self.cfg.project,
            "config": {
                "experiment": config_to_dict(self.experiment),
                "provenance": self.provenance,
            },
            "mode": self.cfg.mode,
            "resume": self.cfg.resume,
            "reinit": True,
        }
        if self.cfg.entity:
            init_kwargs["entity"] = self.cfg.entity
        if self.cfg.group:
            init_kwargs["group"] = self.cfg.group
        if self.cfg.name:
            init_kwargs["name"] = self.cfg.name
        if self.cfg.tags:
            init_kwargs["tags"] = list(self.cfg.tags)
        if self.cfg.notes:
            init_kwargs["notes"] = self.cfg.notes
        if run_id:
            init_kwargs["id"] = run_id
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            init_kwargs["dir"] = str(self.run_dir)

        self._run = wandb.init(**init_kwargs)
        self._run_id = getattr(self._run, "id", None) or run_id
        self.define_eval_axis()
        if self.run_dir is not None:
            self._write_meta(enabled=True)
        # Async startup marker so the UI is not system-metrics-only during the
        # first long collect/PPO update. Never flush/sync on the train path.
        self._log_startup_markers()
        return self._run_id

    def define_eval_axis(self) -> None:
        """Bind ``eval/*`` (scalars + media) to custom axis ``eval/source_step``.

        Official W&B custom-axis pattern: cadence lives in the payload under
        ``eval/source_step`` and must never be passed as the global ``step=``
        argument (async CPU eval often completes after training has advanced).
        """
        if self._eval_axis_defined or not self.active:
            return
        try:
            self._run.define_metric(EVAL_SOURCE_STEP_METRIC)
            self._run.define_metric(
                EVAL_METRIC_GLOB, step_metric=EVAL_SOURCE_STEP_METRIC
            )
        except Exception as exc:
            warnings.warn(
                f"W&B define_metric for eval axis failed; "
                f"continuing without custom eval x-axis: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._eval_axis_defined = True

    def _log_startup_markers(self) -> None:
        if not self.active:
            return
        cfg = self.experiment
        slots = int(cfg.worlds.num_worlds) * int(cfg.worlds.max_agents_per_world)
        rollout = int(cfg.ppo.rollout_length)
        payload = {
            "startup/ready": 1.0,
            "startup/num_worlds": float(cfg.worlds.num_worlds),
            "startup/max_agents_per_world": float(cfg.worlds.max_agents_per_world),
            "startup/num_slots": float(slots),
            "startup/rollout_length": float(rollout),
            "startup/transitions_per_update": float(slots * rollout),
            "startup/report_interval_updates": float(
                cfg.profiling.report_interval_updates
            ),
            "startup/total_updates": float(cfg.ppo.total_updates),
            "startup/wandb_online": float(1.0 if self.cfg.mode == "online" else 0.0),
        }
        self.log_metrics(payload, step=0)

    def _write_meta(self, *, enabled: bool) -> None:
        assert self.run_dir is not None
        write_run_meta(
            self.run_dir,
            {
                "wandb": {
                    "enabled": bool(enabled),
                    "mode": self.cfg.mode,
                    "entity": self.cfg.entity,
                    "project": self.cfg.project,
                    "group": self.cfg.group,
                    "name": self.cfg.name,
                    "tags": list(self.cfg.tags),
                    "notes": self.cfg.notes,
                    "run_id": self._run_id,
                    "resume": self.cfg.resume,
                    "eval_interval_updates": int(self.cfg.eval_interval_updates),
                    "log_artifacts": bool(self.cfg.log_artifacts),
                },
                "provenance": self.provenance,
            },
        )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        if not self.active:
            return
        payload = dict(metrics)
        try:
            if step is None:
                self._run.log(payload)
            else:
                step_i = int(step)
                self._run.log(payload, step=step_i)
                self._last_step = (
                    step_i
                    if self._last_step is None
                    else max(int(self._last_step), step_i)
                )
        except Exception as exc:
            self._logging_disabled = True
            warnings.warn(
                f"W&B metric logging failed; continuing with local logs only: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def update_runtime_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.provenance["runtime"] = dict(metadata)
        if self.active:
            try:
                self._run.config.update(
                    {"runtime": dict(metadata)},
                    allow_val_change=True,
                )
            except Exception as exc:
                warnings.warn(
                    f"W&B runtime metadata update failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self.run_dir is not None:
            self._write_meta(enabled=self.active)

    def log_artifact_refs(
        self,
        *,
        step: int,
        checkpoint_path: str | Path | None = None,
        actor_path: str | Path | None = None,
    ) -> None:
        """Log local checkpoint/actor path references (not binary uploads by default)."""
        if not self.active or not self.cfg.log_artifacts:
            return
        payload: dict[str, Any] = {}
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            payload["artifact/checkpoint_path"] = str(path)
            payload["artifact/checkpoint_exists"] = float(path.is_file())
        if actor_path is not None:
            path = Path(actor_path)
            payload["artifact/actor_path"] = str(path)
            payload["artifact/actor_exists"] = float(path.is_file())
        if payload:
            self.log_metrics(payload, step=step)

    def log_evaluation(
        self,
        reports: Sequence[Any],
        *,
        source_step: int,
        media_paths: Sequence[str | Path] | None = None,
        global_step: int | None = None,
        extra_metrics: Mapping[str, Any] | None = None,
        # Legacy alias: treated as global history step only (never as eval axis).
        step: int | None = None,
    ) -> None:
        """Log eval scalars/media against custom axis ``eval/source_step``.

        ``source_step`` is the cadence/checkpoint index used as the eval x-axis.
        ``global_step`` (or legacy ``step``) is the monotonic W&B history step and
        must never be the stale cadence when async eval finishes late.
        """
        if not self.active:
            return
        self.define_eval_axis()
        source = int(source_step)
        history_step = global_step if global_step is not None else step
        payload: dict[str, Any] = {EVAL_SOURCE_STEP_METRIC: float(source)}
        for report in reports:
            suite = getattr(report, "suite", "eval")
            seed = getattr(report, "seed", 0)
            metrics = getattr(report, "metrics", None)
            prefix = f"eval/{suite}/seed{seed}"
            if metrics is not None:
                data = (
                    metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
                )
                for key, value in data.items():
                    if value is None:
                        continue
                    try:
                        payload[f"{prefix}/{key}"] = float(value)
                    except (TypeError, ValueError):
                        continue
            extras = getattr(report, "extras", None) or {}
            for key, value in extras.items():
                try:
                    payload[f"{prefix}/extra/{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        payload.update(summarize_eval_lap_times(reports))
        if extra_metrics:
            for key, value in extra_metrics.items():
                if key == EVAL_SOURCE_STEP_METRIC:
                    continue
                payload[str(key)] = value
        if media_paths:
            try:
                import wandb
            except ImportError:
                wandb = None
            if wandb is not None:
                images = []
                videos = []
                for raw in media_paths:
                    path = Path(raw)
                    if not path.is_file():
                        continue
                    suffix = path.suffix.lower()
                    caption = f"{path.name} ({EVAL_SOURCE_STEP_METRIC}={source}"
                    if history_step is not None:
                        caption += f", global_step={int(history_step)}"
                    caption += ")"
                    try:
                        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                            images.append(wandb.Image(str(path), caption=caption))
                        elif suffix == ".mp4":
                            videos.append(
                                wandb.Video(str(path), caption=caption, format="mp4")
                            )
                        elif suffix == ".webm":
                            videos.append(
                                wandb.Video(str(path), caption=caption, format="webm")
                            )
                    except Exception as exc:
                        warnings.warn(
                            f"W&B media attach skipped for {path}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                if images:
                    payload["eval/media/images"] = images
                if videos:
                    payload["eval/media/videos"] = videos
        if len(payload) > 1 or media_paths:
            self.log_metrics(payload, step=history_step)

    def finish(self) -> None:
        run = self._run
        self._run = None
        if run is None:
            return
        try:
            run.finish()
        except Exception as exc:
            warnings.warn(
                f"W&B finish/flush failed; local logs remain authoritative: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def build_wandb_session(
    cfg: ExperimentConfig,
    *,
    run_dir: str | Path | None = None,
    checkpoint_wandb_run_id: str | None = None,
) -> WandbSession:
    return WandbSession(
        cfg.wandb,
        experiment=cfg,
        run_dir=run_dir,
        checkpoint_wandb_run_id=checkpoint_wandb_run_id,
    )

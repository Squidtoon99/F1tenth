"""Single-machine self-play trainer: collect → reconstruct → PPO update."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from gigaflow_f1tenth.artifacts import export_actor_artifact
from gigaflow_f1tenth.async_cpu_eval import (
    ACTOR_SNAPSHOT,
    AsyncCpuEvalManager,
    eval_output_dir,
    load_reports_from_json,
)
from gigaflow_f1tenth.buffers import (
    STATE_INDEX,
    allocate_rollout_buffer,
    observation_digest,
)
from gigaflow_f1tenth.config import ExperimentConfig, config_to_dict
from gigaflow_f1tenth.critic import (
    build_critic,
    critic_values_over_time,
    pack_critic_features,
)
from gigaflow_f1tenth.evaluation import (
    BehaviorGateState,
    resolve_eval_device,
)
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.model import ACT_LIMIT, build_actor
from gigaflow_f1tenth.ppo import (
    PreparedPPOInputs,
    build_ppo,
    evaluate_actions_sequence,
    export_resume_state,
    load_resume_state,
)
from gigaflow_f1tenth.rewards import (
    deployment_style,
    normalize_condition_vector,
    styles_to_condition_batch,
)
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import (
    MANIFEST_FILENAME,
    PackedTrackAtlasView,
    load_atlas,
)
from gigaflow_f1tenth.wandb_log import (
    WandbSession,
    build_wandb_session,
    flatten_train_metrics,
    summarize_rollout_aux,
)

CHECKPOINT_VERSION = 8
RUN_MANIFEST_FILENAME = "run_manifest.json"


class ReconstructionParityError(RuntimeError):
    """Raised when replayed compact state does not reproduce the collected obs."""


def track_manifest_hash(cfg: ExperimentConfig) -> str | None:
    """SHA-256 of the configured track manifest, or None when unset."""
    raw = cfg.tracks.manifest_path
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"track manifest not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "outputs" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".yaml", ".toml", ".md"}:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    result: dict[str, Any] = {"revision": None, "diff_hash": None}
    try:
        result["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "--", "gigaflow", "docs/adr"],
            stderr=subprocess.DEVNULL,
        )
        result["diff_hash"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _initial_run_manifest(
    cfg: ExperimentConfig,
    *,
    num_updates: int,
    device: str,
    actor_checkpoint_interval: int,
    full_checkpoint_interval: int,
    resume_from: str | Path | None,
) -> dict[str, Any]:
    config_json = json.dumps(config_to_dict(cfg), sort_keys=True).encode("utf-8")
    manifest_path = (
        None
        if cfg.tracks.manifest_path is None
        else Path(cfg.tracks.manifest_path).expanduser()
    )
    atlas_path = None if manifest_path is None else manifest_path.parent / "atlas.npz"
    return {
        "started_at": _utc_now(),
        "finished_at": None,
        "shutdown_reason": None,
        "resolved_config": config_to_dict(cfg),
        "config_sha256": hashlib.sha256(config_json).hexdigest(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "actual_num_updates": int(num_updates),
        "actual_device": str(device),
        "actor_checkpoint_interval_updates": int(actor_checkpoint_interval),
        "full_checkpoint_interval_updates": int(full_checkpoint_interval),
        "resume_from": None if resume_from is None else str(resume_from),
        "source_tree_sha256": _source_tree_hash(),
        "git": _git_identity(),
        "atlas": {
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "manifest_sha256": (
                None if manifest_path is None else _sha256_file(manifest_path)
            ),
            "atlas_sha256": None if atlas_path is None else _sha256_file(atlas_path),
        },
        "hardware": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }


def _write_run_manifest(run_dir: str | Path, payload: dict[str, Any]) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass
class TrainProgress:
    update_index: int
    transitions: int
    metrics: dict[str, float]


@runtime_checkable
class Trainer(Protocol):
    def setup(self) -> None:
        ...

    def collect_rollout(self) -> Any:
        ...

    def train_update(self) -> TrainProgress:
        ...

    def save_checkpoint(
        self, path: str, *, shutdown_reason: str | None = None
    ) -> None:
        ...

    def load_checkpoint(self, path: str) -> None:
        ...


def _resolve_device(cfg: ExperimentConfig, device: str | None) -> str:
    if device is not None:
        wanted = device
    else:
        wanted = str(cfg.worlds.device)
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={wanted!r} requested but torch.cuda.is_available() is False; "
            "refusing silent CPU fallback for the Warp runtime path"
        )
    return wanted


def _track_preview_atlas(
    atlas: PackedTrackAtlasView, device: str
) -> PackedTrackAtlasView:
    """Device-resident view carrying only what ``sample_track_lookahead`` reads.

    The full atlas also carries the nearest-segment LUT and corridor EDT grids
    (hundreds of MB for the full pinned set), which the simulator already
    copies to device itself as Warp-owned arrays outside the torch allocator.
    Moving the *whole* atlas onto the torch device here would silently double
    that memory inside torch's own accounting instead. The preview sampler
    only touches ``offsets``, ``centerline_xy``, ``tangents_xy``,
    ``widths_rl``, ``cum_length``, and ``lengths``, so only those move.
    """
    if device == "cpu":
        return atlas
    torch_device = torch.device(device)

    def _t(array: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), device=torch_device)

    return dataclass_replace(
        atlas,
        offsets=_t(atlas.offsets).to(torch.int32),
        centerline_xy=_t(atlas.centerline_xy).to(torch.float32),
        tangents_xy=_t(atlas.tangents_xy).to(torch.float32),
        widths_rl=_t(atlas.widths_rl).to(torch.float32),
        cum_length=_t(atlas.cum_length).to(torch.float32),
        lengths=_t(atlas.lengths).to(torch.float32),
    )


def _actor_variant(cfg: ExperimentConfig) -> tuple[str, int]:
    # Ablation section overrides agents.* when set differently from baseline.
    variant = cfg.ablations.actor_variant or cfg.agents.actor_variant
    stack = int(cfg.ablations.frame_stack or cfg.agents.frame_stack)
    if variant == "frame_stack" and stack == 1:
        stack = max(cfg.agents.frame_stack, 4)
    return str(variant), int(stack)


class SelfPlayTrainer:
    """End-to-end collection → compact rollout → reconstruct → PPO loop."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        atlas: PackedTrackAtlasView | None = None,
        device: str | None = None,
        run_dir: str | Path | None = None,
        wandb_session: WandbSession | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg, device)
        self.atlas = atlas
        self._preview_atlas: PackedTrackAtlasView | None = None
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.wandb = wandb_session
        self.layout = world_slot_layout(cfg)
        self.sim = None
        self.actor = None
        self.critic = None
        self.ppo = None
        self.buffer = None
        self._setup_done = False
        self.transitions = 0
        self.profile: dict[str, float] = {}
        self._zero_condition = False
        self._last_reward_terms: dict[str, Any] | None = None
        self._rollout_reward_terms: dict[str, list[Any]] | None = None
        self._rollout_contact: list[Any] | None = None
        self._rollout_wall_contact: list[Any] | None = None
        self._last_batch = None
        # Ablation toggles for throughput A/B (default: optimized path).
        self.force_rebuild_sensors = False
        self.disable_compact_actor = False

    def setup(self) -> None:
        cfg = self.cfg
        # Compact active batches change CNN batch size; keep CuDNN algorithms
        # stable so collection logp matches frozen evaluate rescoring.
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        atlas = self.atlas
        if atlas is None:
            if cfg.tracks.manifest_path:
                atlas = load_atlas(
                    str(Path(cfg.tracks.manifest_path).expanduser())
                ).view()
            else:
                atlas = make_synthetic_oval_atlas(
                    max_agents=cfg.worlds.max_agents_per_world
                )
            self.atlas = atlas
        self._preview_atlas = _track_preview_atlas(atlas, self.device)

        self.sim = build_simulator(cfg, atlas, self.device)
        variant, frame_stack = _actor_variant(cfg)
        self.actor = build_actor(cfg, variant=variant, frame_stack=frame_stack)
        self.critic = build_critic(cfg)
        self.ppo = build_ppo(
            cfg,
            self.actor,
            self.critic,
            orthogonal_init=True,
            device=self.device,
        )
        self.buffer = allocate_rollout_buffer(
            cfg, self.device, state_dim=int(self.sim.state_dim)
        )
        self._zero_condition = not bool(
            cfg.reward_conditioning.expose_condition_to_actor
            and cfg.ablations.condition_to_actor
        )
        # Align private styles + GRU carry with the live world.
        self.sim.resample_styles_for_mask(
            torch.ones(self.layout.num_slots, dtype=torch.bool), cfg.seed + 3
        )
        self.ppo.ensure_carry(self.layout.num_slots)
        self._setup_done = True
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "config.json").write_text(
                json.dumps(config_to_dict(cfg), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if self.wandb is not None:
            self.wandb.start()

    def _condition_tensor(self) -> torch.Tensor:
        assert self.sim is not None
        if hasattr(self.sim, "condition_tensor"):
            cond = self.sim.condition_tensor().to(device=self.device, dtype=torch.float32)
        else:
            raw = styles_to_condition_batch(self.sim.styles, normalize=True)
            cond = torch.as_tensor(raw, device=self.device, dtype=torch.float32)
        if self._zero_condition:
            return torch.zeros_like(cond)
        return cond

    def _valid_mask(self) -> torch.Tensor:
        assert self.sim is not None
        t = self.sim.buffers.torch_arrays
        return (t.active > 0) & (t.trainable > 0)

    def _actor_step_compact(
        self,
        sensor_obs: torch.Tensor,
        cond: torch.Tensor,
        hidden: torch.Tensor,
        pending_reset: torch.Tensor,
        active_mask: torch.Tensor,
    ):
        """Run CNN-GRU only on active slots; scatter back into fixed buffers."""
        assert self.actor is not None
        n = int(sensor_obs.shape[0])
        device = self.device
        actions = torch.zeros(n, 2, device=device, dtype=torch.float32)
        logp = torch.zeros(n, device=device, dtype=torch.float32)
        pre_tanh = torch.zeros(n, 2, device=device, dtype=torch.float32)
        next_hidden = hidden.to(device=device, dtype=torch.float32).clone()
        idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return actions, logp, next_hidden, pre_tanh
        # Actor collect stays FP32: BF16 step-vs-sequence drift breaks logp parity.
        out = self.actor.forward(
            sensor_obs.index_select(0, idx),
            cond.index_select(0, idx),
            next_hidden.index_select(0, idx),
            reset_mask=pending_reset.index_select(0, idx),
            deterministic=False,
        )
        if out.pre_tanh is None:
            raise RuntimeError("actor must return pre_tanh for PPO rescoring")
        actions.index_copy_(0, idx, out.actions.detach())
        logp.index_copy_(0, idx, out.log_prob.detach())
        pre_tanh.index_copy_(0, idx, out.pre_tanh.detach())
        next_hidden.index_copy_(0, idx, out.hidden.detach())
        return actions, logp, next_hidden, pre_tanh

    def collect_rollout(self):
        if not self._setup_done:
            self.setup()
        assert self.sim is not None and self.ppo is not None and self.buffer is not None
        assert self.actor is not None
        cfg = self.cfg
        t_len = cfg.ppo.rollout_length
        n = self.layout.num_slots
        self.buffer.reset()
        gru_start = self.ppo.begin_rollout(n)
        self.buffer.set_rollout_start_hidden(gru_start)
        self._rollout_reward_terms = None
        self._rollout_contact = []
        self._rollout_wall_contact = []

        hidden = gru_start.clone()
        t0 = time.perf_counter()
        for step in range(t_len):
            arrays = self.sim.buffers.torch_arrays
            # Action-time obs = sensors for current SoA (built by reset/last step).
            # Do not rebuild here: sim.step() owns the single post-transition build.
            if self.force_rebuild_sensors:
                live = self.sim.rebuild_sensors()
            else:
                live = self.sim.action_observation()
            sensor_obs = live.detach().to(
                device=self.device, dtype=torch.float32
            ).clone()
            pending_reset = self.ppo.carry_reset
            if pending_reset is None:
                pending_reset = torch.zeros(n, dtype=torch.bool, device=self.device)
            else:
                pending_reset = pending_reset.to(device=self.device)
            cond = self._condition_tensor()
            # Static opponents are active (visible to LiDAR/contact) but never
            # act: exclude them from the policy forward and force full brake
            # below, mirroring the viewer's stationary-obstacle handling.
            learner_mask = (arrays.active > 0) & (arrays.trainable > 0)
            static_mask = (arrays.active > 0) & (arrays.trainable == 0)
            with torch.no_grad():
                if self.disable_compact_actor:
                    out = self.actor.forward(
                        sensor_obs,
                        cond,
                        hidden.to(self.device),
                        reset_mask=pending_reset,
                        deterministic=False,
                    )
                    if out.pre_tanh is None:
                        raise RuntimeError(
                            "actor must return pre_tanh for PPO rescoring"
                        )
                    actions = out.actions.detach().clone()
                    logp = out.log_prob.detach().clone()
                    next_hidden = out.hidden.detach().clone()
                    pre_tanh = out.pre_tanh.detach().clone()
                else:
                    actions, logp, next_hidden, pre_tanh = self._actor_step_compact(
                        sensor_obs,
                        cond,
                        hidden.to(self.device),
                        pending_reset,
                        learner_mask,
                    )
                if bool(static_mask.any()):
                    brake = torch.tensor(
                        [-1.0, 0.0], device=self.device, dtype=torch.float32
                    )
                    actions[static_mask] = brake
                    # pre_tanh must reproduce the forced action under the same
                    # squash the collect/evaluate parity gate rechecks, or an
                    # untrained action with no matching sample fails that gate.
                    clamped = (brake / ACT_LIMIT).clamp(-0.999999, 0.999999)
                    pre_tanh[static_mask] = torch.atanh(clamped)
            cond_store = cond.detach().clone()

            # Valid = agents that were active/trainable when the action was taken.
            valid = self._valid_mask().to(device=self.device)
            compact_before = self.sim.pack_state()
            # Sensor-noise RNG stream at action time: async respawn advances the
            # seed and resets episode_id/episode_step inside step().
            noise_seed = arrays.sensor_noise_seed.to(dtype=torch.int64).clone()
            episode_id = arrays.episode_id.clone()
            episode_step = arrays.episode_step.clone()
            result = self.sim.step(actions)
            rewards = result["rewards"].to(device=self.device, dtype=torch.float32)
            done = result["done"].to(device=self.device, dtype=torch.bool)
            timeout = result["timeout"].to(device=self.device, dtype=torch.bool)
            reset_mask = result["reset_mask"].to(device=self.device, dtype=torch.bool)
            terms = result.get("reward_terms")
            if isinstance(terms, dict):
                self._last_reward_terms = terms
                if self._rollout_reward_terms is None:
                    self._rollout_reward_terms = {k: [] for k in terms}
                for key, tensor in terms.items():
                    if key not in self._rollout_reward_terms:
                        self._rollout_reward_terms[key] = []
                    self._rollout_reward_terms[key].append(
                        torch.as_tensor(tensor).detach().clone()
                    )
            contact = result.get("contact")
            wall = result.get("wall_contact")
            if contact is not None and self._rollout_contact is not None:
                self._rollout_contact.append(
                    torch.as_tensor(contact).detach().clone().to(dtype=torch.bool)
                )
            if wall is not None and self._rollout_wall_contact is not None:
                self._rollout_wall_contact.append(
                    torch.as_tensor(wall).detach().clone().to(dtype=torch.bool)
                )

            # reset_mask must be the action-time GRU clear (pending), not the
            # post-step terminal mask — evaluate_actions clears before scoring.
            self.buffer.store_step(
                step,
                state=compact_before,
                next_state=result["next_compact_state"],
                actions=actions,
                rewards=rewards,
                valid=valid,
                done=done,
                timeout=timeout,
                reset_mask=pending_reset.clone(),
                track_id=arrays.track_id,
                condition=cond_store,
                sensor_noise_seed=noise_seed,
                episode_id=episode_id,
                episode_step=episode_step,
                old_logp=logp,
                obs_digest=observation_digest(sensor_obs),
                pre_tanh=pre_tanh,
            )
            self.ppo.advance_carry(
                next_hidden, done=done, reset_mask=reset_mask
            )
            hidden = self.ppo.carry_hidden
            assert hidden is not None

        self.profile["collect_s"] = time.perf_counter() - t0
        batch = self.buffer.finalize()
        self._last_batch = batch
        return batch

    def _bootstrap_values(self, batch, world_id: torch.Tensor) -> torch.Tensor:
        """V(s'_t) from the pre-respawn next state, for truncation bootstrap."""
        assert self.critic is not None
        cfg = self.cfg
        device = self.device
        next_state = batch.next_state.to(device=device, dtype=torch.float32)
        next_active = next_state[..., STATE_INDEX["active"]] > 0.5
        next_ego, next_others, next_mask = pack_critic_features(
            next_state,
            world_id=world_id,
            active=next_active,
            max_agents_per_world=cfg.worlds.max_agents_per_world,
            track_id=batch.track_id,
            atlas=self._preview_atlas,
            centralized=bool(cfg.ablations.centralized_critic),
        )
        condition = batch.condition.to(device=device, dtype=torch.float32)
        # Same numeric path as the PPO critic pass, so V(s) and V(s') in the
        # GAE delta come from one precision.
        assert self.ppo is not None
        with torch.no_grad(), torch.amp.autocast(
            device_type=torch.device(device).type,
            dtype=self.ppo.amp_dtype,
            enabled=bool(self.ppo.amp),
        ):
            return critic_values_over_time(
                self.critic, next_ego, next_others, next_mask, condition
            )

    def _replay_observations(self, batch) -> torch.Tensor:
        """Re-run the sensor kernels over every stored compact state."""
        assert self.sim is not None
        t_steps, num_slots, _ = batch.state.shape
        device = self.device
        sensor = torch.zeros(
            t_steps,
            num_slots,
            self.cfg.agents.sensor_obs_dim,
            device=device,
            dtype=torch.float32,
        )
        live = self.sim.buffers.torch_arrays
        live_state = self.sim.pack_state()
        live_seed = live.sensor_noise_seed.clone()
        live_episode = live.episode_id.clone()
        live_step = live.episode_step.clone()
        live_track = live.track_id.clone()
        # Slots keep their track for the whole rollout; respawn only moves them.
        live.track_id.copy_(batch.track_id)
        for step in range(t_steps):
            self.sim.restore_state(batch.state[step])
            # Corruption params ride in the compact state; the noise RNG stream is
            # keyed on (seed, slot, episode_id, episode_step) instead.
            live.sensor_noise_seed.copy_(batch.sensor_noise_seed[step])
            live.episode_id.copy_(batch.episode_id[step])
            live.episode_step.copy_(batch.episode_step[step])
            sensor[step] = self.sim.rebuild_sensors().to(device=device)
        self.sim.restore_state(live_state)
        live.sensor_noise_seed.copy_(live_seed)
        live.episode_id.copy_(live_episode)
        live.episode_step.copy_(live_step)
        live.track_id.copy_(live_track)
        self.sim.rebuild_sensors()
        return sensor

    def _assert_observation_digests(
        self, batch, sensor: torch.Tensor, active: torch.Tensor
    ) -> None:
        """Fail the update when replay does not reproduce the collected obs."""
        # Sensor kernels skip inactive rows, so those observations are leftovers
        # from an earlier rebuild and carry no policy input to compare.
        mismatch = (
            observation_digest(sensor) != batch.obs_digest.to(device=sensor.device)
        ) & active
        count = int(mismatch.sum().item())
        if count:
            step, slot = mismatch.nonzero(as_tuple=False)[0].tolist()
            raise ReconstructionParityError(
                f"replayed observation differs from collection on {count} "
                f"agent-steps (first step={step} slot={slot}); compact state no "
                "longer determines what the policy saw"
            )

    def reconstruct_prepared(
        self, batch, *, verify_parity: bool = False
    ) -> PreparedPPOInputs:
        """Replay actor observations from compact state and pack critic features.

        Observations are never stored, so the rollout keeps only a per-step digest
        and a mismatch here is fatal rather than advisory. Behavior log-probs are
        rescored against the replayed observations with the PPO evaluate path, so
        a compact active-only collection batch cannot desync the ratio gate.

        This rescore overwrites ``batch.old_logp`` (collect-path, one GRU step at
        a time) with an evaluate-path value (whole-sequence scoring), computed
        with the same frozen weights. That is intentional: it makes the epoch-0
        PPO ratio exactly 1.0 regardless of any collect/evaluate numerical drift.
        The cost is that ``ppo._assert_collect_evaluate_parity`` runs entirely
        after this line, so it compares the evaluate path against itself and
        cannot detect genuine collect-vs-evaluate divergence (measured up to
        6.1e-3 in log-prob at production dimensions). The bit-exact observation
        digest check just above is what actually guards reconstruction
        correctness; see docs in ``ppo.py`` at the parity-gate call site.

        The actor's running sensor normalizer (see ``normalization.py``) is
        read, never written, anywhere in this method or in ``ppo.update``: its
        statistics stay frozen for the whole collect/reconstruct/update cycle
        so every rescore of this rollout's observations — here and inside
        ``ppo.update`` — sees exactly what collection saw. ``train_update``
        folds this rollout's observations into the statistics only after
        ``ppo.update`` returns, so the next rollout's collection is the first
        thing to see the update.
        """
        assert self.sim is not None and self.actor is not None
        cfg = self.cfg
        t_steps, num_slots, _ = batch.state.shape
        device = self.device

        sensor = self._replay_observations(batch)
        active = (batch.state[..., STATE_INDEX["active"]] > 0.5).to(device=device)
        self._assert_observation_digests(batch, sensor, active)

        with torch.no_grad():
            logp_bt, _, _ = evaluate_actions_sequence(
                self.actor,
                sensor.transpose(0, 1).contiguous(),
                batch.condition.transpose(0, 1).contiguous(),
                batch.gru_start,
                batch.actions.transpose(0, 1).contiguous(),
                reset_mask=batch.reset_mask.transpose(0, 1).contiguous(),
                pre_tanh=batch.pre_tanh.transpose(0, 1).contiguous(),
            )
            batch.old_logp.copy_(logp_bt.transpose(0, 1))

        world_id = torch.arange(num_slots, device=device) // cfg.worlds.max_agents_per_world
        world_id = world_id.view(1, num_slots).expand(t_steps, num_slots)
        ego, others, other_mask = pack_critic_features(
            batch.state.to(device=device),
            world_id=world_id,
            active=active,
            max_agents_per_world=cfg.worlds.max_agents_per_world,
            track_id=batch.track_id,
            atlas=self._preview_atlas,
            centralized=bool(cfg.ablations.centralized_critic),
        )
        prepared = PreparedPPOInputs(
            sensor_obs=sensor,
            ego_state=ego,
            other_agents=others,
            other_mask=other_mask,
            bootstrap_values=self._bootstrap_values(batch, world_id),
        )
        if verify_parity:
            self.profile.update({"reconstruction_digest_mismatches": 0.0})
        return prepared

    def _update_sensor_normalizer(self, batch, prepared: PreparedPPOInputs) -> None:
        """Fold this rollout's raw observations into the actor's running stats.

        Must run after ``ppo.update`` returns, not before: the actor applies
        its normalizer internally on every ``forward``/``evaluate_actions_sequence``
        call, and both the collect-time actions and every rescoring in
        ``reconstruct_prepared``/``ppo.update`` (the parity gate and every PPO
        minibatch) must see the same frozen statistics or ``old_logp`` becomes
        unreproducible mid-update. Statistics only advance for the *next*
        rollout's collection.
        """
        active = (batch.state[..., STATE_INDEX["active"]] > 0.5).to(
            device=prepared.sensor_obs.device
        )
        self.actor.sensor_normalizer.update(prepared.sensor_obs[active])

    def train_update(self) -> TrainProgress:
        if not self._setup_done:
            self.setup()
        assert self.ppo is not None and self.actor is not None
        t0 = time.perf_counter()
        batch = self.collect_rollout()
        t1 = time.perf_counter()
        prepared = self.reconstruct_prepared(batch, verify_parity=True)
        t2 = time.perf_counter()
        stats = self.ppo.update(batch, prepared)
        self._update_sensor_normalizer(batch, prepared)
        t3 = time.perf_counter()
        n_trans = int(batch.valid.sum().item())
        self.transitions += n_trans
        active_end = int(
            (self.sim.buffers.torch_arrays.active > 0).sum().item()
        )
        slots = int(self.layout.num_slots)
        rollout_cells = int(batch.valid.numel())
        self.profile.update(
            {
                "collect_s": t1 - t0,
                "reconstruct_s": t2 - t1,
                "ppo_s": t3 - t2,
                "update_s": t3 - t0,
                "transitions": float(n_trans),
                "transitions_per_s": float(n_trans / max(t3 - t0, 1e-9)),
                "active_end": float(active_end),
                "valid_frac": float(n_trans / max(rollout_cells, 1)),
            }
        )
        arrays = self.sim.buffers.torch_arrays
        spawn = getattr(self.sim, "_spawn_stats", None)
        reward_terms_ts: dict[str, Any] | None = None
        if self._rollout_reward_terms:
            reward_terms_ts = {
                k: torch.stack(vs, dim=0)
                for k, vs in self._rollout_reward_terms.items()
                if vs
            }
        contact_ts = None
        wall_ts = None
        if self._rollout_contact:
            contact_ts = torch.stack(self._rollout_contact, dim=0)
        if self._rollout_wall_contact:
            wall_ts = torch.stack(self._rollout_wall_contact, dim=0)
        reward_means, track_stats, density_stats, sim_stats = summarize_rollout_aux(
            rewards=batch.rewards,
            valid=batch.valid,
            track_id=batch.track_id,
            active_end=arrays.active,
            max_agents_per_world=int(self.cfg.worlds.max_agents_per_world),
            reward_terms=reward_terms_ts or self._last_reward_terms,
            done=batch.done,
            timeout=batch.timeout,
            reset_mask=batch.reset_mask,
            contact=contact_ts,
            wall_contact=wall_ts,
            sim_arrays=arrays,
            spawn_stats=spawn if isinstance(spawn, dict) else None,
        )
        filter_ewma = 0.0
        if getattr(self.ppo, "filter_state", None) is not None:
            filter_ewma = float(self.ppo.filter_state.ewma_max_abs_adv)
        metrics = {
            "retention": float(stats.retention),
            "policy_loss": float(stats.policy_loss),
            "ppo_surrogate_loss": float(stats.surrogate_loss),
            "entropy_loss": float(stats.entropy_loss),
            "entropy_coef": float(stats.entropy_coef),
            "value_loss": float(stats.value_loss),
            "entropy": float(stats.entropy),
            "approx_kl": float(stats.approx_kl),
            "candidate_kl": float(stats.candidate_kl),
            "full_rollout_kl": float(stats.full_rollout_kl),
            "pre_update_approx_kl": float(stats.pre_update_approx_kl),
            "logp_delta_max": float(stats.logp_delta_max),
            "logp_delta_mean": float(stats.logp_delta_mean),
            "action_pretanh_max_abs": float(stats.action_pretanh_max_abs),
            "clip_fraction": float(stats.clip_fraction),
            "grad_norm": float(stats.grad_norm),
            "actor_grad_norm": float(stats.actor_grad_norm),
            "critic_grad_norm": float(stats.critic_grad_norm),
            "actor_update_to_weight_norm": float(
                stats.actor_update_to_weight_norm
            ),
            "actor_steps": float(stats.actor_steps),
            "critic_steps": float(stats.critic_steps),
            "actor_rolled_back": float(stats.actor_rolled_back),
            "actor_rollback_full_rollout": float(
                stats.actor_rollback_full_rollout
            ),
            "actor_rollback_count": float(stats.actor_rollback_count),
            "consecutive_actor_rollbacks": float(
                stats.consecutive_actor_rollbacks
            ),
            "actor_lr_safety_multiplier": float(
                stats.actor_lr_safety_multiplier
            ),
            "rollback_stop_requested": float(stats.rollback_stop_requested),
            "actor_kl_warmup_active": float(stats.actor_kl_warmup_active),
            "rollback_free_accepted_updates": float(
                stats.rollback_free_accepted_updates
            ),
            "learning_rate": float(stats.learning_rate),
            "filter_eta": float(stats.filter_eta),
            "filter_ewma_max_abs_adv": filter_ewma,
            "early_stopped": float(stats.early_stopped),
            "epochs_completed": float(stats.epochs_completed),
            "transitions_per_s": self.profile["transitions_per_s"],
            "valid_transitions": float(n_trans),
            "valid_frac": float(n_trans / max(rollout_cells, 1)),
            "active_end": float(active_end),
            "active_frac_end": float(active_end / max(slots, 1)),
            "reconstruction_digest_mismatches": float(
                self.profile.get("reconstruction_digest_mismatches", 0.0)
            ),
            **stats.telemetry,
            **sim_stats,
        }
        progress = TrainProgress(
            update_index=int(self.ppo.update_index),
            transitions=int(self.transitions),
            metrics=metrics,
        )
        interval = max(1, int(self.cfg.profiling.report_interval_updates))
        should_report = progress.update_index % interval == 0
        if self.run_dir is not None and self.cfg.profiling.enabled and should_report:
            path = self.run_dir / f"metrics_{progress.update_index:06d}.json"
            path.write_text(
                json.dumps(
                    {
                        "progress": progress.metrics,
                        "profile": self.profile,
                        "reward_terms": reward_means,
                        "track": track_stats,
                        "density": density_stats,
                        "sim": sim_stats,
                        "wandb_run_id": (
                            None if self.wandb is None else self.wandb.run_id
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if self.wandb is not None and self.wandb.active and should_report:
            payload = flatten_train_metrics(
                progress_metrics=progress.metrics,
                profile=self.profile,
                update_index=progress.update_index,
                transitions=progress.transitions,
                reward_term_means=reward_means,
                track_stats=track_stats,
                density_stats=density_stats,
                sim_stats=sim_stats,
                device=self.device,
            )
            self.wandb.log_metrics(payload, step=progress.update_index)
        return progress

    def save_checkpoint(
        self,
        path: str,
        *,
        shutdown_reason: str | None = None,
        behavior_gate: dict[str, Any] | None = None,
    ) -> None:
        if not self._setup_done:
            self.setup()
        assert self.ppo is not None and self.sim is not None
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config": config_to_dict(self.cfg),
            "transitions": int(self.transitions),
            "ppo": export_resume_state(self.ppo),
            "sim_compact_state": self.sim.pack_state().detach().cpu(),
            # Device styles are authoritative: respawn scatters them on device
            # and never refreshes the host list.
            "styles_raw": self.sim.raw_styles().detach().cpu().numpy(),
            "track_manifest_hash": track_manifest_hash(self.cfg),
            "numpy_rng": np.random.get_state(),
            "wandb_run_id": (
                None if self.wandb is None else self.wandb.run_id
            ),
            "shutdown_reason": shutdown_reason,
            "saved_at": _utc_now(),
        }
        if behavior_gate is not None:
            payload["behavior_gate"] = behavior_gate
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        if not self._setup_done:
            self.setup()
        assert self.ppo is not None and self.sim is not None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = int(payload.get("checkpoint_version", -1))
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint_version={version}; expected "
                f"{CHECKPOINT_VERSION} with split actor/critic optimizers"
            )
        saved_hash = payload.get("track_manifest_hash")
        current_hash = track_manifest_hash(self.cfg)
        if saved_hash != current_hash:
            raise ValueError(
                "track manifest mismatch: checkpoint was trained against "
                f"{saved_hash!r}, current config resolves to {current_hash!r}"
            )
        load_resume_state(self.ppo, payload["ppo"])
        if payload.get("numpy_rng") is not None:
            np.random.set_state(payload["numpy_rng"])
        self.transitions = int(payload.get("transitions", 0))
        if "sim_compact_state" in payload:
            self.sim.restore_state(
                payload["sim_compact_state"].to(self.sim.buffers.torch_device)
            )
            self.sim.rebuild_sensors()
        if "styles_raw" in payload:
            from gigaflow_f1tenth.rewards import CONDITION_FIELD_NAMES, PrivateStyle

            raw = np.asarray(payload["styles_raw"], dtype=np.float32)
            styles = [
                PrivateStyle(
                    **{
                        name: float(raw[i, j])
                        for j, name in enumerate(CONDITION_FIELD_NAMES)
                    }
                )
                for i in range(raw.shape[0])
            ]
            self.sim.apply_styles(styles)

    def export_actor(self, path: str) -> None:
        if not self._setup_done:
            self.setup()
        assert self.actor is not None
        export_actor_artifact(self.cfg, self.actor, path)


def build_trainer(
    cfg: ExperimentConfig,
    *,
    atlas: PackedTrackAtlasView | None = None,
    device: str | None = None,
    run_dir: str | Path | None = None,
    wandb_session: WandbSession | None = None,
) -> SelfPlayTrainer:
    return SelfPlayTrainer(
        cfg,
        atlas=atlas,
        device=device,
        run_dir=run_dir,
        wandb_session=wandb_session,
    )


def run_training(
    cfg: ExperimentConfig,
    num_updates: int,
    *,
    atlas: PackedTrackAtlasView | None = None,
    device: str | None = None,
    run_dir: str | Path | None = None,
    checkpoint_interval: int = 0,
    resume_from: str | Path | None = None,
    wandb_session: WandbSession | None = None,
) -> TrainProgress:
    """Run PPO updates with transactional shutdown and cadenced evaluation."""
    checkpoint_wandb_run_id = None
    resume_preview = None
    if resume_from is not None:
        resume_preview = torch.load(
            str(resume_from), map_location="cpu", weights_only=False
        )
        checkpoint_wandb_run_id = resume_preview.get("wandb_run_id")

    session = wandb_session
    if session is None:
        session = build_wandb_session(
            cfg,
            run_dir=run_dir,
            checkpoint_wandb_run_id=(
                None if checkpoint_wandb_run_id is None else str(checkpoint_wandb_run_id)
            ),
        )

    trainer = build_trainer(
        cfg,
        atlas=atlas,
        device=device,
        run_dir=run_dir,
        wandb_session=session,
    )
    progress = TrainProgress(update_index=0, transitions=0, metrics={})
    async_eval = AsyncCpuEvalManager(cfg, run_dir=run_dir)
    gate_state = BehaviorGateState()
    if resume_preview is not None:
        gate_payload = resume_preview.get("behavior_gate")
        if isinstance(gate_payload, dict):
            gate_state = BehaviorGateState.from_dict(gate_payload)
    actor_every = int(cfg.ppo.actor_checkpoint_interval_updates)
    full_every = (
        int(checkpoint_interval)
        if int(checkpoint_interval) > 0
        else int(cfg.ppo.full_checkpoint_interval_updates)
    )
    resolved_device = _resolve_device(cfg, device)
    run_manifest = _initial_run_manifest(
        cfg,
        num_updates=int(num_updates),
        device=resolved_device,
        actor_checkpoint_interval=actor_every,
        full_checkpoint_interval=full_every,
        resume_from=resume_from,
    )
    if run_dir is not None:
        _write_run_manifest(run_dir, run_manifest)

    requested_signal: dict[str, str | None] = {"name": None}

    def _request_stop(signum, _frame) -> None:
        if requested_signal["name"] is None:
            requested_signal["name"] = signal.Signals(signum).name

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _request_stop)
        except ValueError:
            previous_handlers.clear()
            break

    shutdown_reason = "completed"
    unexpected: BaseException | None = None
    try:
        trainer.setup()
        session.update_runtime_metadata(
            {
                "argv": run_manifest["argv"],
                "actual_num_updates": int(num_updates),
                "actual_device": resolved_device,
                "actor_checkpoint_interval_updates": actor_every,
                "full_checkpoint_interval_updates": full_every,
                "resume_from": run_manifest["resume_from"],
                "source_tree_sha256": run_manifest["source_tree_sha256"],
                "atlas": run_manifest["atlas"],
                "started_at": run_manifest["started_at"],
            }
        )
        if resume_from is not None:
            trainer.load_checkpoint(str(resume_from))
        eval_every = int(cfg.wandb.eval_interval_updates)
        for _ in range(int(num_updates)):
            progress = trainer.train_update()
            eval_status = async_eval.poll_and_upload(
                session, train_step=progress.update_index
            )
            eval_stop = _consume_behavioral_eval(
                cfg,
                gate_state,
                eval_status,
                session=session,
                run_dir=run_dir,
                train_step=progress.update_index,
            )
            actor_path = None
            if run_dir is not None and actor_every > 0:
                if progress.update_index % actor_every == 0:
                    actor_path = (
                        Path(run_dir) / f"actor_{progress.update_index:06d}.pt"
                    )
                    trainer.export_actor(str(actor_path))
            if run_dir is not None and full_every > 0:
                if progress.update_index % full_every == 0:
                    ckpt = Path(run_dir) / f"ckpt_{progress.update_index:06d}.pt"
                    trainer.save_checkpoint(
                        str(ckpt), behavior_gate=gate_state.to_dict()
                    )
                    if actor_path is None:
                        actor_path = (
                            Path(run_dir)
                            / f"actor_{progress.update_index:06d}.pt"
                        )
                        trainer.export_actor(str(actor_path))
                    session.log_artifact_refs(
                        step=progress.update_index,
                        checkpoint_path=ckpt,
                        actor_path=actor_path,
                    )
            if (
                eval_every > 0
                and progress.update_index > 0
                and progress.update_index % eval_every == 0
            ):
                _run_cadenced_eval(
                    cfg,
                    trainer,
                    session,
                    step=progress.update_index,
                    run_dir=run_dir,
                    async_eval=async_eval,
                )
            if cfg.evaluation.quality_stop_enabled:
                if eval_stop is not None:
                    shutdown_reason = eval_stop
                    break
                if (
                    not gate_state.feasible
                    and progress.update_index
                    >= int(cfg.evaluation.behavior_no_feasibility_stop_updates)
                ):
                    shutdown_reason = "behavior_no_feasibility_by_deadline"
                    break
                if bool(progress.metrics.get("rollback_stop_requested", 0.0)):
                    shutdown_reason = "repeated_actor_rollback"
                    break
            if requested_signal["name"] is not None:
                shutdown_reason = f"requested_{requested_signal['name'].lower()}"
                break
        if run_dir is not None:
            final_ckpt = Path(run_dir) / "ckpt_final.pt"
            final_actor = Path(run_dir) / "actor_final.pt"
            trainer.save_checkpoint(
                str(final_ckpt),
                shutdown_reason=shutdown_reason,
                behavior_gate=gate_state.to_dict(),
            )
            trainer.export_actor(str(final_actor))
            session.log_artifact_refs(
                step=progress.update_index,
                checkpoint_path=final_ckpt,
                actor_path=final_actor,
            )
            # Touch a conservative deployment style vector for artifact consumers.
            _ = normalize_condition_vector(
                deployment_style(
                    cfg.evaluation.conservative_deployment_style
                ).raw_vector()
            )
    except BaseException as exc:
        unexpected = exc
        shutdown_reason = f"exception_{type(exc).__name__}"
        if run_dir is not None and trainer._setup_done:
            emergency_ckpt = Path(run_dir) / (
                f"ckpt_emergency_{progress.update_index:06d}.pt"
            )
            emergency_actor = Path(run_dir) / (
                f"actor_emergency_{progress.update_index:06d}.pt"
            )
            try:
                trainer.save_checkpoint(
                    str(emergency_ckpt),
                    shutdown_reason=shutdown_reason,
                    behavior_gate=gate_state.to_dict(),
                )
                trainer.export_actor(str(emergency_actor))
            except Exception as save_exc:
                warnings.warn(
                    f"emergency checkpoint failed: {save_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    finally:
        async_eval.shutdown(session, train_step=progress.update_index)
        if run_dir is not None:
            run_manifest["finished_at"] = _utc_now()
            run_manifest["shutdown_reason"] = shutdown_reason
            run_manifest["final_update_index"] = int(progress.update_index)
            run_manifest["wandb_run_id"] = session.run_id
            _write_run_manifest(run_dir, run_manifest)
        session.update_runtime_metadata(
            {
                **dict(session.provenance.get("runtime") or {}),
                "finished_at": run_manifest.get("finished_at") or _utc_now(),
                "shutdown_reason": shutdown_reason,
                "final_update_index": int(progress.update_index),
            }
        )
        session.finish()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if unexpected is not None:
        raise unexpected
    return progress


def _consume_behavioral_eval(
    cfg: ExperimentConfig,
    gate_state: BehaviorGateState,
    status: dict[str, Any] | None,
    *,
    session: WandbSession,
    run_dir: str | Path | None,
    train_step: int,
) -> str | None:
    if (
        run_dir is None
        or status is None
        or status.get("state") != "completed"
        or status.get("step") is None
    ):
        return None
    source_step = int(status["step"])
    out_dir = eval_output_dir(run_dir, source_step)
    report_path = out_dir / "eval_report.json"
    if not report_path.is_file():
        return None
    reports = load_reports_from_json(report_path)
    decision = gate_state.observe(
        reports,
        required=cfg.evaluation.suite,
        step=source_step,
        feasibility_passes=int(cfg.evaluation.behavior_feasibility_passes),
    )
    if decision.best_safe:
        source_actor = out_dir / ACTOR_SNAPSHOT
        if source_actor.is_file():
            shutil.copy2(
                source_actor,
                Path(run_dir) / f"actor_best_safe_{source_step:06d}.pt",
            )
            shutil.copy2(source_actor, Path(run_dir) / "actor_best_safe.pt")
    state_payload = {
        "source_step": source_step,
        "train_step": int(train_step),
        "passed": decision.passed,
        "catastrophic": decision.catastrophic,
        "stop_requested": decision.stop_requested,
        "warning_active": decision.warning_active,
        "score": decision.score,
        "gates": decision.gates,
        "consecutive_failures": gate_state.consecutive_failures,
        "consecutive_passes": gate_state.consecutive_passes,
        "feasible": gate_state.feasible,
        "best_safe_lap_s": gate_state.best_safe_lap_s,
        "best_safe_score": gate_state.best_safe_score,
        "frontier": [
            {"vector": list(vector), "score": score, "step": step}
            for vector, score, step in gate_state.frontier
        ],
    }
    (Path(run_dir) / "behavior_gate_state.json").write_text(
        json.dumps(state_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    session.log_metrics(
        {
            "eval/gate/pass": float(decision.passed),
            "eval/gate/catastrophic": float(decision.catastrophic),
            "eval/gate/warning_active": float(decision.warning_active),
            "eval/gate/score": float(decision.score),
            "eval/gate/consecutive_failures": float(
                gate_state.consecutive_failures
            ),
            "eval/gate/consecutive_passes": float(gate_state.consecutive_passes),
            "eval/gate/feasible": float(gate_state.feasible),
        },
        step=int(train_step),
    )
    if not cfg.evaluation.quality_stop_enabled or not decision.stop_requested:
        return None
    return (
        "catastrophic_behavioral_gate"
        if decision.catastrophic
        else "consecutive_behavioral_gate_failures"
    )


def _run_cadenced_eval(
    cfg: ExperimentConfig,
    trainer: SelfPlayTrainer,
    session: WandbSession,
    *,
    step: int,
    run_dir: str | Path | None,
    async_eval: AsyncCpuEvalManager | None = None,
) -> None:
    """Cadenced eval: async CPU subprocess when configured, else sync in-process."""
    if not session.active:
        return
    eval_device = resolve_eval_device(cfg)
    if eval_device == "cpu" and async_eval is not None and run_dir is not None:
        t0 = time.perf_counter()
        result = async_eval.try_launch(
            step=step, actor=trainer.actor, session=session
        )
        # Launch itself is cheap (snapshot + spawn); wall time stays on the child.
        trainer.profile["eval_s"] = time.perf_counter() - t0
        trainer.profile["eval_async_launch"] = 1.0 if result == "launched" else 0.0
        trainer.profile["eval_async_skipped"] = (
            1.0 if result == "skipped_inflight" else 0.0
        )
        return

    from gigaflow_f1tenth.evaluation import run_evaluation

    out_dir = None if run_dir is None else Path(run_dir) / f"eval_{step:06d}"
    t0 = time.perf_counter()
    try:
        reports = run_evaluation(
            cfg,
            suites=cfg.evaluation.suite,
            atlas=trainer.atlas,
            actor=trainer.actor,
            device=eval_device if eval_device == "cpu" else trainer.device,
            output_dir=out_dir,
        )
    except Exception as exc:
        warnings.warn(
            f"cadenced W&B eval failed (continuing train): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    trainer.profile["eval_s"] = time.perf_counter() - t0
    media = []
    if out_dir is not None and out_dir.is_dir():
        media = sorted(
            list(out_dir.glob("*.png"))
            + list(out_dir.glob("*.gif"))
            + list(out_dir.glob("*.mp4"))
            + list(out_dir.glob("*.webm"))
        )
    session.log_evaluation(
        reports,
        source_step=int(step),
        global_step=int(step),
        media_paths=media,
    )

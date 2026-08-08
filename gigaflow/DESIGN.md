# Gigaflow F1TENTH Self-Play — Design Specification

Status: Accepted (see repo ADR 0027).  
Authority for shapes, units, masks, and module seams.

## Isolation

- All runtime code lives under `gigaflow/` (`src/gigaflow_f1tenth/`).
- Runtime **must not** import `training/`, `f1tenth_env`, `f1tenth_sim`, `qrsac`,
  `libs/f1tenth_policy`, or `libs/f1tenth_contract`.
- Existing stacks are references for formulas and parity fixtures only.

## Module map

| Module | Owns | Stable interface |
| --- | --- | --- |
| `config` | Versioned YAML schema, layout constants, startup validation | `ExperimentConfig`, `load_config`, `validate_config` |
| `tracks` | Pinned track prep + packed atlas (23 upstream centerlines) + ego-relative preview sampling | `TrackAtlas`, `PackedTrackAtlasView`, `sample_track_lookahead` |
| `kernels` | Warp physics/contact/sensors + reward overlay | `Simulator`, `SimulatorState`, `WorldSlotLayout` |
| `buffers` | Compact state-only rollouts + pack/unpack | `RolloutBuffer`, `CompactRolloutBatch`, `BufferShapes` |
| `model` | Shared CNN-GRU actor + private condition | `ConditionedActor`, `ActorShapes` |
| `normalization` | Running sensor mean/std, owned by the actor | `SensorNormalizer` |
| `critic` | Training-only Deep Sets value + feature packing | `CentralValueCritic`, `pack_critic_features` |
| `ppo` | Recurrent PPO + adaptive advantage filter | `PPOLearner`, `AdaptiveFilterState` |
| `trainer` | Single-machine collect/reconstruct/update/checkpoint | `Trainer`, `build_trainer`, `run_training` |
| `wandb_log` | Optional W&B session (offline/online; local logs authoritative) | `WandbSession`, `build_wandb_session` |
| `artifacts` | Deployable actor packages | `ArtifactManifest`, `ArtifactStore` |
| `evaluation` | Suites, viz, soak, ablations, promotion gates | `Evaluator`, `EvalReport` |
| `dense_traffic` | Max-agents dense-world experiment metrics + promotion gates | `measure_variant`, `promotion_gates` |
| `cli` | validate-config / prepare-tracks / train / evaluate / soak / benchmark | `gigaflow` entry point |

## Tensor shapes and units

| Quantity | Shape / value | Units / notes |
| --- | --- | --- |
| LiDAR | `[B, 1081]` | metres, 270° FOV, angularly ordered |
| Proprio | `[B, 16]` | deploy-parity sensor channels |
| Sensor obs | `[B, 1097]` | LiDAR ‖ proprio; deploy contract |
| Private condition | `[B, C]` | normalized side channel; **not** in 1097-D |
| Actions | `[B, 2]` | `[force_norm, steering_delta_norm]`, tanh-squashed |
| GRU hidden | `[B, 512]` | per-agent; cleared only on true reset rows |
| CNN projection | `256` | before condition concat into GRU |
| Actor MLP | `[1024, 1024, 1024]` | squashed Gaussian head |
| Critic set | `[B, N, D]` + mask `[B, N]` | unordered; masked max-pool |
| Critic ego branch | `[B, 45 + 100]` | compact state ‖ ordered track preview (never pooled) |
| Track preview | `[B, K=20, 5]` flattened to 100 | ego-frame (x, y), right/left half-width, curvature; nearest-first |
| World slots | `S = num_worlds * max_agents_per_world` | flattened `(world, slot)` |
| Compact rollout | state/next_state/actions/rewards/masks/seeds/h0/obs digest | **no** full obs storage |
| Compact state | `[S, 45]` | pose/dynamics/corruption/active/command history/wheel `omega`/`frenet_segment` |

Cadence: `sim_dt = 0.005 s`, `control_interval = 20`, `control_hz = 10`.  
Vehicle footprint: `0.568 × 0.296 m`.  
Pinned upstream tracks: **23** (`PINNED_UPSTREAM_TRACK_COUNT`).

## Active-agent masks

- `active`: physically present in the world (contacts + LiDAR).
- `trainable`: contributes to PPO (excludes surprise-braking / corrupted rows).
- `done`: episode end for this transition (true terminal or horizon truncation).
- `timeout`: horizon truncation only (`done & timeout`); GAE bootstraps from
  `V(next_state)`, the truncated episode's own next state. True terminals set
  `done=1, timeout=0` and bootstrap nothing. Either boundary cuts the GAE trace.
- `reset_mask`: rows whose GRU state must clear this step (true terminal or
  horizon truncation under async respawn).
- Training `step()` snapshots `done`/`timeout`/contact/wall flags and packs
  `next_compact_state` before async respawn clears slot state, so PPO stores the
  dying transition and bootstraps from its real next state.
- Inactive padding slots must never enter losses, advantage stats, or sensor writes.

## World / agent reset semantics

- Training: asynchronous per-agent respawn into verified empty Frenet regions on
  true terminal or horizon truncation; clear only that agent’s GRU state.
- Evaluation: world-synchronous, no-respawn race episodes (`evaluation.sync_no_respawn`).
- Episode horizon per track:
  `episode_seconds = clamp(target_laps * length / reference_speed, min, 360)`
  with defaults `target_laps=3`, `reference_speed=3.5 m/s`, floor `120 s` (smoke may lower).
- Seeds derive from `(global_seed, world_id, slot, episode_id)`.

## Collision ownership

- Responsibility-neutral contact events (first contact, counterpart, closing speed).
- Deterministic pair ordering; race-free gather/Jacobi impulse update.
- Shared broadphase grid for contact and LiDAR candidates.
- No cross-world contacts or observations.
- Normals / segment lengths are derived from the packed atlas in `sim.geometry`.

## Reward formulas (baseline)

- Primary: uncapped signed Frenet progress `Δs` with multi-lap unwrap.
- Private multiplicative dynamics scales via Gigaflow `X(a) = 0.5 U(a^-1,1) + 0.5 U(1,a)`.
- Table A2-style private weights for collision and boundary, plus bounded-linear
  lane-center shaping kept only as environment variation.
- Passing: GT Sophy-style `alpha_passing * mean(ego_ds - opp_ds)` over gated
  opponents, conditioned per driver with `alpha_passing ~ U(0, 6)`.
- Urban goal/stop-line, lane-align, reverse, velocity, and active-timestep terms
  omitted; finish/rank/blocking/zero-sum disabled by default.
- Simulator `step` overlays conditioned rewards onto the progress buffer each control tick.

## Actor / critic information boundary

- Actor: decentralized, physically occluded LiDAR + proprio + **own** private condition.
- Critic: omniscient masked sets over active cars in the same world + ego private condition,
  plus an ordered ego-relative track preview (`tracks.sample_track_lookahead`) concatenated
  onto the ego branch — never pooled with the opponent set, since max-pooling would destroy
  arc order. Static opponents (`trainable=0`, `active=1`) get a real preview like any other
  active slot; inactive padding is masked to zero.
- Critic inputs/weights never enter deployable artifacts.

## Training loop (integration)

1. Collect: actor inference → simulator step → compact state store (not 1097-D obs).
2. Reconstruct: restore packed state + noise seeds → rebuild sensors / critic sets.
3. Update: recurrent PPO with adaptive advantage filtering; carry GRU across rollouts.
4. Checkpoint: exact resume (actor/critic/optim/sched/filter/RNG/styles/sim state).

**Collect/evaluate parity gate — scope.** Step 2's `reconstruct_prepared`
rescores `old_logp` with the evaluate path (whole-sequence scoring) before
`ppo.update`'s `_assert_collect_evaluate_parity` gate ever runs, so that gate
compares the evaluate path against itself, not the true collect path (one GRU
step at a time) against evaluate. It makes the epoch-0 PPO ratio exactly 1.0
and catches within-evaluate-path kernel nondeterminism, but it structurally
cannot catch genuine collect-vs-evaluate divergence — measured up to 6.1e-3 in
log-prob at production dimensions. The bit-exact observation digest check in
`reconstruct_prepared` is the real guard against broken reconstruction.
Restoring the collect-time `old_logp` would reintroduce that ~6e-3 systematic
ratio bias with no gate to catch it; changing this is a separate decision from
documenting it (see comments at both sites in `trainer.py`/`ppo.py`).

The digest is computed over the **raw** (pre-normalization) observation, never
the normalized one: the actor's running sensor normalizer (`normalization.py`)
mutates over training, so a normalized digest would depend on when it ran
relative to a statistics update rather than on reconstruction correctness. The
normalizer's statistics themselves stay frozen for an entire
collect/reconstruct/update cycle — every rescore of one rollout's observations
sees the same statistics collection did — and only advance once, after
`ppo.update` returns, so they cannot desync `old_logp` mid-update.

## Deterministic RNG

- Global seed in config; per-step sensor noise seeds stored in compact rollouts for reconstruction parity.
- Track assignment uses seeded stratified shuffle so every collection batch can cover all tracks.

## Artifact compatibility

- Format version `1`; scope `simulation_training_only`.
- Contains actor weights, sensor/condition normalizers, architecture metadata, condition schema/ranges, named conservative deployment style.
- The sensor normalizer (running mean/std, Welford) is owned by the actor and applied inside its `forward`, so it rides along in `actor_state_dict` for free; the top-level `sensor_normalizer` field is a plain-tensor copy for consumers that should not have to load the whole actor. `validate_actor_artifact` rejects a missing/empty/malformed copy.
- Exact resume checkpoints (trainer/PPO) are separate and may include critic/optim/RNG/filter state.

## Config sections

`tracks`, `worlds`, `agents`, `reward_conditioning`, `ppo`, `evaluation`, `profiling`, `ablations`, `wandb`  
Validated at startup for layout parity, cadence identity, PPO constraints, and memory budget.

### W&B (optional)

- Disabled by default (`wandb.enabled: false`). Initialized only when enabled and
  `mode != disabled`.
- Modes: `online` (requires local credentials via env/`wandb login`), `offline`,
  `disabled`. API keys are never required in or written to configs / run metadata.
- Cadenced comprehensive logs at `profiling.report_interval_updates` (async
  `wandb.log`; no per-sim-step sync): PPO parity/KL/entropy/loss/grad/filter,
  sim reward/progress/collision/OOB/reset/track/density, throughput, GPU/memory,
  plus config + provenance (git SHA, track manifest, hardware).
- Checkpoint/actor path references on save (`log_artifacts`); optional mid-train
  eval/media via `eval_interval_updates`.
- Local JSON metrics + checkpoints remain authoritative if W&B is off or a
  recoverable log failure occurs.
- Run id is stored in `run_meta.json` and checkpoints for exact resume
  (`resume: allow|must|never|auto`) without creating duplicate runs.

## Evaluation / promotion

- Suites: solo, head_to_head, dense, surprise_braking, all_tracks, conservative_longform.
- Multi-car visualization + incident windows under the eval output directory.
- Live local WebGL viewer: `gigaflow view` + `viewer/` (Three.js). Replays a
  checkpoint over WebSocket without touching training; UI can switch prepared
  tracks and solo/head_to_head/dense at runtime.
- Soak tooling catches non-finite state / broadphase overflow.
- Promotion gates: finite metrics, suite coverage, solo progress, dense finiteness.
- Dense-traffic experiment (optional, non-default): compare
  `max_agents_per_world` 8 vs 10 vs 12 with capacity-rebuilt atlases, interaction /
  occlusion / spawn / throughput / VRAM gates. Configs under
  `configs/experiments/`; harness `tools/dense_traffic_experiment.py`. Production
  defaults stay at 8. Prefer trying 10 before 12.
- **Training vs eval scale:** `worlds.num_worlds` is the GPU training scale.
  `evaluation.num_worlds` (optional) and `evaluation.device` control cadenced /
  offline evaluation only. Production uses `evaluation.device: cpu` with an
  async subprocess (`CUDA_VISIBLE_DEVICES=""` + actor-only CPU snapshot) so
  eval never allocates on the training GPU. The parent W&B run uploads completed
  local `eval_*/` JSON/media at a later poll: `define_metric("eval/*",
  step_metric="eval/source_step")` plots eval against cadence while the global
  history `step=` stays monotonic with training (never the stale cadence); the
  child never writes W&B.

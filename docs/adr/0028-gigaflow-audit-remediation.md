# 0028 — Gigaflow audit remediation: racing reward redesign, corrected bootstrap, and critic track preview

- Status: Accepted
- Date: 2026-08-05

## Context

A production Gigaflow run (the self-play stack isolated by [0027](0027-gigaflow-isolated-selfplay.md))
hit reward totals of `1e17`-`1e35` and was kept alive only by an error-term clamp
(`max=2.0`) and a `+/-100` total clamp treating the symptom. An audit of the
whole stack against its source paper (GT Sophy, Wurman et al. 2022) traced that
explosion to a mistranscribed reward term and, while tracing it, found several
more defects unrelated to the explosion: evaluation that silently scored a
synthetic one-track oval instead of the real atlas, a PPO bootstrap that
consumed a value belonging to a different (respawned) episode, checkpoints that
serialized stale host-side state over correct device state, a passing-gate
lifecycle that depended on world count, respawn leaking prior command history
into a new episode's proprioception, and two conditioning dimensions
(`accel_scale`, `vmax_scale`) that were sampled and shown to the policy but read
by no kernel.

Separately from those defects, the reward and conditioning design itself
diverged from racing's needs: the paper's urban lane terms fight a racing line,
its critic pools opponents only (more map-blind than our actor), and its
recurrent PPO pays full CNN+GRU cost on samples the advantage filter later
discards. This ADR records the decisions made in remediating both — correctness
fixes and deliberate design departures — and the throughput and validation work
declined or narrowed along the way. It does not restate the fix-by-fix plan;
see the `gigaflow_audit_remediation` plan for that. Nothing here touches
`libs/f1tenth_contract` or the on-car stack: Gigaflow's condition vector and
artifact format are private to `gigaflow/`, per 0021's isolation.

## Decision

1. **Delete the mistranscribed and disabled reward terms; keep lane-center only
   as bounded environment variation.** `reward_lane_align` and the velocity and
   timestep terms are gone entirely; `reward_lane_center` is now
   `-alpha_l_center * dt * 1_{cos(theta_f) > 0.5} * |x_f_norm - alpha_center_bias|`,
   a bounded linear shaping term, replacing an inverted exponential that had no
   `alpha` or `dt` and grew without bound instead of decaying — the actual root
   cause of the `1e17`-`1e35` explosion the clamps were papering over. The
   reverse term is also deleted, since uncapped signed progress already
   penalizes driving backwards, which incidentally removes the last consumer of
   the audit's `speed_forward` frame bug (no separate fix needed).
   *Why:* both lane terms were independently found mistranscribed from the
   paper's Table A2, and even transcribed correctly they encode urban
   objectives — hug the centerline, keep heading aligned with it — that fight a
   racing line, which hugs apexes, uses full track width, and needs slip angle
   the alignment bonus was suppressing. Lane-center survives only because a
   little geometric variation between drivers is still useful; it is
   deliberately no longer trying to be an objective.
2. **Replace the global `passing_k` constant with conditioned
   `alpha_passing ~ U(0, 6)`.** `3.0` (today's constant) becomes the sampled
   midpoint and the conservative deployment style's fixed value, so mean
   aggression is unchanged; `0` gives time-trial drivers that ignore opponents
   entirely and `6` roughly doubles today's most aggressive passing, following
   the paper's convention of letting a coefficient reach zero.
3. **Shrink `CONDITION_DIM` 16 to 10**: five reward coefficients
   (`alpha_collision`, `alpha_boundary`, `alpha_l_center`, `alpha_center_bias`,
   `alpha_passing`) plus five estimable dynamics scales (`drive_scale`,
   `steer_scale`, `accel_scale`, `vmax_scale`, `mass_kg`), with no inert
   dimensions and no `pad` slot. `latency_s` is removed rather than wired up,
   because the paper never conditions on latency at all — there was nothing to
   implement.
   *Why:* the paper's stated reason for dynamics conditioning is that "the
   policy is aware of the dynamics of the agent it controls," which only holds
   if the conditioned parameter actually changes those dynamics. A dimension
   the policy sees but that changes nothing is worse than no dimension: it
   spends observation capacity teaching the network to ignore noise.
   *Consequence, accepted:* every artifact and checkpoint saved against the old
   16-D layout is unloadable against the new schema. A behavioral baseline
   (`actor_010000.pt` evaluated on the real atlas) was captured before this
   change specifically so the effort would still have a reference point after
   it.
4. **Correct PPO's truncation bootstrap to use the pre-respawn next-state
   value, and cut the GAE trace at every episode boundary.** `runtime.step`
   now packs compact state right after the terminal/timeout flags are
   snapshotted and before async respawn overwrites it; that `next_state` is
   critic-scored and used as `V(s'_t)` for any row `compute_gae` sees marked
   `timeout`. Continuing rows still bootstrap from `values[t+1]`; true
   terminals still bootstrap zero.
   *Why:* the old code treated `done & timeout` as a continuation and read
   `values[t+1]`, which after an async respawn is the value of a *different,
   newly-spawned* episode, not the state the truncated episode actually ended
   in. The final rollout step compounded this by bootstrapping from
   `old_values[-1]`, an action-time value rather than a next-state value.
5. **Give the critic an ordered ego-relative track preview the actor does not
   see.** Twenty ego-relative centerline samples (position, left/right
   half-widths, signed curvature; `TRACK_PREVIEW_DIM = 100`) at a speed-scaled
   lookahead (`max(speed * 6, 5)` m, the WarpSim/Sophy convention) are
   concatenated onto the critic's 45-D compact-state ego branch (145-D total),
   never passed through the Deep Sets opponent pooling.
   *Why:* the paper's critic and actor are both fed track features; ours pools
   opponents only, so it was more map-blind than our own actor. Concatenation
   instead of pooling is required, not stylistic — max-pooling is
   permutation-invariant by construction, and a lookahead's whole value is its
   arc order (a hairpin two samples out reads differently from one twenty
   samples out); pooling destroys exactly that. Curvature is included because
   corner-entry speed is what a racing value function most needs to predict,
   and it is nearly free once the sampler interpolates tangents anyway.
   *Consequence, accepted:* the actor stays LiDAR-only (unchanged deploy
   contract), so training-only privileged information now diverges further
   between our critic and actor than the paper's symmetric critic/actor split
   — our critic knows more about opponents (full compact state, not sensed) but
   still less about the map than a hypothetical critic with full-track
   knowledge would.
6. **Add light static opponents to training as ordinary pinned, braked slots
   reserved from `max_agents_per_world`** (2 per environment, the light
   preset), excluded from PPO transitions (`trainable = 0`) but fully visible
   to LiDAR and contact. `place_static_opponents` was lifted out of the viewer
   backend into `sim/spawn.py` so training and the viewer share one placement
   and pinning implementation.
   *Why:* the paper models static obstacles as immobile agents in its set, and
   the viewer already implemented exactly that; training just wasn't calling
   it. Sharing the implementation was deferred until there were two real
   callers (training and the viewer) rather than done speculatively when the
   viewer alone used it.
7. **Give the actor a sensor observation normalizer with statistics frozen for
   each collect/update cycle.** `SensorNormalizer` lives on the actor (applied
   inside `forward`), is exported as required, non-empty artifact metadata, and
   is only ever updated after a rollout's `ppo.update` returns — every rescore
   of that rollout (collection, reconstruction replay, the PPO update itself)
   sees the same statistics collection did.
   *Why:* the paper normalizes all observations to `[-1, 1]`; we were feeding
   30 m LiDAR ranges alongside rad/s yaw rates into a CNN with no running
   statistics at all. Freezing statistics per cycle — rather than updating
   continuously — keeps every within-cycle rescore reproducible; the
   alternative (update-as-you-go) would make the same rollout replay to
   different observations depending on when in the cycle it was rescored, which
   the bit-exact digest guard below would then have flagged as corruption
   rather than correctness.
8. **Decline segmented (truncated) BPTT.** The chunked-BPTT throughput change
   the plan called for was not implemented. Measured advantage-filter
   retention at production dimensions is 82-98%, not the paper's assumed ~80%
   discard the change was sized against, and minibatches are formed over whole
   agent slots (not random transitions), which makes an entirely-empty
   32-step segment — the only case segmentation would actually skip work for —
   astronomically improbable in practice. This was originally a session-log-only
   measurement; the Phase 5 validation gate independently reproduced 80-97% at
   production shape, and the Phase 5 relaunch (`prod_rtx4080_audit_remediation`,
   updates 10-310+) reproduced 88-96%. All three are recorded durably in
   [`gigaflow/outputs/retention/advantage_filter_retention_prod_rtx4080_relaunch.json`](../../gigaflow/outputs/retention/advantage_filter_retention_prod_rtx4080_relaunch.json)
   (gitignored, not committed, but reproducible via the same tool and config).
   *Why recorded as a decision:* the plan explicitly called for this change: it
   was evaluated against real measurement and declined, not simply deferred for
   lack of time.
9. **Narrow the checkpoint-corruption fix (C4) to stable resume, not bitwise
   parity.** The defect fixed is specific and small: `resample_styles_for_mask`
   updated device styles but returned the stale, unrefreshed host list, and
   `trainer.py` serialized that stale list and reapplied it over correct device
   state on load. The fix serializes the authoritative device styles directly,
   adds a persisted track-manifest hash (reject resume on mismatch) and numpy
   RNG state, and bumps `CHECKPOINT_VERSION`. Serializing tire lag, wheel
   state, and lifecycle counters for exact bitwise-identical resume was
   evaluated and explicitly not pursued.
   *Why:* bitwise resume is a completeness ambition with no measured need;
   the actual requirement is that resume does not silently corrupt learning
   (comparable losses, KL, and filter retention across a save/load boundary),
   which the narrower fix already establishes. Revisit only if resume is
   observed to perturb training.
10. **Correct the production memory estimator and shrink world counts to what
    it now predicts.** `estimate_memory_bytes` previously ignored autograd
    activation memory entirely; it now accounts for every no-grad rescoring
    pass (`full_rollout_scoring_bytes`) and every PPO minibatch's retained
    backward graph (`ppo_minibatch_backward_bytes`), taking the max of the two
    since collection/reconstruction and the PPO update never hold their
    activations live at the same time. Measured against the corrected
    estimator, the RTX 4080 SUPER config drops from 128 to 48 worlds (verified
    on-card: 5.17 GiB allocated peak against a 5.94 GiB estimate); the
    H100 config drops from 1024 to 448 worlds, by the same estimator run at
    H100 scale.
    *Consequence, accepted:* the H100 figure is extrapolated from RTX 4080
    measurements and has not itself been measured on an H100 — the production
    config says so directly and calls for re-measurement before a long run.

## The silent-fallback anti-pattern

The single most reusable finding of this audit is not any one defect but a
repeated shape: at least five separate places substituted a plausible-looking
default, or downgraded a correctness check to an informational metric, instead
of failing when something was actually wrong.

- Evaluation defaulted to a synthetic one-track oval whenever no atlas was
  wired up, and the async/CPU evaluation path never wired one up at all — every
  such run scored the policy on a track it never raced, silently.
- A suite-config override that failed validation fell back to the base world
  layout, which would have scored every evaluation suite (solo, head-to-head,
  dense) on the same worlds and erased the very differences the suites exist to
  measure.
- An observation-reconstruction check computed a mismatch count and logged it
  as a metric (`reconstruction_digest_mismatches`) rather than failing the
  update, even though a mismatch means the replayed observation the PPO update
  scores is not the one collection actually saw.
- Warp's `ScopedCapture` context only *records* launches during CUDA-graph
  capture; the capturing step's own physics/sensor work was never actually
  replayed, so the first captured step silently did nothing.
- The collect/evaluate parity gate (`_assert_collect_evaluate_parity`)
  compares a rescore against `old_logp` — but `old_logp` had already been
  overwritten by an evaluate-path rescore earlier in the same update, so the
  gate was comparing the evaluate path against itself and could never detect a
  genuine collect-versus-evaluate disagreement.

Each hid a real defect for an unknown amount of training. The fix in every
case was the same shape: fail closed (raise, e.g. `ReconstructionParityError`
or a hard `ConfigError`) instead of substituting a default, and make the
gate structurally capable of catching the failure it is named for rather than
reporting a number and moving on. Apply this pattern on sight in any future
work in this stack.

## Consequences

- Old actor artifacts and trainer checkpoints (16-D condition, pre-C4 checkpoint
  format) are permanently unloadable; the captured pre-redesign baseline on
  `actor_010000.pt` is the only comparison point that survives across the cut.
- The reward's dynamic range changes (no more urban lane-align/velocity/
  timestep contributions, no exponential blowup mode), so absolute reward
  magnitudes are not comparable to the pre-redesign run; only the behavioral
  metrics captured in the baseline (lap time, progress rate, collisions/km,
  OOB/km, completion) are.
- The critic is now asymmetric with respect to the actor in two directions at
  once: more privileged about opponents (full compact state via Deep Sets) and
  now also about the track (the ordered preview), while the actor remains
  LiDAR-only. This is a wider actor/critic information gap than the paper's
  symmetric design and is an accepted, deliberate trade for keeping the deploy
  contract unchanged.
- **The collect/evaluate parity gate has a structural blind spot that remains
  after this remediation, not a bug to be fixed next**:
  `reconstruct_prepared` rescoring `old_logp` through the evaluate path before
  `ppo.update`'s own parity check runs means that check can only catch
  within-evaluate-path kernel nondeterminism (~5e-2 in log-prob from cuDNN/
  cuBLAS algorithm selection), never true collect-versus-evaluate divergence
  (measured up to 6.1e-3 in log-prob at production dimensions). The real
  reconstruction guard is the bit-exact observation digest check that runs
  immediately before it. Any future change to `old_logp`'s handling must
  preserve or replace this guard explicitly — silently restoring the
  collection-time `old_logp` would reintroduce a ~6e-3 systematic ratio bias
  with nothing left to catch it.
- D4 (segmented BPTT) stays declined until a future measurement shows either
  materially higher discard or transition-level (not slot-level) minibatching;
  revisiting it should re-run the retention measurement, not assume the
  paper's ~80% figure. Three independent measurements now back this (session
  log, validation gate, and the Phase 5 relaunch itself) — see the durable
  artifact cited under decision 8 above.
- The H100 production config carries a known re-measurement obligation before
  a long run: its 448-world estimate is extrapolated, not verified, on that
  architecture.

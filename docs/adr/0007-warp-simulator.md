# 0007 — Use Warp for batched RL simulation

- Status: Accepted
- Date: 2026-07-14
- Supersedes: 0002's designation of TorchSim as the simulation source of truth

## Context

The PyTorch simulator coupled physics, environment bookkeeping, observation
construction, and compiler behavior. Its launch and memory scaling did not meet
the large-batch GPU target. Training still needs zero-copy PyTorch policy
inference, deterministic resets, the fixed 390-value deployment contract, and
the PhysX-grounded four-wheel model described in ADR 0002.

## Decision

1. Use NVIDIA Warp 1.15 or newer for vehicle physics and environment execution.
   Keep PyTorch for policy inference, replay, and optimization. Share tensors
   through `wp.from_torch` on the current PyTorch CUDA stream.
2. Execute each control tick as three ordered launches: pair-parallel vehicle
   physics, reward/termination/reset transaction, then observation publication.
   A solo-specialized physics kernel avoids idle opponent work. Runtime loop
   bounds prevent compile-time substep unrolling.
3. Fix simulation at float32, `sim_dt=0.005`, ten substeps per control tick, and
   60 future-track samples. The policy observation is always 390 values; solo or
   hidden-opponent rows publish exact zeros in values 384 through 389.
4. Use struct-of-arrays persistent storage and deterministic reset randomness
   derived from the global seed, environment index, and episode counter. Done
   and reward describe the terminal state while the returned observation
   describes the reset state.
5. Remove the runtime backend selector and Torch simulator after cutover.
   Rollback uses source and image history, not two production implementations.

## Persistent state ownership

- Physics storage owns pose, body velocity, yaw rate, steering, effort endpoint,
  interval effort, acceleration, wheel speed, slip/load diagnostics, mass,
  friction, drive scale, and steering bias. Physics reads and writes it; reset
  initializes it; readback exposes views to training diagnostics.
- Control storage owns current and prior actions, the action-latency ring and
  head, opponent mode, scripted target and offset, and policy-opponent action.
  Reset samples per-episode values; physics consumes commands; observation
  advances prior-action state.
- Transaction storage owns Frenet segment seeds, current progress, episode and
  lap counters, termination streaks, reward terms, terminal snapshots, and
  metrics. The transaction launch is the sole owner of terminal-state commit
  and reset.
- Observation storage owns ping-pong raw and published ego buffers plus the
  opponent-policy buffer. Observation publication is the sole writer.
- Simulator state and latency history are transient training state and are not
  checkpointed. Policy artifacts record simulator and preprocessing versions.

## Consequences

Warp compilation and cache writability are runtime requirements. macOS supports
small CPU tests only; GPU training requires a tested NVIDIA driver, CUDA-enabled
PyTorch, and matching Warp CUDA support. Contact is resolved transactionally
after both cars complete the control interval. Performance claims require
synchronized scaling benchmarks and GPU profiles on each target architecture.

## Fidelity restoration addendum (2026-07-15)

Follow-up work re-established independent validation and closed a physics gap
found after cutover, while keeping the model simple.

- **Torch parity is the authority.** An external Torch `TorchVehicleSim`
  (`develop`) run is recorded once into a committed fixture
  (`training/tests/data/torch_reference_trajectory.npz`, regenerated via
  `analysis/warp_parity/generate_torch_reference.py`) and
  `training/tests/test_warp_torch_parity.py` replays the identical seeded schedule
  through Warp. Restored component tests reuse the deleted TorchSim expected
  values; a Warp value may diverge only with a documented physical-realism
  justification. There is no self-referential Warp "physical golden".
- **Load-sensitivity reference (Fz0).** `combined_pacejka` now takes a fixed
  nominal `fz0_ref = mass_nominal·g/4` (via `SimParams.fz0_ref`), not the per-env
  randomized mass. Using the randomized mass cancelled the intended load
  sensitivity under mass domain randomization. `load_ratio` keeps its per-wheel
  static denominator (mass-invariant by design). Verified by parity plus
  mass-invariance and load-sensitivity known-answer tests.
- **Suspension: single quasi-static model.** Load transfer stays quasi-static
  (front/rear/lateral weight transfer → per-wheel `Fz`). The dynamic
  spring-damper `SuspensionFilter` and a `suspension_mode` knob are intentionally
  **not** restored: the Torch reference itself defaults to quasi-static, and the
  filter's ~15 ms critically-damped lag settles within a single 20 Hz control
  tick, so its effect on the published observation is marginal.
- **No internal substeps.** `sim_dt=0.005` with the linearized-implicit
  wheel-spin update already gives tight Torch parity and NaN-free aggressive-input
  behavior; a configurable `internal_substeps` is not added (no measured accuracy
  need, and it reintroduces compile-time substep unrolling pressure that reason
  ADR point 2 explicitly avoids).
- **Tyre relaxation (default off).** Optional first-order contact-force lag
  (`tire_relax_len`, time constant `τ = tire_relax_len / max(|v|, blend)`);
  `tire_relax_len=0` is an exact pass-through so parity is unaffected.
- **Deterministic fixed-start reset.** `WarpF1tenthEnv.reset_to(pose, yaw, speed,
  …)` places cars at explicit world poses (seeded bookkeeping via the normal
  reset, then pose overwrite + Frenet reprojection) for eval/telemetry
  reproduction; identical inputs yield identical trajectories.
- **Seeded spawn jitter.** Ego heading and opponent lateral+heading spawns are
  seeded-random (bounded by local track width / a config yaw range), preserving
  full seed determinism.

### Review fixes (train/deploy parity)

- **`obs[11]` is the track-boundary proximity flag.** The fused observation
  kernel had been publishing the 1v1 box-collision flag into `OBS_CONTACT_FLAG`;
  it now publishes `boundary_distance < contact_margin_m` (default 0.08 m), the
  same signal as the C++ deploy path (`rl_obs_core.cpp`) and the contract. The
  now-unused `ContactResult` argument was dropped from `write_raw_observation`.
- **Mixed-opponent policy speed cap applies in-kernel.** The Python
  `MixedOpponentController._apply_speed_cap` never ran in the Warp path (the env
  drives the inner policy directly and resolves the mix per row in the kernel).
  A per-env `opponent_speed_cap` is now sampled on reset for policy-mode rows
  (probability `policy_speed_cap_prob`, skewed toward `policy_speed_cap_range`
  high end) and a capped policy opponent coasts (zero throttle) whenever its
  forward speed exceeds the cap, matching the intended GT Sophy-style passable
  self-play population.

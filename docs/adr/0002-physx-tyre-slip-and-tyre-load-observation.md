# 0002 — PhysX-grounded tyre slip + tyre-load observation block

- Status: Accepted
- Date: 2026-07-08

## Context

The RL observation carries per-wheel tyre slip so the policy can race at the
traction limit (GT-Sophy parity). Two problems had accumulated:

1. **Two different slip definitions.** `f1tenth_env/car.py::compute_tyre_slip`
   (the Genesis-fallback / on-car path) and the TorchSim tyre model
   (`f1tenth_sim/dynamics.py`) each normalized slip differently, and both used a
   legacy `max(|wheel_speed|, |v_fwd|)` longitudinal denominator rather than the
   modern PhysX Vehicle SDK form. TorchSim is our physically-grounded backend, so
   its tyre model should follow the PhysX source, not a legacy approximation.

2. **No normal-load signal.** The observation exposed slip but not per-wheel
   normal load `Fz`. Slip alone is ambiguous — the same slip angle at a lightly vs
   heavily loaded tyre implies very different available grip — so GT-Sophy-style
   control at the limit needs the load ratio too.

The observation layout is a shared contract (`libs/f1tenth_contract`, its C++
mirror `f1tenth_common/observation_layout.hpp`, and the deploy builders), so any
change fans out to training, the contract, both parity tests, and the on-car C++
node.

## Decision

1. **One slip definition, the modern PhysX form.** Slip normalizes by the
   longitudinal ground speed plus a fixed offset:
   `slip_ratio = (wheel_speed − v_fwd) / (|v_fwd| + min_long)` and
   `slip_angle = atan(v_lat / (|v_fwd| + min_lat))`, where `min_long` switches
   between an active offset (drive/brake torque applied) and a larger passive
   offset (coasting). Offsets (`slip_min_lat` 0.2, `slip_min_active_long` 0.1,
   `slip_min_passive_long` 0.4 m/s) are scaled down from the PhysX full-car
   defaults (1.0 / 0.1 / 4.0) for the 1/10 car and live in the env config +
   `VehicleParams`. *Why:* one PhysX-grounded definition shared by the TorchSim
   tyre model, the Genesis-fallback `compute_tyre_slip`, and the on-car C++
   builder removes a sim2real mismatch and matches the reference we trust.

2. **TorchSim is the source of truth for slip and load.** `TorchVehicleSim`
   computes slip inside its tyre model and exposes native `tyre_slip` (ratio +
   geometric slip angle) and `tyre_load` (`Fz / Fz_static`) via
   `read_wheel_state`. The env prefers backend-native values and only falls back
   to `compute_tyre_slip` / quasi-static load transfer on the Genesis backend
   (which has no tyre model). *Why:* avoids recomputing slip from world-frame
   velocities when the physics engine already has it, and keeps the observation
   consistent with the forces the car actually experienced.

3. **Add a 4-dim tyre-load block to the observation.** Per-wheel normal-load
   ratio `Fz / Fz_static` (order [LR, RR, LF, RF]) is inserted at `[380, 384)`,
   growing the base observation from 380 → 384 and the 1v1 observation from
   387 → 391 (opponent block shifts to `[384, 391)`). *Why:* gives the policy the
   load context that disambiguates slip at the grip limit.

4. **Additive, deadzoned combined-slip penalty.** The tyre-slip reward reads the
   single-source `tyre_slip` and penalizes the longitudinal and lateral channels
   additively, each past a configurable deadzone
   (`slip_deadzone_ratio`, `slip_deadzone_angle`, `slip_angle_weight`), leaving a
   controlled grip-limit regime unpenalized. *Why:* a controllable slide should
   not be punished the way pure wheelspin or an uncontrolled drift is.

## Consequences

- **Contract blast radius handled.** Updated together: the Python contract, its
  C++ mirror, `interfaces.py`, the training and deploy observation builders, the
  C++ obs builder + node caller, both parity tests, the regenerated C++ obs
  fixture, and the affected training tests. A 380-dim checkpoint is not
  loadable by the 384-dim stack — models must be retrained on the new layout.
- **Dynamics shifted.** Modernizing the TorchSim longitudinal/lateral
  denominators changes low-speed tyre forces, so the golden-trajectory regression
  reference and the tyre-slip-penalty scale were re-tuned. Pacejka coefficients
  were left unchanged; a dedicated re-tune is possible if handling drifts.
- **Deploy load is quasi-static for now.** The on-car C++ `VehicleState.tyre_load`
  defaults to the static ratio (1.0); wiring an IMU-derived quasi-static estimate
  (the node already has body `ax/ay`) is a follow-up, mirroring how per-wheel slip
  is only estimated when enabled.
- **C++ must be rebuilt in the dev container.** The C++ edits (mirror, builder,
  node, fixture, parity tests) were made but colcon build/test runs only inside
  the ROS 2 dev container, not the training venv.

# Theory-Grounded Drivetrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Warp simulator's fixed force/power drivetrain with a deterministic Velineon 3500, stock Slash 4X4 gearing, 3S battery, and VESC-equivalent current-limited drivetrain, then cold-start the approved Courtyard PPO run.

**Architecture:** Normalized policy effort is converted to an asymmetric physical motor-current request at the control boundary. A reduced steady-state electrical model applies voltage/modulation, motor-current, and battery-current constraints on every physics substep, converts allowed q-axis current to four-wheel axle torque, and leaves traction to the existing tire model. Simulator state retains physical current for slew/fault behavior and normalized applied effort for the unchanged observation contract.

**Tech Stack:** Python 3, NVIDIA Warp device functions/kernels, PyTorch tensors, pytest, flake8, JSON training configuration, ROS 2 dev-container verification.

**Spec:** `docs/superpowers/specs/2026-09-01-theory-grounded-drivetrain-design.md`

## Global Constraints

- Do not change the observation or action layout.
- Do not edit `src/vehicle/**`.
- Do not add dependencies, ROS packages, CI changes, or deploy-image changes.
- Use 3.444 kg as nominal race-ready mass.
- Use +80/-20 A motor current, +60/-10 A battery current, 120 A absolute current, and 200 A/s slew.
- Use 3500 mechanical rpm/V, four poles, 50,000 mechanical rpm maximum, 11.8225 reduction, and 0.05255 m nominal unloaded radius.
- Remove the drivetrain's independent 23 N, 5.2 N, whole-car traction, and 320 W caps.
- Keep electrical dynamics deterministic; do not add per-step physics noise.
- Treat 65 A continuous and 100 A burst as diagnostics/safety envelopes, not instantaneous torque clamps.
- Start a new Courtyard checkpoint lineage; never resume a 65 A checkpoint into this model.
- Preserve unrelated user changes in the dirty worktree.

## File structure

- Modify `training/f1tenth_sim/params.py`: define and validate the reduced electrical drivetrain parameters and copy them into `SimParams`.
- Modify `training/f1tenth_sim/drivetrain.py`: implement current/modulation limiting and return torque plus diagnostics.
- Modify `training/f1tenth_sim/dynamics.py`: hold physical current/fault/constraint state and connect the drivetrain result to wheel dynamics.
- Modify `training/f1tenth_sim/sim_warp.py`: allocate, reset, and expose drivetrain diagnostics.
- Modify `training/tests/warp_probe.py`: expose the real Warp drivetrain and command functions to CPU tests.
- Modify `training/tests/test_warp_sim_unit.py`: replace fixed-force tests with analytical and integration tests.
- Modify `training/config.py`: provide theory-grounded defaults and remove effective use of the old force/power settings.
- Modify `training/configs/courtyard_2_e2e_ppo.json`: select the approved vehicle and VESC settings for the new run.
- Modify `training/tests/test_sensor_config.py`: lock configuration values and mass-domain behavior.
- Modify `libs/f1tenth_policy/f1tenth_policy/artifact.py`: record optional simulator drivetrain metadata without changing deployment action semantics.
- Modify `libs/f1tenth_policy/test/test_layout_actor.py`: verify the optional metadata is serialized and validated when present.
- Modify `training/standalone_trainer.py`: pass the drivetrain metadata into saved artifacts.
- Modify `training/tests/test_trainer_core.py`: verify the saved-artifact call receives the selected model.

---

### Task 1: Define the physical parameter surface and invariants

**Files:**
- Modify: `training/f1tenth_sim/params.py`
- Test: `training/tests/test_warp_sim_unit.py`

**Interfaces:**
- Produces: `VehicleParams.validate() -> None` and matching scalar fields on `VehicleParams` and `SimParams`.
- Consumes: existing `VehicleParams.from_config()` and `VehicleParams.to_warp()` construction flow.

- [ ] **Step 1: Write failing tests for the published and derived parameters**

Add focused tests that require these fields and values:

```python
def test_electrical_drivetrain_defaults_match_vehicle_spec():
    p = VehicleParams()
    assert p.mass == pytest.approx(3.444)
    assert p.motor_kv_rpm_per_v == pytest.approx(3500.0)
    assert p.motor_pole_pairs == 2
    assert p.gear_ratio == pytest.approx((54 / 13) * (37 / 13))
    assert p.wheel_radius == pytest.approx(0.3302 / (2 * math.pi))
    assert p.motor_kt_nm_per_a == pytest.approx(60 / (2 * math.pi * 3500))
    assert p.i_motor_max_a == pytest.approx(80.0)
    assert p.i_motor_brake_max_a == pytest.approx(20.0)
    assert p.i_battery_max_a == pytest.approx(60.0)
    assert p.i_battery_regen_max_a == pytest.approx(10.0)
    assert p.i_abs_max_a == pytest.approx(120.0)
    assert p.i_slew_a_per_s == pytest.approx(200.0)
```

Add parametrized invalid-configuration tests covering non-positive mass, radius,
gearing, voltage, resistance, efficiency and current magnitudes; efficiency above
one; motor limit at or above the absolute threshold; and nominal voltage outside
the declared 11.1--12.6 V interval.

- [ ] **Step 2: Run the parameter tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'electrical_drivetrain_defaults or rejects_invalid_electrical' -vv
```

Expected: failures because the electrical fields and validation do not exist.

- [ ] **Step 3: Add the minimal parameter fields and validation**

Replace the force-cap fields in both parameter types with the consumed electrical
surface:

```python
motor_kv_rpm_per_v: float = 3500.0
motor_pole_pairs: int = 2
motor_max_rpm: float = 50_000.0
gear_ratio: float = (54.0 / 13.0) * (37.0 / 13.0)
battery_voltage_nominal: float = 11.34
battery_voltage_min: float = 11.1
battery_voltage_max: float = 12.6
modulation_max: float = 0.95
motor_resistance_ohm: float = 0.01
drivetrain_efficiency: float = 0.85
i_motor_max_a: float = 80.0
i_motor_brake_max_a: float = 20.0
i_battery_max_a: float = 60.0
i_battery_regen_max_a: float = 10.0
i_abs_max_a: float = 120.0
i_slew_a_per_s: float = 200.0
```

Expose `motor_kt_nm_per_a` as a property computed from `motor_kv_rpm_per_v` so the
reciprocity calculation has one source of truth. Parse the same names from
`env_cfg`, copy them to `SimParams`, and call `validate()` after URDF/config
application. Document 0.01 ohm, 0.85 efficiency, and 11.34 V as assumptions in
the configuration rather than manufacturer facts.

- [ ] **Step 4: Run parameter tests and existing config construction tests**

Run:

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'electrical_drivetrain_defaults or rejects_invalid_electrical' -vv
.venv/bin/python -m pytest training/tests/test_sensor_config.py -vv
```

Expected: new tests pass; existing failures, if any, identify old force-parameter
expectations to update only in Task 4.

- [ ] **Step 5: Commit the parameter boundary**

```bash
git add training/f1tenth_sim/params.py training/tests/test_warp_sim_unit.py
git commit -m "sim: define electrical drivetrain parameters"
```

---

### Task 2: Implement the reduced VESC drivetrain algebra

**Files:**
- Modify: `training/f1tenth_sim/drivetrain.py`
- Modify: `training/tests/warp_probe.py`
- Test: `training/tests/test_warp_sim_unit.py`

**Interfaces:**
- Produces: Warp struct `DrivetrainResult` with `wheel_torque`, `motor_current_a`, `battery_current_a`, `modulation`, `constraint`, and `fault`.
- Produces: `electrical_drive_torque(current_a, omega, drivetrain_scale, params) -> DrivetrainResult`.
- Consumes: electrical scalar fields introduced in Task 1.

- [ ] **Step 1: Write failing tests for lossless constants and wheel-speed conversion**

Add probe tests for:

```python
expected_force_per_amp = (
    (60 / (2 * math.pi * 3500))
    * ((54 / 13) * (37 / 13))
    / (0.3302 / (2 * math.pi))
)
assert expected_force_per_amp == pytest.approx(0.614, abs=0.001)
```

Require four wheel torques to sum to `Kt * current * gear_ratio` at unit
efficiency, and require 11.1 V and 12.6 V no-load speeds to be approximately
18.1 m/s and 20.5 m/s.

- [ ] **Step 2: Run those tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'lossless_force_per_amp or no_load_speed or wheel_torque_sum' -vv
```

Expected: failures because the electrical result/probe does not exist.

- [ ] **Step 3: Implement speed, modulation, and torque without current limiting**

Define constraint integer constants in `drivetrain.py` and implement these exact
steps inside the Warp function:

```text
omega_motor = mean(abs(driven wheel omega)) * gear_ratio
back_emf = Kt * omega_motor
required_voltage = back_emf + motor_resistance * current
modulation = clamp(required_voltage / battery_voltage,
                   -modulation_max, modulation_max)
motor_torque = Kt * current * drivetrain_efficiency * drivetrain_scale
```

Split motor torque by `k_drive_front`, then by two wheels per axle. Do not accept
mass or friction as inputs and do not clamp torque with `mu*m*g`.

- [ ] **Step 4: Run the lossless tests and verify GREEN**

Run the Task 2 Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Write failing motor, battery, voltage, regen, and fault tests**

Use the real Warp probe to require:

- +100 A request becomes +80 A;
- -40 A request becomes -20 A;
- at modulation 0.80, +80 A becomes +75 A because `60 / 0.8 = 75`;
- at modulation 0.80, -20 A becomes -12.5 A because `10 / 0.8 = 12.5`;
- decreasing available voltage cannot increase allowed current;
- increasing resistance cannot increase allowed current;
- zero current and zero wheel speed remain finite;
- 120 A absolute current sets `fault=1` and returns zero wheel torque; and
- every non-fault result satisfies electrical power greater than or equal to
  non-negative mechanical output power within float32 tolerance.

- [ ] **Step 6: Run the limiter tests and verify RED**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'motor_current_limit or battery_current_limit or voltage_limit or absolute_current_fault or drivetrain_power' -vv
```

Expected: failures showing requested current is not yet limited.

- [ ] **Step 7: Implement the minimal limiter intersection**

Apply limits in this order:

```text
1. reject abs(measured current) >= absolute limit as a fault
2. clip request to -motor_brake_max .. +motor_max
3. calculate voltage-feasible current from (Vmax - back_emf) / R
4. calculate input-current interval from battery limits / abs(modulation)
5. intersect intervals without reversing the request's sign
6. recompute required voltage, modulation, battery current, and torque
```

Use an epsilon branch at zero modulation so division cannot produce infinities.
Return the first active limiting reason using stable integer constants. Do not
simulate filtering, switching ripple, field weakening, or thermal derating.

- [ ] **Step 8: Run all drivetrain component tests**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py -k drivetrain -vv
```

Expected: all drivetrain algebra, conservation, monotonicity, and fault tests pass.

- [ ] **Step 9: Commit the drivetrain algebra**

```bash
git add training/f1tenth_sim/drivetrain.py training/tests/warp_probe.py \
  training/tests/test_warp_sim_unit.py
git commit -m "sim: model VESC current-limited torque"
```

---

### Task 3: Integrate physical current state and diagnostics

**Files:**
- Modify: `training/f1tenth_sim/dynamics.py`
- Modify: `training/f1tenth_sim/sim_warp.py`
- Modify: `training/f1tenth_env/kernel.py`
- Modify: `training/f1tenth_env/sensors.py`
- Modify: `training/tests/warp_probe.py`
- Test: `training/tests/test_warp_sim_unit.py`
- Test: `training/tests/test_warp_env.py`

**Interfaces:**
- Produces: current state fields `requested_motor_current_a`, `applied_motor_current_a`, `battery_current_a`, `drivetrain_modulation`, `drivetrain_constraint`, `drivetrain_fault`, `current_above_continuous_s`, and `current_rms_accumulator`.
- Produces: `WarpVehicleSim.read_drivetrain_state() -> dict[str, torch.Tensor]`.
- Preserves: `vehicle.applied_effort` as signed directional current fraction for existing sensor observations.

- [ ] **Step 1: Write failing asymmetric physical-slew tests**

Replace the normalized-slew test with cases requiring a 200 A/s change:

```python
# dt=0.05 gives 10 A per control tick
# zero -> full drive: 0, 10, 20, ... 80 A
# zero -> full brake: 0, -10, -20 A
# +80 -> -20 crosses zero in ten 10 A steps without a discontinuity
```

Assert `applied_effort = current/80` for drive and `current/20` for braking so the
existing observation remains in `[-1, 1]`.

- [ ] **Step 2: Run the slew tests and verify RED**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'physical_current_slew or directional_current_fraction' -vv
```

Expected: failures because current is stored as normalized effort.

- [ ] **Step 3: Implement current-state command handling**

In `apply_command`, map the clamped normalized action to physical target current,
slew the stored current by `i_slew_a_per_s * control_dt`, retain trapezoidal
control-interval averaging, and derive normalized `applied_effort` from the
averaged current. Add the physical current fields to `VehicleLocal` and
`VehicleBuffers`.

- [ ] **Step 4: Run slew tests and verify GREEN**

Run the Task 3 Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Write failing simulator diagnostics and traction-ownership tests**

Require a real `WarpVehicleSim` to:

- reset all current, fault, and diagnostic state;
- expose motor current, battery current, modulation, constraint, and fault;
- remain bitwise deterministic for equal action sequences;
- produce greater wheel slip at low friction without changing commanded motor
  torque, proving traction belongs to the tire model;
- latch an injected absolute-current fault and produce zero axle torque afterward;
- accumulate time and RMS current above the 65 A continuous reference; and
- remain finite under aggressive randomized actions.

- [ ] **Step 6: Run integration tests and verify RED**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'drivetrain_state or traction_owned_by_tire or fault_latches or continuous_current_diagnostic or no_nans' -vv
```

Expected: failures because the state and diagnostic API do not exist.

- [ ] **Step 7: Connect the drivetrain result to the integrator**

Call `electrical_drive_torque` from each physics substep using the applied motor
current and wheel speeds. Store its diagnostics, latch fault state, accumulate
time and squared-current integral above 65 A, and pass only its four wheel torques
to the existing tire/wheel ODE. Update reset paths in both `sim_warp.py` and the
environment kernel so all new state starts at zero.

Keep `sensors.py` reading normalized `applied_effort`; update its nearby comment
only if necessary to state that the value is derived from physical current.

- [ ] **Step 8: Add the readback and pass integration tests**

Implement `read_drivetrain_state()` as a zero-copy-style dictionary of the backing
PyTorch tensors, matching existing read methods. Run:

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py training/tests/test_warp_env.py -vv
```

Expected: both test files pass.

- [ ] **Step 9: Commit simulator integration**

```bash
git add training/f1tenth_sim/dynamics.py training/f1tenth_sim/sim_warp.py \
  training/f1tenth_env/kernel.py training/f1tenth_env/sensors.py \
  training/tests/warp_probe.py training/tests/test_warp_sim_unit.py \
  training/tests/test_warp_env.py
git commit -m "sim: integrate physical current drivetrain state"
```

---

### Task 4: Configure Courtyard and preserve artifact reproducibility

**Files:**
- Modify: `training/config.py`
- Modify: `training/configs/courtyard_2_e2e_ppo.json`
- Modify: `training/tests/test_sensor_config.py`
- Modify: `libs/f1tenth_policy/f1tenth_policy/artifact.py`
- Modify: `libs/f1tenth_policy/test/test_layout_actor.py`
- Modify: `training/standalone_trainer.py`
- Modify: `training/tests/test_trainer_core.py`

**Interfaces:**
- Produces: artifact key `sim_drivetrain` containing the reproducibility-only electrical parameter dictionary.
- Preserves: `longitudinal_mode="force"`, action dimension, current scaling fields, and all deployment validation behavior.

- [ ] **Step 1: Write failing configuration tests**

Require the Courtyard config after merge to contain:

```python
assert env["i_drive_max_a"] == 80.0
assert env["i_brake_max_a"] == 20.0
assert env["i_battery_max_a"] == 60.0
assert env["i_battery_regen_max_a"] == 10.0
assert env["i_abs_max_a"] == 120.0
assert env["i_slew_a_per_s"] == 200.0
assert env["vehicle_mass"] == 3.444
assert low <= 3.444 <= high
```

Also require the old force and power keys to be absent from the Courtyard config,
and require any retained `drive_scale_range` to be exactly `[1.0, 1.0]` until the
sensitivity gate demonstrates a justified efficiency range.

- [ ] **Step 2: Run config tests and verify RED**

```bash
.venv/bin/python -m pytest training/tests/test_sensor_config.py \
  -k 'courtyard or electrical or vehicle_mass' -vv
```

Expected: failures on the current 65 A and legacy fixed-force configuration.

- [ ] **Step 3: Update defaults and Courtyard configuration**

Set the selected physical values, remove the Courtyard `f_drive_max`,
`f_brake_max`, and `power_max` keys, and replace the invalid 3.6--3.9 kg mass range
with a deterministic `[3.444, 3.444]` interval. Do not invent a mass range.

Keep the existing LiDAR, occupancy-map, sensor-noise, opponent, and PPO settings
unchanged.

- [ ] **Step 4: Run config tests and verify GREEN**

Run Task 4 Step 2. Expected: selected tests pass.

- [ ] **Step 5: Write failing artifact metadata tests**

Require `build_sensor_artifact_payload(..., sim_drivetrain=metadata)` to preserve
this dictionary:

```python
{
    "model": "vesc_reduced_electrical_v1",
    "motor_kv_rpm_per_v": 3500.0,
    "motor_pole_pairs": 2,
    "gear_ratio": pytest.approx(11.8225),
    "wheel_radius_m": pytest.approx(0.05255),
    "battery_voltage_v": 11.34,
    "i_motor_max_a": 80.0,
    "i_motor_brake_max_a": 20.0,
    "i_battery_max_a": 60.0,
    "i_battery_regen_max_a": 10.0,
    "i_abs_max_a": 120.0,
}
```

Validation must reject a present non-dictionary `sim_drivetrain`, but must accept
older artifacts where the optional key is absent.

- [ ] **Step 6: Run artifact tests and verify RED**

```bash
.venv/bin/python -m pytest libs/f1tenth_policy/test/test_layout_actor.py \
  training/tests/test_trainer_core.py -k 'sim_drivetrain or artifact' -vv
```

Expected: failures because the builder does not accept the metadata.

- [ ] **Step 7: Serialize optional simulator metadata**

Add a keyword-only optional `sim_drivetrain: Mapping[str, Any] | None` argument to
the artifact builder, copy it into the payload when supplied, and validate only
its container type in the shared loader. Build the exact dictionary at the
`save_policy_artifact` call from merged `env` configuration.

Do not change the artifact format version or deployment action validation because
the metadata is optional, reproducibility-only, and does not change policy tensor
or action semantics.

- [ ] **Step 8: Run configuration and artifact tests**

```bash
.venv/bin/python -m pytest training/tests/test_sensor_config.py \
  training/tests/test_trainer_core.py libs/f1tenth_policy/test/test_layout_actor.py -vv
```

Expected: all three test files pass.

- [ ] **Step 9: Commit configuration and metadata**

```bash
git add training/config.py training/configs/courtyard_2_e2e_ppo.json \
  training/tests/test_sensor_config.py training/standalone_trainer.py \
  training/tests/test_trainer_core.py \
  libs/f1tenth_policy/f1tenth_policy/artifact.py \
  libs/f1tenth_policy/test/test_layout_actor.py
git commit -m "training: select Courtyard electrical drivetrain"
```

---

### Task 5: Prove sensitivity and remove unjustified randomization

**Files:**
- Modify: `training/tests/test_warp_sim_unit.py`
- Modify: `training/configs/courtyard_2_e2e_ppo.json`

**Interfaces:**
- Consumes: `electrical_drive_torque` probe and final Courtyard speed/reset range.
- Produces: executable sensitivity assertions and the final episode-level parameter ranges.

- [ ] **Step 1: Write a deterministic operating-envelope sweep test**

Evaluate commands `[-1.0, -0.5, 0.0, 0.5, 1.0]` and vehicle speeds from 0 through
20.5 m/s for the Cartesian product:

```text
motor resistance: 0.005, 0.010, 0.050, 0.100 ohm
drivetrain efficiency: 0.60, 0.80, 1.00
battery voltage: 11.1, 11.34, 12.6 V
```

The resistance sweep stays below the zero-speed feasibility ceiling
`0.95 * 11.1 V / 80 A = 0.1318 ohm`; the efficiency sweep spans a deliberately
lossy drivetrain through the lossless mathematical ceiling. These are sensitivity
bounds, not claims about the installed car.

Assert every sample is finite and inside all current, voltage, fault, speed, and
power invariants. Record which constraints activate through test assertion
messages so a failure identifies the exact parameter tuple.

- [ ] **Step 2: Run the sweep and verify RED if a model invariant is incomplete**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k theory_grounded_sensitivity_sweep -vv
```

Expected: the new test initially fails until all missing edge handling is present.

- [ ] **Step 3: Fix only invariant failures in the reduced model**

Change equations only where a named limiting case fails. Do not widen ranges or
relax assertions to hide a failure. Re-run the sweep after each correction.

- [ ] **Step 4: Select episode-level ranges by materiality**

For each assumed parameter, calculate maximum relative wheel-torque change over
the sweep against the nominal tuple. A parameter is material if it changes wheel
torque by at least 2% anywhere in the configured operating envelope. Put only
material parameters into episode-level domain randomization; leave immaterial
parameters fixed. Store the resulting bounds directly in the Courtyard JSON.

- [ ] **Step 5: Run the sweep and config tests**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  -k 'theory_grounded_sensitivity_sweep or drivetrain' -vv
.venv/bin/python -m pytest training/tests/test_sensor_config.py -vv
```

Expected: all tests pass and no per-step electrical randomness exists.

- [ ] **Step 6: Commit the justified uncertainty envelope**

```bash
git add training/tests/test_warp_sim_unit.py \
  training/configs/courtyard_2_e2e_ppo.json training/f1tenth_sim/drivetrain.py
git commit -m "sim: verify drivetrain sensitivity envelope"
```

---

### Task 6: Full verification, GPU smoke test, and training launch

**Files:**
- No source files expected.
- Runtime outputs: ignored training output/checkpoint directories only.

**Interfaces:**
- Consumes: completed simulator and `training/configs/courtyard_2_e2e_ppo.json`.
- Produces: verification evidence, a short GPU smoke artifact, and a new Courtyard training process.

- [ ] **Step 1: Run focused drivetrain and configuration tests**

```bash
.venv/bin/python -m pytest training/tests/test_warp_sim_unit.py \
  training/tests/test_warp_env.py training/tests/test_sensor_config.py \
  training/tests/test_trainer_core.py libs/f1tenth_policy/test/test_layout_actor.py -vv
```

Expected: zero failures.

- [ ] **Step 2: Run the full pure-Python training suite and lint**

```bash
.venv/bin/python -m pytest training/tests
.venv/bin/python -m flake8 training
```

Expected: zero failures and zero lint diagnostics.

- [ ] **Step 3: Run repository build, test, and lint gates**

```bash
./tools/build.sh
./tools/test.sh
```

Expected: successful build, unit tests, and ament lint. Run ROS/colcon work only
inside the dev container as orchestrated by these scripts.

- [ ] **Step 4: Inspect the final diff and configuration lineage**

```bash
git diff --check
git status --short
git diff develop...HEAD -- training/f1tenth_sim training/f1tenth_env \
  training/config.py training/configs/courtyard_2_e2e_ppo.json \
  libs/f1tenth_policy
```

Confirm no rosbag, weight, map output, build tree, unrelated user file, vendored
vehicle code, observation layout, or action layout entered the changes.

- [ ] **Step 5: Run a new-lineage GPU smoke test**

First verify GPU availability:

```bash
nvidia-smi
```

Then launch a short run with no checkpoint initialization and a unique run ID:

```bash
cd training
../.venv/bin/python standalone_trainer.py \
  --config configs/courtyard_2_e2e_ppo.json \
  --algorithm ppo --num-envs 8 --total-transitions 1024 \
  --device cuda --seed 55 --run-id courtyard-electrical-v1-smoke \
  --no-wandb --no-compile
```

Expected: initialization succeeds, simulator diagnostics stay finite, at least one
optimization completes, and a policy artifact contains `sim_drivetrain`.

- [ ] **Step 6: Inspect the smoke artifact and logs**

Load the saved payload with `.venv/bin/python` and assert the model name and all
five current settings. Inspect logs for NaNs, faults, impossible current, or a
single constraint dominating every sample. Any such observation blocks launch.

- [ ] **Step 7: Launch the full Courtyard run**

Use the repository's established detached-session pattern with a unique run ID,
the approved Courtyard configuration, no initialization checkpoint, CUDA, 4096
environments, two billion transitions, and seed 55:

```bash
tmux new-session -d -s courtyard-electrical-v1 \
  'cd /home/ubuntu/projects/F1tenth/training && \
   ../.venv/bin/python standalone_trainer.py \
     --config configs/courtyard_2_e2e_ppo.json \
     --algorithm ppo --num-envs 4096 --total-transitions 2000000000 \
     --device cuda --seed 55 \
     --run-id courtyard-electrical-v1-ppo-s55-a001'
```

Record the exact command, tmux session, output directory, seed, and first healthy
progress line in the handoff.

- [ ] **Step 8: Report verification and live-run evidence**

Report exact test counts, lint/build outcomes, smoke output path, full-run session
identifier, full-run output path, seed, active current settings, and whether the
first training interval is progressing without numerical or controller faults.

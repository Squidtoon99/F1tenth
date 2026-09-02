# Theory-Grounded Drivetrain Design

**Date:** 2026-09-01

**Research:** [`docs/research/2026-09-01-slash-4x4-vehicle-model.md`](../../research/2026-09-01-slash-4x4-vehicle-model.md)

## Goal

Replace the simulator's independent fixed-force and fixed-power drivetrain limits
with one deterministic, energy-consistent reduced electrical drivetrain for the
race-ready Traxxas Slash 4X4. The model must reproduce the configured VESC motor,
battery, and absolute-current semantics and make every non-specified physical
parameter explicit.

The Courtyard training run must not start until analytical unit tests, simulator
integration tests, and repository verification pass.

## Physical configuration

The nominal simulated vehicle uses:

- race-ready mass: 3.444 kg;
- Velineon 3500 motor: 3500 mechanical rpm/V, four poles, 50,000 rpm maximum;
- stock reduction: `(54 / 13) * (37 / 13) = 11.8225`;
- stock tire circumference: 0.3302 m, giving an unloaded radius of approximately
  0.05255 m;
- 3S LiPo: 11.1 V nominal and 12.6 V maximum;
- motor-current limits: +80 A drive and -20 A brake;
- battery-current limits: +60 A discharge and -10 A regeneration;
- absolute-current fault threshold: 120 A; and
- current command slew: 200 A/s.

The installed battery part number and installed tooth counts are not confirmed.
The implementation will therefore describe these values as the selected vehicle
configuration, not as observations of the installed hardware.

## Model boundary

The new drivetrain owns the conversion from normalized longitudinal effort to
per-wheel axle torque. The existing wheel-spin, tire-slip, combined-slip,
suspension/load-transfer, and rigid-body integration remain responsible for
turning axle torque into vehicle motion.

The drivetrain will not apply a whole-car `mu * mass * gravity` force clamp. That
duplicates the tire model and suppresses wheel-slip behavior. It will also remove
the separate 320 W cap because configured battery current and voltage replace its
electrical purpose.

No new observation or action fields are introduced. The policy continues to emit
normalized longitudinal effort and steering. This avoids changing the shared
observation/action contract.

## Parameter provenance

Parameters are divided conceptually into three groups:

1. `specified`: manufacturer or VESC firmware values;
2. `derived`: deterministic calculations from specified values; and
3. `assumed`: physical quantities not supplied by the component documentation.

The implementation will add only parameters consumed by the reduced model. It
will not add a generic provenance framework because there is currently only one
consumer. Provenance and units will be recorded beside the configuration values
and checked by focused tests.

Assumed quantities must be constant within an episode. If randomized, they are
sampled once at reset over documented physical bounds. The electrical equations
must never receive per-step white noise.

## Electrical and mechanical data flow

### 1. Commanded current

For normalized effort `u` in `[-1, 1]`, requested q-axis current is

```text
Iq_requested = 80 * u       when u >= 0
Iq_requested = 20 * u       when u < 0
```

The simulator applies a physical slew limit of 200 A/s to current, rather than a
single normalized slew rate. This preserves the asymmetric drive and brake ranges.

### 2. Motor and wheel speed

Mechanical motor speed follows driven-wheel speed:

```text
omega_motor = gear_ratio * omega_wheel
mechanical_rpm = omega_motor * 60 / (2*pi)
erpm = mechanical_rpm * pole_pairs
```

For this four-pole motor, `pole_pairs = 2`.

### 3. Motor voltage and modulation

The reduced steady-state motor equation is

```text
Vq_required = motor_resistance * Iq + Ke * omega_motor
modulation = clamp(Vq_required / battery_terminal_voltage, -modulation_max,
                   modulation_max)
```

where

```text
Ke = 60 / (2*pi*Kv)
```

Motor resistance is an explicit assumed parameter until a manufacturer value or
motor-detection result is available. Inductive current-loop dynamics, field
weakening, inverter dead time, and FOC switching ripple are outside this reduced
model. The existing 200 A/s command slew represents the actuator bandwidth at the
policy time scale.

If the requested current requires voltage beyond the modulation limit, allowed
current is reduced to the steady-state voltage-feasible value. The result cannot
increase magnitude or reverse sign relative to the request.

### 4. VESC current limits

The ideal reduced inverter relationship is

```text
I_battery = modulation * Iq
```

with sign-safe handling around zero modulation. The current applied to the motor
is the intersection of:

```text
-20 A <= Iq <= 80 A
-10 A <= modulation * Iq <= 60 A
voltage-feasible current interval
```

This ordering represents the VESC firmware's q-axis motor-current and DC input-
current constraints without simulating the FOC control loop itself.

An estimated or measured absolute motor current at or above 120 A produces a
controller fault state. It does not smoothly scale torque. Nominal commands cannot
reach that state because the configured motor-current limit is 80 A.

### 5. Torque

The ideal reciprocal torque constant is

```text
Kt = 60 / (2*pi*Kv) = 0.002728 N*m/A
```

The reduced model computes

```text
motor_torque = Kt * Iq_allowed
axle_torque = motor_torque * gear_ratio * drivetrain_efficiency
```

Drivetrain efficiency is explicit and assumed. Axle torque is divided equally
between front and rear differentials according to the existing four-wheel-drive
split and then equally between the two wheels on each axle.

The lossless theoretical envelope at radius 0.05255 m is 0.614 N/A: 49.1 N and
14.3 m/s^2 at +80 A, and 12.3 N and 3.56 m/s^2 at -20 A. These are analytical
upper bounds, not claimed realized accelerations.

### 6. Battery voltage

The first implementation uses a configured terminal voltage that is constant
within an episode and bounded by the documented 11.1 V nominal and 12.6 V maximum
3S range. It does not equate 90% state of charge with 90% of maximum voltage.

Battery capacity, state-of-charge integration, open-circuit-voltage curves,
internal resistance, and thermal behavior are excluded because the installed
battery identity and the required curves are not established. The design leaves
voltage as an episode-level operating condition so a later battery model can
replace it without changing drivetrain equations.

## Assumption policy

The nominal configuration must give every assumed quantity a named value and a
short physical basis. The initial reduced model needs only:

- motor phase resistance;
- drivetrain efficiency;
- terminal battery voltage; and
- maximum modulation.

Existing tire, rolling-resistance, aerodynamic, inertia, and geometry parameters
remain in their existing owning modules. They are not duplicated in the
drivetrain.

Sensitivity sweeps must show the effect of motor resistance, drivetrain
efficiency, and voltage across the Courtyard operating speed range. A parameter
whose sweep does not materially affect torque or acceleration in that range will
not be randomized for this training run.

The 65 A continuous and 100 A burst motor ratings are validation envelopes. They
do not alter instantaneous torque without thermal capacity, thermal resistance,
cooling, initial temperature, and burst-duration data. The simulator will expose
time and RMS current above 65 A as diagnostics but will not invent thermal
derating.

## Configuration changes

The Courtyard configuration will be cold-started with the new model and these
settings:

```text
mass = 3.444 kg
motor current max = 80 A
motor brake current max = -20 A
battery current max = 60 A
battery regen current max = -10 A
absolute current max = 120 A
current slew = 200 A/s
```

The mass domain-randomization interval must contain 3.444 kg and represent only
known configuration variation. The prior 3.6--3.9 kg range is invalid because it
excludes the measured race-ready vehicle.

The old `f_drive_max`, `f_brake_max`, and `power_max` configuration values will no
longer control the Warp drivetrain. Compatibility handling will be confined to
loading older artifacts where required; new training artifacts record the
electrical model parameters needed to reproduce their dynamics.

## Diagnostics and error handling

Each simulation environment records the active longitudinal constraint as one of:

- command/motor current;
- battery discharge current;
- battery regeneration current;
- voltage/modulation;
- absolute-current fault; or
- none.

Configuration construction rejects non-positive mass, radius, gearing, voltage,
efficiency, slew, or current magnitudes; efficiency above one; a nominal voltage
outside the declared battery bounds; motor limits above the absolute-current
threshold; and inconsistent positive/negative limit signs.

An absolute-current fault produces zero commanded axle torque for the rest of the
episode and a diagnostic flag. It must not silently continue with clipped torque.

## Verification

Implementation follows red-green-refactor. Tests use the real reduced drivetrain
and Warp simulator; they do not mock physical dependencies.

Analytical tests must prove:

- the 11.8225 reduction and RPM/ERPM conversions;
- `Kt = 60/(2*pi*3500)` and the lossless 0.614 N/A force envelope;
- asymmetric +80/-20 A current mapping and 200 A/s slew;
- +60/-10 A input-current limiting at positive and regenerative modulation;
- voltage feasibility and no-load speed at 11.1 V and 12.6 V;
- 120 A creates a fault rather than a torque clamp;
- zero command, zero speed, low modulation, and sign transitions are finite;
- electrical input power covers mechanical output plus declared losses within
  numerical tolerance;
- increasing resistance or loss cannot increase available torque;
- wheel torque sums to the computed axle torque; and
- the tire model, rather than the drivetrain, limits traction and produces slip.

Integration tests must verify that the simulator steps deterministically, remains
finite under randomized episode parameters, reports active constraints, and no
longer uses the independent fixed-force or 320 W caps.

Before launch, run the focused training tests, all training tests, training lint,
the repository build/test/lint commands required by `AGENTS.md`, and a short GPU
smoke training run. The full Courtyard run starts from a new run ID and checkpoint
lineage only after all gates pass.

## Out of scope

- Changing the observation/action contract.
- Editing vendored vehicle code.
- A switching-level inverter or full FOC controller simulation.
- Battery SOC, voltage sag, or thermal dynamics without identified parameters.
- Motor or VESC thermal derating without thermal parameters.
- Recalibrating the tire model from rosbags.
- Treating arbitrary noise as physical uncertainty.

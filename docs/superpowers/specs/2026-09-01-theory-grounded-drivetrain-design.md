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

For motoring, the reduced model computes

```text
motor_torque = Kt * Iq_allowed
axle_torque = motor_torque * gear_ratio * drivetrain_efficiency
```

Regeneration uses the inverse efficiency relationship defined in the implementation
formula reference so mechanical input power at the wheels remains greater in
magnitude than recovered electrical power.

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

## Implementation formula reference

This section fixes the conventions and numerical edge behavior used by the
reduced model. Implementations must preserve these relationships even if symbols
are renamed to match surrounding code.

### Symbols and signs

Positive current, torque, wheel angular velocity, and vehicle longitudinal speed
mean forward motoring. Negative current means regenerative braking while the
vehicle is moving forward. Current magnitudes in configuration are stored as
positive numbers; their direction is introduced by the command.

| Symbol | Meaning | Unit |
| --- | --- | --- |
| `u` | normalized longitudinal policy action | 1 |
| `I_cmd` | requested signed q-axis current | A |
| `I_q` | allowed signed q-axis current | A |
| `I_dc` | signed battery/DC-link current | A |
| `V_dc` | battery terminal voltage at the inverter | V |
| `m_q` | signed q-axis modulation ratio | 1 |
| `omega_w` | mechanical wheel angular speed | rad/s |
| `omega_m` | mechanical motor angular speed | rad/s |
| `G` | motor-to-wheel reduction | 1 |
| `K_v` | mechanical no-load motor speed constant | rpm/V |
| `K_e` | ideal back-EMF constant | V s/rad |
| `K_t` | ideal reciprocal torque constant | N m/A |
| `R_s` | reduced effective motor phase resistance | ohm |
| `eta_d` | mechanical drivetrain efficiency | 1 |

The four-pole motor has two pole pairs, so

```text
mechanical_rpm = omega_m * 60 / (2*pi)
electrical_rpm = 2 * mechanical_rpm
```

### Action mapping and current slew

The asymmetric action map is

```text
I_target(u) = u * I_motor_max          if u >= 0
I_target(u) = u * I_motor_brake_max    if u < 0
```

For control interval `dt_c`, slew physical current rather than normalized effort:

```text
Delta_I_max = I_slew * dt_c
I_next = I_prev + clamp(I_target - I_prev, -Delta_I_max, Delta_I_max)
I_interval = 0.5 * (I_prev + I_next)
```

When the target is reached partway through the interval, integrate the linear ramp
and constant-current remainder exactly:

```text
t_ramp = abs(I_target - I_prev) / I_slew
I_interval = (
    0.5 * (I_prev + I_target) * t_ramp
    + I_target * (dt_c - t_ramp)
) / dt_c
```

The observation-facing directional fraction remains

```text
effort = I_interval / I_motor_max          if I_interval >= 0
effort = I_interval / I_motor_brake_max    if I_interval < 0
```

This keeps it in `[-1, 1]` without changing the observation layout.

### Speed constants and gearing

Convert the published mechanical speed constant to SI:

```text
Kv_SI = Kv * 2*pi / 60
Ke = 1 / Kv_SI = 60 / (2*pi*Kv)
Kt = Ke
```

`Kt = Ke` is the ideal reciprocal relation under consistent voltage and current
definitions. It is an explicit reduced-model assumption because the VESC's native
FOC torque equation uses detected flux linkage.

For four driven wheels, use the mean signed wheel speed for the center-drivetrain
shaft speed:

```text
omega_w_mean = 0.25 * sum(omega_w[i], i=0..3)
omega_m = G * omega_w_mean
```

Using signed rather than absolute wheel speed preserves reverse and regenerative
power direction. Individual wheel-speed differences remain handled by the
differentials and tire model; the reduced model does not simulate differential
internal dynamics.

### Voltage feasibility

With zero d-axis current and steady-state current at the policy time scale:

```text
V_emf = Ke * omega_m
V_required(Iq) = V_emf + R_s * Iq
V_available = modulation_max * V_dc
```

Allowed current must satisfy

```text
-V_available <= V_emf + R_s * Iq <= V_available
```

For positive `R_s`, the exact voltage-feasible interval is

```text
I_voltage_min = (-V_available - V_emf) / R_s
I_voltage_max = ( V_available - V_emf) / R_s
```

Intersect the current request with that interval. This form handles forward,
reverse, motoring, and regeneration without separate sign-specific formulas. The
configured model rejects non-positive `R_s`, so division by zero is not permitted.

After choosing a candidate current, calculate signed modulation as

```text
m_q = clamp((V_emf + R_s * Iq) / V_dc,
            -modulation_max, modulation_max)
```

At no load and zero current, the modulation-limited mechanical speed is

```text
omega_motor_no_load = modulation_max * V_dc / Ke
v_no_load = omega_motor_no_load * r / G
```

The published 11.1 V and 12.6 V speed figures use the lossless analytical ceiling
`modulation_max = 1`. A configured value below one must reduce those speeds by the
same factor.

### Motor and battery current intersection

First clip commanded current to configured motor limits:

```text
I_motor_min = -I_motor_brake_max
I_motor_max = +I_motor_max
I_candidate = clamp(I_interval, I_motor_min, I_motor_max)
```

The ideal reduced inverter relationship is power-consistent:

```text
I_dc = m_q * I_q
P_dc = V_dc * I_dc
P_motor_electrical = (m_q * V_dc) * I_q
P_dc = P_motor_electrical
```

For fixed nonzero modulation magnitude, battery-current limits imply

```text
I_battery_min_q = -I_battery_regen_max / abs(m_q)
I_battery_max_q = +I_battery_max / abs(m_q)
```

The zero-modulation limit is the entire real current line because `I_dc = 0` in
the ideal algebra. Use a small positive denominator only for numerical evaluation;
do not make that epsilon create an artificial current limit.

Because modulation depends on current through `R_s * I_q`, the final current must
satisfy both relationships simultaneously. Solve the one-dimensional monotone
constraint deterministically with a fixed-count bisection over the interval formed
by motor and voltage limits:

```text
f(I) = abs(m_q(I) * I)
```

For positive requested current, choose the largest current between zero and the
candidate whose `f(I) <= I_battery_max`. For negative requested current, choose the
most negative current between the candidate and zero whose
`f(I) <= I_battery_regen_max`. A fixed iteration count gives identical CPU/GPU
control flow and bounded error. Recompute modulation and `I_dc` from the solved
current rather than retaining values calculated before the intersection.

The result must satisfy, within float32 tolerance:

```text
-I_motor_brake_max <= I_q <= I_motor_max
-I_battery_regen_max <= I_dc <= I_battery_max
abs(m_q) <= modulation_max
sign(I_q) is either sign(I_interval) or zero
abs(I_q) <= abs(I_interval)
```

### Torque and four-wheel distribution

Motor and delivered center-drivetrain torque are

```text
T_motor = Kt * I_q
T_center = T_motor * G * eta_d
```

Let `gamma` be the existing front torque fraction. Wheel torques are

```text
T_LR = T_RR = 0.5 * (1 - gamma) * T_center
T_LF = T_RF = 0.5 * gamma * T_center
```

Therefore

```text
sum(T_wheel) = T_center
```

for both positive and negative current. The drivetrain does not change torque as
a function of mass or tire friction. The existing tire equations decide how much
of this axle torque becomes longitudinal force and how much becomes wheel slip.

For a lossless, no-slip analytical check:

```text
F_x = T_center / r
a_x = F_x / m
```

At `eta_d = 1`, the chosen values produce approximately 0.614 N/A, 49.1 N and
14.3 m/s^2 at 80 A, and 12.3 N and 3.56 m/s^2 at 20 A.

### Energy inequalities

The reduced motor assigns copper loss

```text
P_copper = R_s * I_q^2
P_airgap = V_emf * I_q
P_motor_electrical = P_copper + P_airgap
P_shaft_ideal = T_motor * omega_m
```

Since `Ke = Kt`, `P_airgap = P_shaft_ideal`. Mechanical delivery is

```text
P_wheel = eta_d * P_shaft_ideal
```

For motoring, validation requires

```text
P_dc >= P_shaft_ideal >= P_wheel >= 0
```

For regeneration, compare magnitudes and directions:

```text
P_shaft_ideal <= 0
P_dc <= 0
abs(P_dc) <= abs(P_shaft_ideal)
```

The simple fixed efficiency multiplier above is defined for motoring torque. In
regeneration, energy consistency requires inverse loss mapping rather than
multiplying generated shaft power by a number below one in the wrong direction.
The reduced model will use:

```text
T_center = T_motor * G * eta_d              for T_motor*omega_m >= 0
T_center = T_motor * G / eta_d              for T_motor*omega_m < 0
```

so a smaller-magnitude electrical braking torque is required to absorb a given
wheel-side regenerative power. At zero speed, select the motoring branch and rely
on continuity; power is zero in either case.

### Constraint reporting

Constraint selection is based on which operation first changes requested current:

```text
motor current -> voltage/modulation -> battery discharge or battery regen
```

If multiple limits are active within float32 tolerance, report the later physical
bottleneck in that sequence because it determines the final current. Constraint
codes are stable internal simulator diagnostics, not policy observations.

### Absolute-current fault and ratings

The absolute-current check compares externally supplied or internally calculated
measured-current magnitude against 120 A before nominal soft limiting:

```text
fault_now = abs(I_measured) >= I_absolute_max
fault_latched_next = fault_latched_previous or fault_now
```

A latched fault commands zero wheel torque until environment reset. The 80 A
normal limit cannot trigger it by itself. Tests may inject measured current to
exercise the fault path; the production reduced model must not invent overshoot.

For diagnostic accumulation over physics interval `dt`:

```text
time_above_continuous += dt if abs(I_q) > 65 A else 0
current_squared_integral += I_q^2 * dt
elapsed_current_time += dt
I_rms = sqrt(current_squared_integral / elapsed_current_time)
```

The 100 A burst rating is checked as an invariant on non-fault modeled current.
Neither rating creates torque derating without a thermal model.

### Required limiting cases

The implementation and tests must cover:

- `u = 0`: zero requested current and torque;
- `omega_m = 0`: zero back-EMF and finite resistive modulation;
- `m_q = 0`: zero ideal battery current without division artifacts;
- forward speed plus negative current: regeneration and negative DC power;
- reverse speed plus positive braking direction: sign-consistent opposing torque;
- back-EMF above available voltage: no sign reversal from voltage clipping;
- `eta_d = 1`: equality with the analytical lossless torque envelope;
- equal wheel speeds: exact center-to-wheel torque sum;
- unequal wheel speeds: finite mean-shaft approximation;
- current exactly at 80, -20, 60, -10, and 120 A boundaries;
- reset after a fault: cleared latch and zeroed diagnostics; and
- all supported cases: finite float32 outputs on CPU and CUDA.

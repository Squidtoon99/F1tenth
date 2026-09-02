# Theory-grounded Traxxas Slash 4X4 longitudinal model

**Date:** 2026-09-01  
**Scope:** Primary manufacturer documentation, VESC firmware source, and analytical
derivations. No rosbag measurements are used. No simulator code or configuration
was changed.

---

## 1. Question and conclusion

Can the battery, controller, motor, gearing, tire, and measured vehicle-mass
specifications determine a robust longitudinal simulator without real-world data?

**They determine a useful ideal drivetrain envelope, but they do not determine a
unique high-fidelity vehicle model.** The specifications establish voltage, speed,
current, gearing, ideal torque, and ideal energy relationships. They do not publish
the motor parameters needed to calculate loaded voltage, the drivetrain losses,
the battery's loaded voltage, or the tire/surface forces that bound realized
acceleration. Those omissions are structural, not numerical inconvenience: many
physically different cars satisfy the same published specifications.

The sound implementation is therefore two-layered:

1. an analytically exact **electrical and kinematic envelope**, with VESC-equivalent
   current-limit ordering; and
2. a **vehicle-force model** whose unavailable parameters are explicit priors or
   bounded variables, never hidden constants presented as measured facts.

This is still a major improvement over fixed force and power caps. It makes every
limit dimensionally traceable and prevents arbitrary noise. It cannot, by itself,
prove that simulated acceleration matches the physical car.

## 2. Configuration identity

The user identified the platform as a Traxxas Slash 4X4 with its stock Velineon
3500 motor and stock 3S battery, driven by a TRAMPA VESC 6 MK6, with a race-ready
mass of 3.444 kg. “Stock” is model-year dependent, so the installed part numbers
and tooth counts remain pre-run inspection items.

Traxxas documents the Slash 4X4 VXL powertrain as a Velineon 3500 with a 13-tooth
pinion and 54-tooth spur. It specifies the motor as sensorless, 3500 rpm/V,
four-pole, with a 50,000 rpm maximum. The same manual calls for a 3S, 11.1 V,
5000 mAh-or-larger LiPo for the stock 13/54 high-speed setup.

Sources:

- [Traxxas Slash 4X4 manual, model 68077-4](https://traxxas.com/sites/default/files/68077-4-OM-EN-R07.pdf)
- [Traxxas Slash 4X4 VXL product sheet](https://traxxas.com/media/productattach/C-68286-4/1/kd2552-r00-68286-4-slash-4x4-vxl-fox-box.pdf)

Traxxas's current Slash 4X4 parts documentation identifies the front and rear
differentials as 13/37. This is the secondary reduction after the 13/54
pinion/spur pair.

Source: [Traxxas Slash 4X4 VXL parts list](https://traxxas.com/media/productattach/C-68386-4/3/68386-4_parts.pdf).

Traxxas identifies battery 2872X as an 11.1 V, 5000 mAh, 3-cell, 25C LiPo and
uses that part to produce full 3S performance in an official Slash 4X4 build.
The exact installed battery part number has not been confirmed, so 2872X is a
candidate, not a fact about this car.

Sources:

- [Traxxas battery application guide](https://traxxas.com/sites/default/files/150406-R00-Battery-Counter-Mat-Chart.pdf)
- [Traxxas Slash 4X4 three-phase build](https://traxxas.com/news/slash-4x4-3-phase-upgrade)
- [Traxxas LiPo safety and discharge instructions](https://traxxas.com/media/productattach/2970-3S/10/GKC20006-R00-LiPo-battery-fold-out-large%20-%20EN.pdf)

TRAMPA specifies the VESC 6 MKVI at 11.1--60 V, 80 A continuous, and 120 A peak,
with separate configurable protection for motor current, input current, motor
regeneration, and input regeneration. Hardware ratings depend on cooling and are
not the same thing as the user's configured software limits.

Source: [TRAMPA VESC 6 MKVI technical data](https://trampaboards.com/vesc-6-mkvi-in-cnc-t6-silicone-sealed-aluminium-box-with-genuine-xt90-connectors--vedder-electronic-speed-controller-trampa-special-p-27536.html?download=248).

## 3. Parameter ledger

| Parameter | Value | Status | Basis and consequence |
| --- | ---: | --- | --- |
| Race-ready mass, \(m\) | 3.444 kg | Measured input | User-provided; includes the race vehicle rather than factory curb mass. |
| Motor speed constant, \(K_v\) | 3500 rpm/V | Specified | Traxxas Velineon 3500 manual. It is a no-load speed constant. |
| Motor poles | 4 | Specified | Traxxas manual; two pole pairs. Required to distinguish mechanical RPM from VESC ERPM. |
| Motor maximum speed | 50,000 rpm | Specified | Traxxas manual; a mechanical safety envelope, not the 3S operating speed. |
| Pinion/spur | 13/54 | Specified stock | Traxxas manual. Must be confirmed on the installed car. |
| Differential pinion/ring | 13/37 | Specified stock | Traxxas parts list. Must be confirmed on the installed car. |
| Total reduction, \(G\) | 11.8225:1 | Derived | \((54/13)(37/13)\). |
| Stock tire circumference, \(C\) | about 13 in = 0.3302 m | Specified approximately | Traxxas describes a Slash tire as rolling about 13 inches per revolution; tire growth and loaded radius are not specified. [Traxxas tire comparison](https://traxxas.com/news/monster-slash-conversion). |
| Unloaded effective radius, \(r=C/(2\pi)\) | about 0.05255 m | Derived approximate | Suitable for kinematic gearing calculations, not a precise loaded rolling radius. |
| Battery chemistry | 3S LiPo | Specified platform | Traxxas manual. |
| Nominal battery voltage | 11.1 V | Specified | Three-cell nominal pack voltage. |
| Maximum charged voltage | 12.6 V | Specified | Traxxas states no more than 4.2 V/cell. |
| Candidate capacity | 5 Ah | Candidate spec | Battery 2872X; installed part number unconfirmed. |
| Candidate discharge rating | 25C = 125 A | Candidate spec/derived | \(5\text{ Ah}\times25\text{ h}^{-1}\). This is above the proposed 60 A software input limit. |
| VESC hardware envelope | 80 A continuous, 120 A peak | Specified hardware | TRAMPA rating; temperature and cooling dependent. |
| Configured motor limits | +80 A, -20 A | User setting | Software torque-current clamps, not motor hardware ratings. |
| Configured battery limits | +60 A, -10 A | User setting | Software DC-link current clamps, not battery capability. |
| Configured absolute current | 120 A | User setting | Fault threshold, not a normal torque clamp. |
| Motor continuous/burst references | 65 A / 100 A | User-provided rating | Thermal/survival envelope. A time or temperature model cannot be derived from two current numbers. |
| Motor phase resistance, \(R_s\) | unavailable | Missing | Required for copper loss, voltage demand, loaded speed, and motor heating. VESC normally detects it. |
| Motor inductance, \(L_d,L_q\) | unavailable | Missing | Required for current-loop transients and high-speed voltage demand. VESC normally detects it. |
| Rotor flux linkage, \(\lambda\) | unavailable | Missing | Required to use the VESC's native FOC torque equation without a current-convention assumption. VESC normally detects it. |
| No-load current | unavailable | Missing | Needed to separate motor/core/friction loss from useful shaft torque. |
| Controller efficiency/loss map | unavailable | Missing | Needed for DC power, heating, and loaded battery current beyond the ideal inverter relation. |
| Battery OCV-versus-SOC curve | unavailable | Missing | Nominal and maximum voltage do not define voltage at 90% state of charge. |
| Battery internal resistance | unavailable | Missing | Required for voltage sag and peak power at the DC link. |
| Drivetrain efficiency/loss map | unavailable | Missing | Bearings, gear meshes, differentials, and slipper clutch reduce wheel torque. |
| Tire longitudinal friction/slip curve | unavailable | Missing | The dominant bound on launch and braking acceleration. |
| Loaded/dynamic tire radius | unavailable | Missing | Pneumatic deformation and tire growth alter force and speed conversion. |
| Wheel/drivetrain rotational inertia | unavailable | Missing | Adds effective mass during acceleration and stores energy during braking. |
| Rolling resistance | unavailable | Missing | Required for coast-down and terminal speed. |
| Aerodynamic \(C_dA\) | unavailable | Missing | Required for terminal speed and high-speed energy use. |
| CG height and axle distribution | unavailable | Missing | Required for load transfer and per-axle traction under acceleration/braking. |
| Thermal capacities/resistances | unavailable | Missing | Required to turn 65 A continuous and 100 A burst ratings into time-dependent derating. |

## 4. Exact analytical envelope

### 4.1 Gearing and wheel kinematics

For motor mechanical speed \(\omega_m\), wheel speed and ideal vehicle speed are

\[
\omega_w=\frac{\omega_m}{G}, \qquad v=r\omega_w.
\]

The total stock reduction is

\[
G=\frac{54}{13}\frac{37}{13}=11.8225.
\]

Ignoring voltage drop and all load, the motor's published \(K_v\) gives

\[
n_m=K_vV, \qquad
v_0(V)=\frac{K_vV}{G}\frac{2\pi r}{60}.
\]

Using the approximate stock rolling radius:

| Pack voltage | Ideal no-load motor speed | Ideal no-load vehicle speed |
| ---: | ---: | ---: |
| 11.1 V nominal | 38,850 rpm | 18.1 m/s (40.4 mph) |
| 11.34 V, if “90%-full voltage” literally means \(0.9\times12.6\) | 39,690 rpm | 18.5 m/s (41.3 mph) |
| 12.6 V fully charged | 44,100 rpm | 20.5 m/s (45.9 mph) |

These are kinematic upper bounds, not predicted terminal speeds. They are below
the motor's specified 50,000 rpm maximum. Ninety-percent state of charge is not
equivalent to 90% of full-pack voltage; calculating the former requires the
missing OCV/SOC curve.

VESC reports electrical RPM. Its firmware states that reported RPM must be
divided by half the motor pole count to obtain mechanical RPM. For this four-pole
motor, \(\mathrm{ERPM}=2n_m\).

Source: [VESC FOC firmware RPM documentation](https://github.com/vedderb/bldc/blob/master/motor/mcpwm_foc.c#L1057-L1066).

### 4.2 Ideal torque constant and its limitation

Convert the published speed constant to SI:

\[
K_{v,SI}=3500\frac{2\pi}{60}=366.52\ \mathrm{rad\,s^{-1}V^{-1}}.
\]

For an ideal reciprocal electromechanical conversion using mutually consistent
voltage and current definitions,

\[
K_t=\frac{1}{K_{v,SI}}=\frac{60}{2\pi K_v}
=0.002728\ \mathrm{N\,m/A}.
\]

This is an **ideal motor-current torque constant**, not a complete proof that one
VESC-reported ampere produces exactly that shaft torque. BLDC \(K_v\) can be
quoted using line-to-line/back-EMF conventions while FOC software uses transformed
q-axis current and flux linkage. VESC's own torque calculation is

\[
T_m=\frac{3}{2}p\lambda I_q,
\]

where \(p\) is the pole-pair count and \(\lambda\) is detected motor flux linkage.
Its official torque-measurement package implements that equation. Without the
Velineon's flux linkage or a manufacturer statement defining its \(K_v\) and
current conventions, \(60/(2\pi K_v)\) is the ideal envelope conversion, not a
validated VESC-current calibration.

Source: [VESC official torque-measurement package](https://github.com/vedderb/vesc_pkg/blob/main/lib_nau7802/examples/measure_torque.lisp).

### 4.3 Ideal wheel force and acceleration

With ideal gears and unloaded tire radius,

\[
T_w=G K_t I_q,
\qquad
F_{wheel}=\frac{G K_t I_q}{r},
\qquad
a_{ideal}=\frac{F_{wheel}}{m}.
\]

The ideal force constant is

\[
\frac{F_{wheel}}{I_q}
=\frac{(11.8225)(0.002728)}{0.05255}
=0.614\ \mathrm{N/A}.
\]

Therefore the configured current limits imply these lossless upper bounds:

| Mode | Motor current | Ideal motor torque | Ideal wheel force | Ideal acceleration magnitude |
| --- | ---: | ---: | ---: | ---: |
| Drive | 80 A | 0.218 N m | 49.1 N | 14.3 m/s² (1.45 g) |
| Motor brake | 20 A | 0.0546 N m | 12.3 N | 3.56 m/s² (0.36 g) |

Real drive force must instead satisfy

\[
F_x=\min\left(
\frac{G\eta_d T_m}{r_d},\ F_{tire}(\kappa,F_z,\mu,\ldots)
\right)-F_{rr}-F_{aero},
\]

and acceleration must include rotational inertia:

\[
a=\frac{F_x}{m+m_{eq}}, \qquad
m_{eq}=\sum_j J_j\left(\frac{\omega_j/v}{1}\right)^2.
\]

None of \(\eta_d\), the tire law, \(F_{rr}\), \(F_{aero}\), or \(m_{eq}\) is
fixed by the component specifications. The ideal 1.45 g result is itself evidence
that current and mass alone cannot predict launch acceleration: the tire contact
patch must decide whether that force reaches the floor.

### 4.4 Motor voltage, back-EMF, and current feasibility

The ideal back-EMF constant is reciprocal to the SI speed constant:

\[
K_e=1/K_{v,SI}=0.002728\ \mathrm{V\,s/rad}.
\]

A reduced steady-state motor model requires

\[
V_q=R_sI_q+K_e\omega_m
\]

before cross-coupling, inductive dynamics, field weakening, inverter dead time,
and modulation details. The requested current is feasible only if the inverter
can produce the required voltage. A dynamic dq model additionally requires

\[
V_d=R_sI_d+L_d\dot I_d-\omega_eL_qI_q,
\]

\[
V_q=R_sI_q+L_q\dot I_q+\omega_e(L_dI_d+\lambda).
\]

Traxxas does not publish \(R_s\), \(L_d\), \(L_q\), or \(\lambda\). VESC firmware
contains configuration fields and detection routines for precisely these values,
which confirms they are inputs rather than consequences of current limits.

Sources:

- [VESC motor-configuration fields](https://github.com/vedderb/bldc/blob/master/datatypes.h)
- [VESC motor resistance/inductance/flux detection](https://github.com/vedderb/bldc/blob/master/conf_general.c)

Consequently, component sheet data cannot calculate the loaded duty cycle, the
speed at which 80 A becomes voltage-limited, copper loss \(3I_{rms}^2R_s\), or an
exact torque-speed curve. An invented resistance would merely move uncertainty
into an undocumented constant.

## 5. VESC-equivalent current limits

The firmware distinguishes four runtime limits:

- `l_current_max` and `l_current_min`: motor/q-axis current limits;
- `l_in_current_max` and `l_in_current_min`: DC input/battery current limits; and
- `l_abs_current_max`: an absolute measured-current fault threshold.

Source: [VESC `mc_configuration`](https://github.com/vedderb/bldc/blob/master/datatypes.h).

For FOC, the current loop first constrains q-axis current using the input-current
limits divided by filtered q-axis modulation, then applies motor-current limits.
In the positive-modulation case, the essential firmware algebra is

\[
I_q\in
\left[
\frac{I_{in,min}}{m_q},
\frac{I_{in,max}}{m_q}
\right]
\quad\text{and}\quad
I_q\in[I_{motor,min},I_{motor,max}].
\]

The firmware separately computes bus current from the modulation and alpha-beta
currents when the hardware has no input-current sensor. This is the authoritative
basis for a reduced inverter relationship; `duty × motor current` is only an
approximation because displayed duty and q-axis modulation are not identical.

Source: [VESC FOC current-limiter and bus-current implementation](https://github.com/vedderb/bldc/blob/master/motor/mcpwm_foc.c#L3368-L3398).

For a zero-d-axis reduced model, a faithful idealization is

\[
I_{in}\approx m_q I_q,
\]

\[
I_q^*=\operatorname{clip}(I_{command},-20,80),
\]

\[
I_q=\operatorname{clip}\left(
I_q^*,
\frac{-10}{|m_q|},
\frac{60}{|m_q|}
\right),
\]

with sign handling matching torque and power direction and a safe branch as
\(|m_q|\rightarrow0\). The exact firmware also includes temperature, voltage,
wattage, RPM, duty, BMS, and recursively filtered input-current overrides; these
should not be invented if their configuration is unknown.

Source: [VESC runtime override-limit implementation](https://github.com/vedderb/bldc/blob/master/motor/mc_interface.c#L2334-L2390).

The 120 A absolute maximum is different. Firmware compares absolute measured or
filtered motor current against `l_abs_current_max` and calls
`mc_interface_fault_stop(FAULT_CODE_ABS_OVER_CURRENT, ...)`. It is a fault, not
a soft force clamp.

Source: [VESC absolute-overcurrent fault implementation](https://github.com/vedderb/bldc/blob/master/motor/mc_interface.c#L1876-L1885).

Thus the requested settings must have these simulator semantics:

| Setting | Correct nominal behavior |
| --- | --- |
| Motor Current Max +80 A | Clamp requested torque-producing current. |
| Motor Current Max Brake -20 A | Clamp requested regenerative/braking motor current. |
| Battery Current Max +60 A | Reduce allowed motor current as modulation and input power increase. |
| Battery Current Max Regen -10 A | Reduce regenerative motor current as modulation and generated DC current increase. |
| Absolute Maximum Current 120 A | Trigger a controller fault if measured absolute current crosses it; do not smoothly scale force. |

The motor's 65 A continuous and 100 A burst ratings are a separate thermal and
survival envelope. They do not replace the configured +80 A current clamp. Without
thermal resistance, thermal capacity, cooling, initial temperature, and a defined
burst duration, they cannot produce a mathematically determined derating curve.

## 6. Battery power and energy

For the candidate 2872X battery,

\[
E_{nom}=V_{nom}Q=(11.1)(5)=55.5\ \mathrm{Wh}.
\]

At the configured 60 A input-current limit,

\[
P_{DC}=VI_{in}
\]

is 666 W at 11.1 V, 680 W at 11.34 V, and 756 W at 12.6 V. A lossless 5 Ah pack
held at 60 A would reach its nominal amp-hour capacity in

\[
t=Q/I=5/60\ \mathrm{h}=5\ \mathrm{min}.
\]

These are correct bookkeeping results, not a voltage trajectory. A battery state
model requires at minimum

\[
\dot{SOC}=-\frac{I_{in}}{Q},
\qquad
V_{terminal}=V_{OC}(SOC,T)-I_{in}R_{int}(SOC,T),
\]

plus charge/regen efficiency if energy recovery matters. The candidate spec gives
capacity and C-rating but not \(V_{OC}(SOC,T)\) or \(R_{int}(SOC,T)\). A fixed
11.34 V operating point is a declared scenario assumption; it is not a derived
90%-SOC battery voltage.

The 25C candidate rating implies 125 A sustained capability, while the user's
known safe physical envelope is 80 A. Both exceed the configured 60 A software
limit, so battery hardware capability should not bind nominal simulation. This
does not eliminate voltage sag.

## 7. What the specification-only model can validate

The following properties can be validated exactly with algebraic and invariant
tests:

1. Gear ratio and all RPM/ERPM/wheel-speed conversions.
2. Dimensional consistency and lossless upper bounds for torque, force, speed,
   power, energy, and acceleration.
3. Motor-current command clipping at +80/-20 A.
4. Battery-current clipping at +60/-10 A using the same modulation relationship
   and limit ordering as VESC firmware.
5. Separation of normal current limiting from the 120 A absolute-current fault.
6. Conservation checks: electrical input power equals motor electrical power plus
   declared losses; mechanical wheel power cannot exceed motor shaft power.
7. Monotonicity: increasing loss, resistance, drag, or slip cannot improve
   acceleration or terminal speed.
8. Limiting cases: zero speed, zero current, no-loss, open-circuit, full
   modulation, and regenerative operation.

These tests prove the implementation matches the chosen equations. They do not
prove that unavailable parameter values match the car.

## 8. Why specifications alone cannot validate sim-to-real acceleration

An example demonstrates non-identifiability. Two hypothetical Slash 4X4 vehicles
can have identical battery voltage, current limits, \(K_v\), gearing, tire size,
and mass, while one runs on a high-grip floor with efficient gears and the other
runs on dust with a slipping clutch. Both satisfy every published component
specification; their acceleration and braking can differ by multiples. No
algebraic manipulation of the shared specifications selects the real trajectory.

The same applies electrically. Motors with the same no-load \(K_v\) can have
different winding resistance and no-load current, producing different loaded
speed and heat. Batteries with the same cell count, capacity, and C-rating can
have different internal resistance and voltage sag. A controller current rating
does not provide its loss map.

Therefore a claim that these specifications are “all of the information needed”
would not be mathematically defensible. They are sufficient for:

- a traceable ideal drivetrain;
- controller protection-limit behavior;
- hard physical upper bounds; and
- principled centers and constraints for uncertain parameters.

They are insufficient for:

- realized longitudinal or lateral acceleration;
- stopping distance;
- terminal speed under load;
- wheel slip and combined tire force;
- voltage sag over a lap;
- current/torque transient response;
- thermal derating or safe burst duration; and
- a demonstrated sim-to-real error bound.

## 9. Recommended simulator boundary

A specification-grounded implementation should introduce only behavior supported
by the ledger:

1. Replace the fixed 320 W cap with the configured VESC motor/input-current limit
   equations; retaining both would double-limit the same electrical envelope.
2. Use 3.444 kg as the vehicle's nominal race-ready mass.
3. Represent the stock gearing and tire circumference explicitly, with a startup
   assertion that their configured product matches the intended reduction.
4. Use a motor dq or reduced steady-state electrical model only after assigning
   explicit provenance to resistance and flux linkage. Until then, treat the
   \(K_v\)-derived result as an ideal upper envelope.
5. Keep drivetrain efficiency, loaded radius, rotating inertia, tire parameters,
   rolling resistance, drag, battery resistance, and thermal parameters in a
   named `assumed` group. Do not inject white noise into the equations.
6. Randomize assumed physical parameters once per episode, not once per time
   step, over documented physically plausible bounds. This represents a family of
   possible cars rather than sensor noise.
7. Expose which constraint is active—motor current, battery current, voltage,
   traction, or fault—so training and validation can detect accidental dominance.
8. Validate theory separately from reality: analytical tests establish equation
   correctness; real measurements, when allowed, establish parameter accuracy.

This boundary produces a simulator grounded in theory without overstating what
the spec sheets establish. Robust sim-to-real training should cover uncertainty
in the missing physical quantities, but coverage is not the same as adding
unstructured noise.


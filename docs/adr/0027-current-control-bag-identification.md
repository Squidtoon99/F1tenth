# 0027 — Catalog 80/40 current-control bag identification

- Status: Proposed (identification snapshot; Warp command replay not yet run)
- Date: 2026-09-08

## Context

The on-car current-control bags from 2026-09-06 (`sep_6_current_rosbags.zip`, ROS 2
sqlite3, unpacked at `/tmp/sep_6_current_rosbags`) plus a scale mass of **3.444 kg**
are the first closed-loop evidence at an **80 A drive / 40 A brake** envelope.
Commands are joystick `/teleop`, not `/rl/actuator`. Accel bags map
`teleop.acceleration ∈ [-1, 1]` as `+1 → 80 A`, `−1 → 40 A`. The skidpad bag is a
different contract: **80 A drive / 20 A brake**. Warp still uses
`VehicleParams` defaults in [`training/f1tenth_sim/params.py`](../../training/f1tenth_sim/params.py)
(`mass=3.74`, `f_drive_max=23.0`, `f_brake_max=5.2`, `tire_mu=0.65`) and
[`training/config.py`](../../training/config.py) still randomizes mass in
`[3.6, 3.9]` with `i_brake_max_a=20`. Chassis geometry that bags cannot measure
(wheelbase, tire OD → radius, stock curb mass, gearing) is taken from the Traxxas
Slash 4X4 VXL spec sheet and cataloged below; it does not retune Warp. A Warp
replay of the recorded commands is the planned verifier; this ADR records the
bag inference so that replay does not overwrite it.

## Decision

1. Treat the bag-inferred dynamics in the table below as the **current
   identification**, not as a Warp retune.
   *Why:* the numbers are odom `F = m a` at current plateaus with a measured mass;
   they are not closed-loop sim residuals.
2. Prefer **odom** for longitudinal force and acceleration. Report IMU as a
   second, conflicting estimate. Do not mix skidpad 20 A brake into the 40 A
   brake force.
   *Why:* IMU long-axis is `y` with opposite sign and poor correlation; skidpad
   brake saturates at 20 A.
3. Leave Warp parameters unchanged until a command replay fills the **Verified**
   column (add a subsection here, or supersede with a follow-on ADR if the
   retune is a separate decision).
   *Why:* identification and model change are distinct; do not rewrite this
   snapshot.
4. Use Traxxas Slash 4X4 VXL listed wheelbase and tire OD for this snapshot's
   geometry/torque catalog. Keep **3.444 kg** as the bag vehicle. Do not
   overwrite Warp or `vesc.yaml`.
   *Why:* odom+steer wheelbase is circular (`vesc_to_odom` already uses 0.325 m);
   bags do not measure rolling radius; stock curb mass is chassis-only.

## Dynamics (measured vs Warp vs verified)

Plateau forces use odom `a_x = dv_x/dt` with `dt ∈ [8, 40] ms`, `|a_x| < 25 m/s²`,
and a 5-sample speed moving average, at `m = 3.444 kg`. N/A is median `a_x / I`
on the 80 A / 40 A plateaus (`accel_4`; LS-through-current discarded, r² ≈ 0).
Best bags: `accel_4` then `accel_1` for longitudinal; skidpad for yaw. Skip empty
`accel_multiple_bag`. Treat `accel_5` as dirty (VESC overcurrent, 90 A measured).

| Quantity | Bag (this snapshot) | Warp now | Verified |
| --- | --- | --- | --- |
| Mass | **3.444 kg** (scale) | 3.74 kg; DR `[3.6, 3.9]` | **3.444 kg** |
| `i_drive_max_a` | **80 A** (accel + skidpad) | `config.py` 80 A | **80 A** |
| `i_brake_max_a` | **40 A** (accel); **20 A** (skidpad) | `config.py` **20 A** | **40 A** (skidpad 20 A is decode-only) |
| Current slew p95 | ≈ **200 A/s** | gate 200 A/s | 200 A/s |
| `longitudinal_slew_rate_per_s` | **2.5** (`200/80`) | `params.py` 4.444 (45 A-era default); `config.py` `warp_sim` **2.5** | **2.5** |
| Drive N/A | **0.23 N/A** (IQR 0.22–0.25) | 23/80 = 0.29 N/A | 26.5/80 = 0.331 N/A |
| Brake N/A | **0.50 N/A** (IQR 0.43–0.56) | 5.2/20 = 0.26 N/A | 23.1/40 = 0.578 N/A |
| `f_drive_max` @ 80 A | **18.6 N** (IQR 17.3–20.1) | **23.0 N** (~24% high) | **26.5 N** |
| `f_brake_max` @ 40 A | **20.0 N** (IQR 17.2–22.5) | **5.2 N** (~4× low; 20 A-era force) | **23.1 N** |
| `a_x` @ 80 A (odom median) | **+5.4 m/s²** (IQR ~5.0–5.8) | 23/3.74 ≈ 6.15 m/s² | holdout drive `a_x` err **0.027** (green) |
| `a_x` @ 40 A brake (odom median) | **−5.8 m/s²** (IQR ~5.0–6.5) | 5.2/3.74 ≈ 1.39 m/s² | holdout brake `a_x` err **0.032** (green) |
| IMU long. (same plateaus) | ~**16 N** drive / ~**23 N** brake (`−a_y`) | — | not used |
| `tire_mu` | realized \|a\|/g **0.73** median / **0.95** p95 (skidpad, ~29 A) | **0.65** (DR 0.50–1.13) | **0.71** |
| `max_steer` | **±0.33 rad** | 0.33 rad | 0.33 rad |
| Servo map | gain ≈ **−1.21**, offset ≈ **0.4495** | `vesc.yaml` −1.2135 / 0.4495 | unchanged |
| `wheelbase` | **0.324 m** (Traxxas listed; not from odom) | **0.325 m** (`params.py`, `vesc.yaml`; +1 mm) | replay overlay **0.324 m** |
| `wheel_radius` | listed D/2 **0.0546 m**; loaded/rolling **0.0526 m** | **0.053 m** | **0.053 m** |
| Axle torque @ plateau (`F·R`, R=0.0546 m) | drive **1.02 N·m**, brake **1.09 N·m** | 1.22 / 0.28 N·m (Warp F × 0.053 m) | 1.45 / 1.26 N·m (26.5 / 23.1 N × 0.0546 m) |

Extractable from these bags: current limits, slew, servo map, IMU units (accel
in g with `az≈1`, gyro in deg/s), and the odom forces above. Wheelbase and tire
radius are **not** in bags; they come from the Traxxas spec sheet in the next
section. Still missing: motor Kt / VESC XML, aero, CG, `/rl` latency.

## Spec-sheet physicals

This repo names a Traxxas Slash 4X4 / Slash 4X4 VXL (`training/tests/test_vehicle_geometry.py`
`car_length`/`car_width`, `calibration/drivetrain_priors.py`). `docs/context.md`
does not name an SKU. Catalog **Slash 4X4 VXL**, current production **68386-4**
(VXL HD). Geometry matches **68286-4** (clipless VXL) and **68086-4**. Ultimate
(**68077-4** / **68277-4**) shares 324 mm wheelbase and 296 mm listed track but
lists a different curb mass and 47 mm vs 72 mm ground clearance (low-CG). The
exact race-car SKU is not in these bags; VXL vs Ultimate is the remaining
ambiguity and does not change wheelbase or tire OD.

Traxxas listed (RTR, battery typically sold separately):

- Wheelbase **12.75 in (324 mm)**. Front/rear track **11.65 in (296 mm)**, same
  as overall width (outside-to-outside, not hub centerline). Length **568 mm**,
  height **193 mm**, ground clearance **72 mm**. Curb mass **5.82 lb (2.64 kg)**.
- Internal ratio **11.82:1**. Stock pinion/spur **13/54**; diffs **13/37**
  (`(54/13)×(37/13) = 11.8225`). Motor **Velineon 3500**, 3500 rpm/V, 4-pole,
  50,000 rpm max, 262 g.
- Stock SCT tire **4.3 in × 1.7 in** (part **6871**; dual-profile 2.2/3.0 in
  wheels). Listed diameter/2 = **0.0546 m**. Traxxas quotes a Slash SCT tire as
  rolling **~13 in/rev** → effective rolling radius **0.0526 m** (~96% of
  geometric; typical loaded foam-insert radius). `vesc.yaml`
  `speed_to_erpm_gain` **4300** matches 4-pole × 11.82:1 at the **rolling**
  radius (4297), not listed D/2 (4135).
- Top speed **60+ mph** with optional 19/54 and 3S; stock 13/54 + 3S is **45+
  mph**. Ignore as dynamics ID.

Sources:
[68386-4 product](https://traxxas.com/slash-4x4-vxl-68386-4),
[68286-4 product](https://traxxas.com/68286-4-110-slash-4x4-vxl-brushless-short-course-truck-w-tqi),
[68386-4 manual](https://traxxas.com/media/productattach/C-68386-4/2/68386-4-OM-EN-R00.pdf),
[68286-4 box sheet](https://traxxas.com/media/productattach/C-68286-4/1/kd2552-r00-68286-4-slash-4x4-vxl-fox-box.pdf),
[68386-4 parts](https://traxxas.com/media/productattach/C-68386-4/3/68386-4_parts.pdf),
[6871 tires](https://traxxas.com/6871-bfg-mud-terrain-ta-km2-off-road-tires-2),
[~13 in/rev](https://traxxas.com/news/monster-slash-conversion) (2WD Slash SCT
family; not 4X4-specific).

Until Warp replay, use the **Use** column. Warp/`vesc.yaml` stay as they are.

| Quantity | Traxxas listed | Warp / vesc now | Bag-inferred | Use until replay |
| --- | --- | --- | --- | --- |
| SKU | Slash 4X4 VXL 68386-4 (geom. = 68286-4) | unnamed | — | VXL geometry |
| Wheelbase | **0.324 m** (324 mm) | **0.325 m** | circular (odom uses 0.325 m) | **spec 0.324 m** |
| Track (F/R) | **0.296 m** overall | **0.253 m** centerline (`0.296 − 1.7"`) | — | Warp 0.253 m (centerline) |
| Length / width | 0.568 / 0.296 m | `config.py` 0.568 / 0.296 m | — | already matched |
| Tire OD | **4.3 in** (109.2 mm) | — | — | spec |
| Rolling radius | listed D/2 **0.0546 m**; loaded **0.0526 m** | **0.053 m** | not in bags | **listed D/2 for τ**; 0.0526 m for ERPM/kinematics |
| Mass | **2.64 kg** stock RTR | 3.74 kg; DR `[3.6, 3.9]` | **3.444 kg** (scale) | **3.444 kg** (do not overwrite) |
| Gearing | 13/54, diffs 13/37, **11.82:1** | not in Warp force model | not counted | spec prior; unverified on car |
| Motor | Velineon 3500, 4-pole, 3500 kV | — | not in bags | spec prior; unverified |
| `speed_to_erpm_gain` | 4297 at R=0.0526 m | **4300** | odom uses 4300 | vesc 4300 (matches rolling R) |
| Top speed | 45+ / 60+ mph claims | `max_speed` 15 m/s | — | ignore |
| Axle τ @ 18.6 / 20.0 N | **1.02 / 1.09 N·m** (R=0.0546 m); 0.98 / 1.05 at R=0.0526 m | 1.22 / 0.28 N·m (23 N / 5.2 N × 0.053 m) | force yes; R from spec | spec R × bag F |
| Slip angles | — | — | not measured | still none |

## What we can and cannot identify

**Acceleration — yes.** Filtered odom `dv/dt` and IMU both exist. Prefer odom
for longitudinal. At the 80 A / 40 A plateaus on `accel_4`, odom median is
**+5.4 m/s²** drive and **−5.8 m/s²** brake. IMU long-axis appears to be **y**,
opposite sign versus odom, with poor correlation; the same plateaus give roughly
**−4.6 / +6.6 m/s²** (`−a_y`). Naive unfiltered `dv/dt` spikes to 100–1000 m/s²
and is discarded.

**Force — yes, longitudinal only.** `F = m a` with the measured 3.444 kg yields
**0.23 N/A → 18.6 N @ 80 A** and **0.50 N/A → 20.0 N @ 40 A**. This is not
ground-truth wheel force if the tires slip: `/odom.vx = ERPM/4300`, `vy = 0`, so
wheel slip is invisible. IMU would give ~16 N drive / ~23 N brake; those numbers
are not the identification.

**Torque — axle only, under spec radius.** Bags do not measure rolling radius or
gearing. Using Traxxas listed tire OD/2 **R = 0.0546 m**, `τ_axle = F · R` is
**1.02 N·m** drive and **1.09 N·m** brake. Warp `R=0.053 m` on the same forces
is 0.99 / 1.06 N·m; loaded/rolling R = 0.0526 m is 0.98 / 1.05 N·m. Motor-shaft
torque is **not identified**: no VESC XML, flux, or counted pinion/spur/diff in
these recordings. The repo's Velineon `Kt ≈ 0.00273 N·m/A` and Traxxas 11.82:1
remain unverified assumptions, not bag fits.

**Slip angles — no, as a measurement.** `odom.vy = 0`, there is no OptiTrack, and
particle-filter pose is not ground truth. A bicycle residual (yaw vs
`v tanδ / L` with L = 0.325 m) shows an understeer-like slope **0.42–0.66**, but
that mixes steer lag, the servo map, tire force, and IMU mount — it is not α.
Replacing L with spec 0.324 m is a 0.3% change and still not α. A later bound
can be inferred from Warp vs bag yaw after replay; not from these bags alone.

## Caveats

- `teleop.acceleration` is a current fraction, not m/s²; `teleop.speed` is unused.
- Drive and brake current live on separate topics; a gap in `/commands/motor/current`
  is often braking, not coast.
- Skidpad μ is realized lateral accel at partial throttle (~29 A), not a proven
  friction ceiling.
- Kinematic wheelbase from odom+steer is circular (`vesc_to_odom` already uses
  0.325 m). Independent wheelbase is Traxxas **0.324 m**.
- Traxxas listed track 296 mm is overall width; Warp `track_width` 0.253 m is
  centerline (listed width minus 1.7 in tire width).
- IMU x/y is not body-forward / body-left on these mounts.

## Consequences

- Do not mix skidpad 20 A brake force into an 80/40 `f_brake_max`.
- Warp is **not** retuned by this ADR. Drive force in sim is ~24% high versus
  odom identification; brake force is still a 20 A-era 5.2 N against a 40 A,
  ~20 N plant. Wheelbase is 1 mm long vs Traxxas 0.324 m; `wheel_radius` 0.053 m
  sits between listed D/2 (0.0546 m, +3%) and loaded rolling (0.0526 m, −1%).
- Next work is a Warp replay of the accel/skidpad command traces. Fill the
  **Verified** column (or supersede) with residuals; do not rewrite the bag
  numbers in this snapshot.
- Domain-randomization mass `[3.6, 3.9]` sits entirely above the scale reading.
  Stock Slash curb mass 2.64 kg is chassis-only, not a replacement for 3.444 kg.

## Verified

Warp IPEM command replay of `/tmp/sep_6_current_rosbags` (diagnosis rounds 0–5)
stopped at round 5. Bag numbers in this snapshot are unchanged.

- **ACCEPT** `tire_friction` **0.71**. Train/holdout drive and brake plateau
  `a_x` are all green; holdout drive **0.027**. μ 0.70 also helped; 0.71 is
  strictly better.
- **REJECT** `t_delta` 0.07 and 0.13. Leave **0.1**. Do not NLS Pacejka B/C/E.
- Champion overlay: mass **3.444 kg**, **80 A / 40 A**, `f_drive_max` **26.5 N**,
  `f_brake_max` **23.1 N**, slew **2.5**, wheelbase **0.324 m**. Production
  envelope is 80/40 only; skidpad 80/20 is bag decode (`20/40` teleop scale).
- Leftover red (stop): train TTS **0.160** vs 0.15 (one 0.02 s bin) and skidpad
  yaw RMSE ~**0.52** (`odom.vy` is zero — unobservable).
- Plant promotion into training is [0028](0028-promote-80-40-current-plant.md).

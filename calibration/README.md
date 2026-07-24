# calibration/

Sim-to-real and per-car physics calibration. Tooling and procedures for measuring
the parameters that make the simulator match the real car and each chassis behave
consistently. Longitudinal actuation is **current/brake only** (ADR 0006).

## Drivetrain priors (no car required)

Installed motor prior: **Traxxas Velineon 3500** (sensorless 4-pole, 3500 Kv,
ideal `Kt ≈ 0.00273 N·m/A`). ESC prior: **TRAMPA VESC 6 MKV** (80 A continuous /
120 A peak — controller limits only). Stock Slash 4X4 VXL gearing prior is
13/54 pinion/spur × 13/37 diffs (~11.82:1); **count the installed gears** before
promoting values.

Seed wheel force:

```text
F = I_phase × Kt × gear_ratio × efficiency / wheel_radius
```

```bash
.venv/bin/python calibration/export_drivetrain_priors.py --i-drive-max 10 --i-brake-max 10
```

Treat published 65 A continuous / 100 A burst motor figures as upper-bound
references, not operating setpoints. Safe envelope = min(motor, VESC thermal,
battery/BMS, wiring, drivetrain).

## Offline fitting (Part 1 — no car required)

Dependencies (training `.venv` already has `rosbags` if you used the analysis
scripts; otherwise install):

```bash
.venv/bin/pip install -r calibration/requirements.txt
```

Fit the three downloaded bags (paths are off-repo by design):

```bash
.venv/bin/python calibration/fit_vehicle_params.py \
  --bags ~/f1tenth_calib_bags/static_imu_205106 \
         ~/f1tenth_calib_bags/steering_205300 \
         ~/f1tenth_calib_bags/accel_205714 \
  --out calibration/results/car01_sim.yaml \
  --report calibration/results/car01_report.md \
  --json calibration/results/car01_fits.json
```

Unit tests (synthetic data, no bags):

```bash
.venv/bin/python -m pytest calibration/test_fit_vehicle_params.py \
  calibration/test_drivetrain_priors.py -q
```

Outputs under `calibration/results/` are gitignored. Raw rosbags stay off-repo.

### Identifiable from current bags

- IMU unit/sign/bias (`static_imu_*`)
- ERPM↔speed gain, odom sign/scale (`accel_*` / `steering_*`)
- Servo gain/offset, weak max-steer and steer lag (`steering_*`)
- Acceleration limit, equivalent `f_drive_max`, weak `c_roll`, μ lower bound

### Not identifiable yet (needs Part 2 bags)

- Active `f_brake_max`, open-loop current→force gain, `power_max`
- Suspension stiffness/damping / anti-roll

## Part 2 — boxed-wheel current/brake (car required)

1. Export live VESC motor/application config + FOC detection (current limits,
   battery/regen limits, flux linkage, R/L, pole count, ERPM, voltage/temp).
2. Confirm battery cell count/voltage and installed gearing.
3. Bring up the car with wheels off the ground. Record while stepping current:

```bash
ros2 bag record -o ~/f1tenth_calib_bags/boxed_current_$(date +%H%M%S) \
  /commands/motor/current /commands/motor/brake /sensors/core /odom \
  /sensors/imu/raw /ackermann_cmd &
python3 calibration/record_boxed_current.py --i-max 8 --steps 5 --hold-s 1.5
```

4. Re-run `fit_vehicle_params.py` with that bag included —
   `fit_current_to_force` / `fit_brake_current` activate when those topics are
   present. Reject fits with poor excitation or sign inconsistency.

Force/current actuation: [`docs/adr/0006-current-only-vesc-actuation.md`](../docs/adr/0006-current-only-vesc-actuation.md).

Outputs feed:

- the simulator model used by [`training/`](../training/)
- the per-car overlay [`deploy/cars/<carNN>/params.yaml`](../deploy/cars/)

## IMU bias/noise characterization (sensor policy)

Quantify per-axis bias, noise, drift, sample rate, and VESC current/temperature
correlation from sensor-policy calibration bags. Outputs are small JSON/Markdown
summaries under `calibration/results/` (gitignored); raw rosbags stay off-repo.

### On-car recording (later hardware step)

Run with the sensor-policy stack in **recording mode** (`recording_mode:=true` on
the policy node). The runtime publishes converted SI means and actor IMU values
without changing the control path:

| Topic | Content |
| --- | --- |
| `/sensors/imu/raw` | Driver-frame IMU (required) |
| `/sensor_policy/imu_raw_record` | Converted SI mean per policy interval |
| `/sensor_policy/imu_actor_record` | Actor indices `[1081:1087]` after preprocess |
| `/sensors/core` | VESC temp/current (optional; needed for correlation) |
| `/scan`, `/odom` | Context for dynamic runs (optional for static analysis) |

Capture three conditions before accepting bias/sign changes:

1. **static_off** — car level, electronics/motor off where possible (~60 s).
2. **static_on** — full VESC/LiDAR/compute powered, zero drive current (~60 s).
3. **dynamic** — boxed-wheel, low-speed, or live-track segments with motor load.

Example bag record (adjust launch/remaps to your on-car workflow):

```bash
ros2 bag record -o ~/f1tenth_calib_bags/static_imu_on_$(date +%H%M%S) \
  /sensors/imu/raw /sensor_policy/imu_raw_record /sensor_policy/imu_actor_record \
  /sensors/core /scan /odom
```

### Offline analysis

```bash
.venv/bin/python calibration/characterize_imu.py \
  --bag ~/f1tenth_calib_bags/static_imu_on_HHMMSS \
  --condition static_on \
  --cal-yaml src/racing_rl/f1tenth_rl_agent/config/sensor_policy.yaml \
  --out-json calibration/results/imu_static_on.json \
  --out-report calibration/results/imu_static_on.md
```

Flags:

- `--condition {static_off,static_on,dynamic,unknown}` — tags the summary.
- `--cal-yaml` — optional IMU params (`sensor_policy.yaml` or a multi-node car
  overlay with `imu_*` keys); defaults to static-fit inference from the bag.
- `--require-core` — exit non-zero when `/sensors/core` is missing (dynamic
  runs where current/temp correlation is required).

The tool fails clearly when `/sensors/imu/raw` is absent. It replays raw IMU
through the deploy preprocessor and compares against recorded actor values when
present. Review `proposed_sensor_policy_params` in the JSON before editing car
params; re-run the same bag after any change to confirm replay parity.
Proposed `imu_az_bias` stays `0` so gravity is not nulled in converted SI.

Each summary includes per-axis mean/std/outlier counts, sample rate, short-term
drift slope, 10 Hz actor replay stats, and (when `/sensors/core` is present)
Pearson correlations of rolling noise vs `|current_motor|` and VESC FET/motor
temperature.

Unit tests (synthetic data, no bags):

```bash
.venv/bin/python -m pytest calibration/test_imu_characterization.py -q
```

### Remaining physical recording steps

1. Launch the sensor-policy stack with `recording_mode:=true` (no actuator /
   remapped outputs for the first passes).
2. Record **static_off** (~60 s, car level) and **static_on** (~60 s, stack
   powered, zero current).
3. Run `characterize_imu.py` on each bag; compare powered vs unpowered noise and
   review proposed ax/ay/gyro biases.
4. During boxed-wheel / low-speed / live-track tests, keep recording raw + actor
   IMU with `/sensors/core`; re-run with `--condition dynamic --require-core`.
5. Accept param changes only after replaying the same bag through preprocessing;
   do not tune filters mid-lap. Feed measured bias/noise into a later retraining
   follow-up if `az/gx/gy` should stop being frozen.

## Sensor-policy four-condition parity capture

This capture is a future physical step. Keep motor actuation disabled: use a
mechanical stand/chock for stationary captures and hand-push the car for moving
captures. Do not infer or fabricate these measurements while the car is
unavailable.

Collect exactly four bags with unchanged sensor-policy parameters and checkpoint:

1. `race_corridor_stationary`: bounded race corridor, level and still, 60 seconds.
2. `race_corridor_hand_push`: same placement, three straight hand-pushed passes at
   approximately 0.5–1.0 m/s.
3. `hallway_open_stationary`: hallway with one open side, level and still, 60 seconds.
4. `hallway_open_hand_push`: same placement, three straight hand-pushed passes at
   approximately 0.5–1.0 m/s.

Enable `recording_mode` and `publish_raw_observation` on `sensor_racer`, verify
that `/commands/motor/current` remains zero and `/rl/actuator/applied` never
reports `SOURCE_RL`, then record each condition:

```bash
CONDITION=race_corridor_stationary
ros2 bag record -o ~/f1tenth_calib_bags/sensor_parity_${CONDITION}_$(date -u +%Y%m%dT%H%M%SZ) \
  /scan /sensors/imu/raw /sensor_racer/imu_raw_record \
  /sensor_racer/imu_actor_record /sensor_racer/observation \
  /sensor_racer/diagnostics /odom /sensors/core \
  /rl/actuator/desired /rl/actuator/applied \
  /commands/motor/current /commands/motor/brake \
  /commands/servo/position
```

Repeat with each condition name. Before accepting a bag, check its duration,
topic counts, zero motor current, source ownership, and observation/action
replay with `training/analysis/sensor_policy_bag.py`. Record corridor dimensions,
open-side orientation, checkpoint SHA-256, parameter-file SHA-256, and whether
the car was stationary or hand-pushed beside each bag.

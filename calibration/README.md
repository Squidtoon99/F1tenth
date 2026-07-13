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

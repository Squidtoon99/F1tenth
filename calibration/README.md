# calibration/

Sim-to-real and per-car physics calibration. Tooling and procedures for measuring
the parameters that make the simulator match the real car and each chassis behave
consistently.

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
.venv/bin/python -m pytest calibration/test_fit_vehicle_params.py -q
```

Outputs under `calibration/results/` are gitignored. Raw rosbags stay off-repo.

### Identifiable from current bags

- IMU unit/sign/bias (`static_imu_*`)
- ERPM↔speed gain, odom sign/scale (`accel_*` / `steering_*`)
- Servo gain/offset, weak max-steer and steer lag (`steering_*`)
- Acceleration limit, equivalent `f_drive_max`, weak `c_roll`, μ lower bound

### Not identifiable yet

- Active `f_brake_max`, motor `Kt` / current limits, `power_max`
- Suspension stiffness/damping / anti-roll
- Open-loop current→force gain

These need boxed-wheel current/brake bags with `/sensors/core` (Part 2): record a
bag while commanding `/commands/motor/current` and `/commands/motor/brake` with the
wheels off the ground, then pass it to `fit_vehicle_params.py` alongside the others —
`fit_current_to_force` / `fit_brake_current` activate automatically when those topics
are present. Force-mode actuation is described in
[`docs/adr/0005-force-mode-vesc-actuation.md`](../docs/adr/0005-force-mode-vesc-actuation.md).

Outputs feed:

- the simulator model used by [`training/`](../training/)
- the per-car overlay [`deploy/cars/<carNN>/params.yaml`](../deploy/cars/)

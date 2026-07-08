# calibration/

Sim-to-real and per-car physics calibration. Tooling and procedures for measuring
the parameters that make the simulator match the real car and each chassis behave
consistently:

- VESC `speed_to_erpm` gain/offset and `steering_angle_to_servo` gain/offset
- wheelbase, mass, tire/friction parameters used by the sim
- sensor extrinsics (LiDAR/IMU mounting)

Outputs feed two places:

- the simulator model used by [`training/`](../training/) (so trained policies
  transfer), and
- the per-car overlay [`deploy/cars/<carNN>/params.yaml`](../deploy/cars/).

Keep calibration **scripts/notebooks and small result files** here. Raw calibration
logs and rosbags are heavy and gitignored — store them off-repo and reference them.

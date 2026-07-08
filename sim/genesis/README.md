# sim/genesis

The synthetic-data simulator used by RL training. It is consumed as a **pip
dependency** (`genesis-world`) by [`training/`](../../training/), not as a ROS
package — so there is no code to vendor here.

This directory holds notes and any small helper assets (e.g. the vehicle URDF used
by the sim) that are worth versioning. Large artifacts stay out of git.

Notes:

- Installed via `training/requirements.txt` / `training/environment.yml`.
- The training environment (`training/f1tenth_env/`) wraps this simulator into a
  vectorized RL environment.
- The observation layout the sim env produces comes from the shared contract
  ([`libs/f1tenth_contract`](../../libs/f1tenth_contract/)), so trained policies
  match the on-car inference node.

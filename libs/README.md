# `libs/`

Cross-cutting libraries that are not tied to a single ROS package.

- [`f1tenth_contract/`](f1tenth_contract/) — the observation/action contract shared
  by RL training and the on-car inference node (dual pip + ament package).

Packages here are discoverable by colcon from the repo root (they carry a
`package.xml`) and pip-installable for the pure-Python training code.

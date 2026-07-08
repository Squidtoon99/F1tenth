# f1tenth_contract

The single source of truth for the observation/action layout, shared by RL training
and the on-car RL inference node.

This is a **dual package**: a normal Python package (`pyproject.toml` + `setup.py`)
that also carries a `package.xml`, so it works in both build worlds without
duplication.

## Use from training (no rebuild on changes)

```bash
pip install -e libs/f1tenth_contract     # one time, wired into environment.yml/dev.sh
```

Because it is installed editable, editing `observation.py` (e.g. adding a field) is
picked up the next time you run the trainer — no build, no reinstall.

## Use from ROS (`racing_rl`)

`src/racing_rl` declares `<exec_depend>f1tenth_contract</exec_depend>` and the
workspace is built with `colcon build --symlink-install`, so Python-only edits need
no rebuild either. A full colcon build only happens when freezing a car image.

## C++ parity

The on-car C++ node cannot import Python, so `src/common/f1tenth_common` holds a C++
mirror of this layout. A parity test keeps them in sync. This is a deploy-time
concern and does not affect the training loop.

## Migration note

Consolidate the observation math currently duplicated across
`F1tenth-Genesis/f1tenth_env/observations.py`,
`ros2_deploy/.../obs_core.py`, and the C++ `rl_obs_core.cpp` behind this package (+
the C++ mirror).

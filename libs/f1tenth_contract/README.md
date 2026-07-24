# f1tenth_contract

The shared observation/action **format** (dimensions and field slices), referenced
by RL training and the on-car RL inference stack. It defines the layout only; the
field-building math stays with each consumer (training's `f1tenth_env`, the deploy
`f1tenth_rl_agent`, and the C++ `f1tenth_common` mirror). Parity tests in those
consumers assert they match this contract so the format cannot silently drift.

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

## Scope

This package is intentionally **format-only and additive**: it encodes the vector
dimensions (384 base + 8-dim opponent block → 392) and the `(start, stop)` slice of
each field, mirroring the deployed `interfaces.py`. It does not reimplement or
replace the observation math in the training env, the deploy nodes, or the C++
mirror — those keep their own implementations, guarded by parity tests against these
values.

# tools/

Helper scripts for building and developing the workspace.

- `dev-setup.sh` — one-time host setup: checks prerequisites (git, docker, buildx)
  and enables cross-arch builds. Run once per machine.
- `dev.sh` — build/enter the dev container with a mounted, built workspace
  (`up` / `build` / `shell`). The one-command entry point for newcomers.
- `build.sh` — colcon build wrapper. Whole workspace, or `-t <group>` for one module
  group.

Lint configs live at the repo root: `.flake8` (Python) and `.clang-format` (C++).
They match what `ament_flake8` / `ament_clang_format` enforce in CI.

Typical flow:

```bash
./tools/dev-setup.sh        # once per machine
./tools/dev.sh up           # build + enter the dev container
# inside the container:
./tools/build.sh            # build everything
./tools/build.sh -t control # or one group
```

# Minimal Sensor-Policy Jetson Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and non-powered-stage a minimal native ARM64 ROS 2 sensor-policy image on `shereef@f1tenth`, with joystick control and a strictly matched 80 A drive / 20 A motor-brake policy contract.

**Architecture:** Refine the existing generic multi-stage `f1tenth-sensor-policy` image rather than introducing another runtime. Strengthen static and in-image inventory tests, then use the existing content-addressed checkpoint and rollback flow to stage the image and model without starting the hardware-driving graph.

**Tech Stack:** Docker BuildKit, NVIDIA JetPack/L4T R36.4.7, NVIDIA PyTorch iGPU, ROS 2 Humble, colcon, Bash, pytest, SSH

**Spec:** `docs/superpowers/specs/2026-08-17-minimal-sensor-policy-jetson-deployment-design.md`

## Global Constraints

- The runtime target is native `linux/arm64` on `shereef@f1tenth`; an amd64 or QEMU image is not certifiable.
- Keep model weights and `/config` out of the image and mount both read-only.
- Keep the ROS/model envelope exactly 80 A drive, 20 A motor brake, and 200 A/s slew.
- Treat 4 A battery regen as a manually verified VESC firmware prerequisite; never invoke a VESC configuration writer.
- Exclude particle filtering, `range_libc`, localization, mapping, SLAM, Nav2, planning, the classical racing graph, the 392-D RL graph, and `ackermann_mux`.
- Do not edit vendored `src/vehicle/**`; only build the required existing vehicle packages.
- Never start the powered graph automatically. Build, smoke, stage, and stop.
- Preserve the classical image and `policy.pt`, and retain the previous sensor-policy image/checkpoint rollback targets.
- Do not build from a dirty tree under a clean git-SHA tag.

---

### Task 1: Lock the Minimal Image Contract with Host Tests

**Files:**
- Create: `deploy/test/test_sensor_policy_image.py`
- Modify: `deploy/docker/Dockerfile.sensor_policy`

**Interfaces:**
- Consumes: `sensor_policy.launch.py` package closure and the accepted package list in ADR 0025.
- Produces: static tests defining the permitted Dockerfile build scope and forbidden runtime subsystems.

- [ ] **Step 1: Write failing static image-boundary tests**

Create `deploy/test/test_sensor_policy_image.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "deploy" / "docker" / "Dockerfile.sensor_policy"


def _dockerfile() -> str:
    return DOCKERFILE.read_text()


def test_sensor_policy_image_uses_multi_stage_runtime():
    source = _dockerfile()
    assert "FROM ${BASE_IMAGE} AS ros_runtime" in source
    assert "FROM ros_runtime AS build" in source
    assert "FROM ros_runtime AS runtime" in source
    assert "COPY --from=build /ws/install /ws/install" in source
    assert "COPY --from=build /ws/src" not in source


def test_sensor_policy_build_excludes_unneeded_subsystems():
    source = _dockerfile()
    ignored = source.split("--packages-ignore", 1)[1]
    for package in (
        "range_lib", "particle_filter", "f1tenth_localization",
        "f1tenth_rl_vehicle", "f1tenth_racing_algo", "f1tenth_planning",
        "ackermann_mux", "f1tenth_mapping", "f1tenth_perception",
    ):
        assert package in ignored


def test_sensor_policy_runtime_omits_build_and_localization_dependencies():
    runtime_apt = _dockerfile().split(
        "FROM ${BASE_IMAGE} AS ros_runtime", 1
    )[1].split("FROM ros_runtime AS build", 1)[0]
    for forbidden in (
        "python3-colcon-common-extensions", "python3-rosdep",
        "slam-toolbox", "nav2-map-server", "range-libc", "python3-scipy",
    ):
        assert forbidden not in runtime_apt
```

- [ ] **Step 2: Run the test and confirm RED**

Run:

```bash
.venv/bin/python -m pytest deploy/test/test_sensor_policy_image.py -v
```

Expected: a new assertion fails if the Dockerfile does not express the complete forbidden package set exactly.

- [ ] **Step 3: Make the Dockerfile closure deterministic**

Keep the three stages and final-stage copy boundary. Remove the trailing `|| true` from `rosdep install` so unresolved build dependencies fail the image build. Retain this explicit build closure:

```dockerfile
RUN source /opt/ros/humble/setup.bash \
    && rosdep update \
    && rosdep install --from-paths src libs --ignore-src -y -r \
    && colcon build --merge-install --packages-up-to f1tenth_bringup \
      --packages-ignore \
        range_lib particle_filter f1tenth_localization \
        f1tenth_rl_vehicle f1tenth_racing_algo f1tenth_planning \
        ackermann_mux f1tenth_common f1tenth_mapping f1tenth_perception \
      --cmake-args -DBUILD_TESTING=OFF
```

Do not change the digest-pinned base or add packages.

- [ ] **Step 4: Run static and formatting checks**

```bash
.venv/bin/python -m pytest deploy/test/test_sensor_policy_image.py -v
git diff --check -- deploy/docker/Dockerfile.sensor_policy deploy/test/test_sensor_policy_image.py
```

Expected: PASS and no diff errors.

- [ ] **Step 5: Commit**

```bash
git add deploy/docker/Dockerfile.sensor_policy deploy/test/test_sensor_policy_image.py
git commit -m "deploy: lock minimal sensor-policy image closure"
```

---

### Task 2: Prove Required and Forbidden Runtime Inventory

**Files:**
- Modify: `deploy/docker/smoke_sensor_policy.sh`
- Modify: `deploy/test/test_sensor_policy_image.py`

**Interfaces:**
- Consumes: the installed `/ws/install` tree from Task 1 and optional `CHECKPOINT_PATH`.
- Produces: `smoke_sensor_policy.sh` with `REQUIRE_CHECKPOINT=0|1`, required executable checks, and forbidden package checks.

- [ ] **Step 1: Add failing smoke-contract tests**

Append:

```python
SMOKE = ROOT / "deploy" / "docker" / "smoke_sensor_policy.sh"


def test_sensor_policy_smoke_rejects_forbidden_packages():
    source = SMOKE.read_text()
    assert "FORBIDDEN_PACKAGES=(" in source
    for package in (
        "particle_filter", "f1tenth_localization", "f1tenth_mapping",
        "f1tenth_planning", "f1tenth_racing_algo", "f1tenth_rl_vehicle",
        "ackermann_mux",
    ):
        assert f'  "{package}"' in source
    assert "unexpected package ${pkg}" in source


def test_sensor_policy_smoke_can_require_checkpoint():
    source = SMOKE.read_text()
    assert 'REQUIRE_CHECKPOINT="${REQUIRE_CHECKPOINT:-0}"' in source
    assert "checkpoint required but missing" in source
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
.venv/bin/python -m pytest deploy/test/test_sensor_policy_image.py \
  -k 'smoke_rejects or smoke_can_require' -v
```

Expected: both fail against the current smoke script.

- [ ] **Step 3: Implement checkpoint and forbidden-package gates**

Add `REQUIRE_CHECKPOINT="${REQUIRE_CHECKPOINT:-0}"` beside the smoke settings and document it in the script header. Use this missing-checkpoint branch:

```bash
elif [ "${REQUIRE_CHECKPOINT}" = "1" ]; then
  echo "checkpoint required but missing: ${CHECKPOINT_PATH}" >&2
  exit 1
else
  echo "==> No checkpoint at ${CHECKPOINT_PATH} — skipping artifact load"
fi
```

After required executable inventory, add:

```bash
echo "==> Forbidden package inventory"
FORBIDDEN_PACKAGES=(
  "particle_filter"
  "f1tenth_localization"
  "f1tenth_mapping"
  "f1tenth_planning"
  "f1tenth_racing_algo"
  "f1tenth_rl_vehicle"
  "ackermann_mux"
)
for pkg in "${FORBIDDEN_PACKAGES[@]}"; do
  if ros2 pkg prefix "${pkg}" >/dev/null 2>&1; then
    echo "unexpected package ${pkg} in sensor-policy runtime" >&2
    exit 1
  fi
  echo "absent ${pkg}"
done
```

- [ ] **Step 4: Run tests and shell syntax validation**

```bash
.venv/bin/python -m pytest deploy/test/test_sensor_policy_image.py -v
bash -n deploy/docker/smoke_sensor_policy.sh
git diff --check -- deploy/docker/smoke_sensor_policy.sh deploy/test/test_sensor_policy_image.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/docker/smoke_sensor_policy.sh deploy/test/test_sensor_policy_image.py
git commit -m "deploy: verify sensor-policy runtime inventory"
```

---

### Task 3: Make Release Builds and Staging Fail Closed

**Files:**
- Modify: `deploy/scripts/build_image.sh`
- Modify: `deploy/scripts/load_to_jetson.sh`
- Modify: `deploy/test/test_sensor_policy_image.py`

**Interfaces:**
- Consumes: a clean git worktree, a native ARM64 Docker host, and a gate-passing checkpoint.
- Produces: SHA-truthful builds, mandatory checkpoint smoke before sensor-policy promotion, and no automatic powered launch.

- [ ] **Step 1: Add failing script-safety tests**

Append:

```python
BUILD = ROOT / "deploy" / "scripts" / "build_image.sh"
LOAD = ROOT / "deploy" / "scripts" / "load_to_jetson.sh"


def test_release_build_rejects_dirty_worktree_by_default():
    source = BUILD.read_text()
    assert 'ALLOW_DIRTY="${ALLOW_DIRTY:-0}"' in source
    assert "git status --porcelain" in source
    assert "refusing SHA-tagged build from dirty worktree" in source


def test_sensor_policy_stage_requires_checkpoint_and_smoke():
    source = LOAD.read_text()
    assert "CHECKPOINT is required for sensor_policy staging" in source
    assert "REQUIRE_CHECKPOINT=1" in source
    assert "smoke_sensor_policy.sh" in source
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
.venv/bin/python -m pytest deploy/test/test_sensor_policy_image.py \
  -k 'release_build or stage_requires' -v
```

Expected: both fail.

- [ ] **Step 3: Reject misleading dirty SHA builds**

In `build_image.sh`, document and add. The second branch permits a `git archive`
build context only when its provenance is supplied explicitly:

```bash
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "${ALLOW_DIRTY}" != "1" ] && [ -n "$(git status --porcelain)" ]; then
    echo "refusing SHA-tagged build from dirty worktree; commit/stash changes or set ALLOW_DIRTY=1 for a non-release diagnostic build" >&2
    exit 1
  fi
elif [ -z "${GITSHA:-}" ]; then
  echo "GITSHA is required outside a git worktree" >&2
  exit 1
fi
```

Release staging must never set `ALLOW_DIRTY=1`.

- [ ] **Step 4: Require and smoke the sensor-policy checkpoint before promotion**

Reject an empty checkpoint:

```bash
if [ "${TARGET}" = "sensor_policy" ] && [ -z "${CHECKPOINT:-}" ]; then
  echo "CHECKPOINT is required for sensor_policy staging" >&2
  exit 1
fi
```

After loading the immutable SHA tag, but before retagging `:sensor-policy`, stage the content-hashed checkpoint and run:

```bash
CKPT_CONTAINER="/policies/$(basename "${CKPT_REMOTE}")"
docker run --rm --runtime nvidia \
  -e REQUIRE_CHECKPOINT=1 \
  -e CHECKPOINT_PATH="${CKPT_CONTAINER}" \
  -v /opt/f1tenth/policies:/policies:ro \
  --entrypoint smoke_sensor_policy.sh "${IMAGE}:${GITSHA}"
```

Only after this succeeds may the script update `e2e_policy.pt`, retag the moving image, update `/config`, and write the run helper. Keep the helper operator-invoked; never run it.

- [ ] **Step 5: Run tests and syntax checks**

```bash
.venv/bin/python -m pytest deploy/test/test_sensor_policy_image.py -v
bash -n deploy/scripts/build_image.sh
bash -n deploy/scripts/load_to_jetson.sh
git diff --check -- deploy/scripts/build_image.sh deploy/scripts/load_to_jetson.sh \
  deploy/test/test_sensor_policy_image.py
```

Expected: PASS.

- [ ] **Step 6: Exercise the read-only deployment preview**

```bash
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the gate-passing policy selected by evaluation}"
TARGET=sensor_policy DRY_RUN_DOCS=1 CHECKPOINT="${CHECKPOINT}" \
  deploy/scripts/load_to_jetson.sh shereef@f1tenth - car01
```

Expected: inventory succeeds and the script stops before copy, load, tag, config, policy, or helper changes.

- [ ] **Step 7: Commit**

```bash
git add deploy/scripts/build_image.sh deploy/scripts/load_to_jetson.sh \
  deploy/test/test_sensor_policy_image.py
git commit -m "deploy: fail closed during sensor-policy staging"
```

---

### Task 4: Align Operator Documentation and the Deployment ADR

**Files:**
- Create: `docs/adr/0027-minimal-sensor-policy-jetson-deployment.md`
- Modify: `docs/deployment.md`
- Modify: `deploy/README.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: one authoritative non-powered build/stage procedure and an ADR recording the runtime boundary.

- [ ] **Step 1: Write ADR 0027 from `docs/adr/0000-template.md`**

Record that the runtime includes CUDA inference, ROS hardware drivers, joystick ownership, and the current gate, while excluding localization and classical racing. Record native clean-revision ARM64 builds, required/forbidden package inventory, read-only mounted artifacts, no automatic powered launch, the 80/20 ROS envelope, and the manual external 4 A regen prerequisite.

Use this decision text:

```markdown
## Decision

Use the dedicated generic multi-stage sensor-policy image, build its exact clean
revision natively on the Jetson, prove required and forbidden package inventory,
mount config and a content-addressed checkpoint read-only, and stop deployment
before powered launch. Enforce 80 A drive and 20 A motor brake in artifact and
ROS metadata. Treat 4 A battery regen as a manual VESC prerequisite and never
write VESC firmware configuration from deployment tooling.
```

- [ ] **Step 2: Update deployment commands**

In the sensor-policy section of `docs/deployment.md`, document the native clean-tree build, checkpoint-required smoke, snapshot/load, and final staged-state inspection. State that staging does not invoke the run helper and does not verify or modify VESC firmware. Correct any statement that describes 25 A as the requested Jetson-controlled brake maximum; that maximum is 20 A.

- [ ] **Step 3: Update the short deploy README**

Point to ADR 0027, list excluded subsystems, and state the 80/20 Jetson envelope plus external manual 4 A regen prerequisite. Keep the full command sequence only in `docs/deployment.md`.

- [ ] **Step 4: Check documentation consistency**

```bash
rg -n "25 A|5 A|particle|range_libc|80 A|20 A|4 A|powered" \
  docs/deployment.md deploy/README.md \
  docs/adr/0027-minimal-sensor-policy-jetson-deployment.md
git diff --check -- docs/deployment.md deploy/README.md \
  docs/adr/0027-minimal-sensor-policy-jetson-deployment.md
```

Expected: behavior is internally consistent and no automatic powered launch is claimed.

- [ ] **Step 5: Commit**

```bash
git add docs/adr/0027-minimal-sensor-policy-jetson-deployment.md \
  docs/deployment.md deploy/README.md
git commit -m "docs: record minimal sensor-policy deployment"
```

---

### Task 5: Run the Full Repository Release Gate

**Files:**
- No source changes expected.
- Record command output in the handoff; do not commit logs or build trees.

**Interfaces:**
- Consumes: Tasks 1-4 and the eventual gate-passing model.
- Produces: evidence that the clean release candidate is ready for native Jetson build.

- [ ] **Step 1: Run pure-Python deployment and artifact tests**

```bash
PYTHONPATH="libs/f1tenth_policy:src/racing_rl/f1tenth_rl_agent:training" \
  .venv/bin/python -m pytest \
    deploy/test/test_sensor_policy_image.py \
    libs/f1tenth_policy/test \
    src/racing_rl/f1tenth_rl_agent/test/test_sensor_policy_model.py \
    src/racing_rl/f1tenth_rl_agent/test/test_sensor_inference_runtime.py -v
```

Expected: PASS with no skipped dependency test introduced by this work.

- [ ] **Step 2: Build and test the ROS workspace in the dev container**

```bash
./tools/build.sh
./tools/test.sh
```

Expected: the complete repository build, unit tests, and ament lint pass. Vendored packages remain excluded from testing according to `AGENTS.md`, not from building.

- [ ] **Step 3: Run shell and diff validation**

```bash
bash -n deploy/docker/smoke_sensor_policy.sh \
  deploy/scripts/build_image.sh deploy/scripts/load_to_jetson.sh
git diff --check
git status --short
```

Expected: syntax and diff checks pass. Do not build a release until all intended deployment and compatible 80/20 contract changes are committed. Preserve unrelated user changes.

- [ ] **Step 4: Correct only verified in-scope failures**

For any failure, rerun its narrow test, apply the smallest in-scope fix, rerun the relevant full gate, and commit only those files with a `deploy:` message. Never weaken or skip a test.

---

### Task 6: Build and Non-Powered-Stage on the Jetson

**Files:**
- Remote generated artifacts: Docker image tags and `/opt/f1tenth/rollback/sensor_policy/**`.
- Remote staged artifacts: `/opt/f1tenth/config/**` and `/opt/f1tenth/policies/e2e_policy.pt`.
- Modify after successful staging: `deploy/releases/manifest.csv`.

**Interfaces:**
- Consumes: the clean tested release revision and an explicitly selected gate-passing checkpoint.
- Produces: native `f1tenth-sensor-policy:${release_sha}`, staged read-only model/config, run helper, rollback metadata, and no running container.

- [ ] **Step 1: Verify target state read-only**

```bash
ssh shereef@f1tenth 'uname -m; cat /etc/nv_tegra_release; docker version; \
  docker info | sed -n "/Runtimes/,+2p"; docker ps; df -h /var/lib/docker'
```

Expected: `aarch64`, L4T R36.4.7, usable Docker/NVIDIA runtime, no conflicting VESC-owning container, and sufficient disk.

- [ ] **Step 2: Transfer the exact committed release**

```bash
release_sha="$(git rev-parse --short HEAD)"
git archive --format=tar.gz -o "/tmp/f1tenth-${release_sha}.tar.gz" HEAD
scp "/tmp/f1tenth-${release_sha}.tar.gz" shereef@f1tenth:/tmp/
ssh shereef@f1tenth "mkdir -p /tmp/f1tenth-build-${release_sha} && \
  tar -xzf /tmp/f1tenth-${release_sha}.tar.gz -C /tmp/f1tenth-build-${release_sha}"
```

Expected: the remote context contains only committed release files.

- [ ] **Step 3: Build natively without running hardware nodes**

```bash
ssh shereef@f1tenth "cd /tmp/f1tenth-build-${release_sha} && \
  GITSHA=${release_sha} TARGET=sensor_policy SMOKE=0 deploy/scripts/build_image.sh"
```

Expected: the native immutable SHA tag is created. Do not run the default image command.

- [ ] **Step 4: Run checkpoint-required non-powered smoke**

```bash
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the gate-passing policy selected by evaluation}"
test -f "${CHECKPOINT}"
scp "${CHECKPOINT}" shereef@f1tenth:/tmp/e2e-policy-candidate.pt
ssh shereef@f1tenth "docker run --rm --runtime nvidia \
  -e REQUIRE_CHECKPOINT=1 \
  -v /tmp/e2e-policy-candidate.pt:/policies/e2e_policy.pt:ro \
  --entrypoint smoke_sensor_policy.sh \
  f1tenth-sensor-policy:${release_sha}"
```

Expected: CUDA, rclpy, artifact inference, required executable inventory, forbidden package inventory, and driver/gate-free launch all pass.

- [ ] **Step 5: Snapshot and stage through the rollback-aware loader**

```bash
ssh shereef@f1tenth "cd /tmp/f1tenth-build-${release_sha} && \
  GITSHA=${release_sha} TARGET=sensor_policy deploy/scripts/snapshot.sh"
scp "shereef@f1tenth:/tmp/f1tenth-build-${release_sha}/deploy/snapshots/f1tenth-sensor-policy-${release_sha}.tar.gz" \
  deploy/snapshots/
TARGET=sensor_policy CHECKPOINT="${CHECKPOINT}" \
  ACTOR_LAYOUT=2 POLICY_FORMAT=4 JETPACK_L4T=R36.4.7 \
  deploy/scripts/load_to_jetson.sh shereef@f1tenth \
  "deploy/snapshots/f1tenth-sensor-policy-${release_sha}.tar.gz" car01
```

Expected: rollback state is preserved; smoke passes before promotion; the isolated tag, content-hashed checkpoint, and helper are staged; nothing starts.

- [ ] **Step 6: Verify staged state without actuation**

```bash
ssh shereef@f1tenth 'docker ps --format "{{.Names}} {{.Image}} {{.Status}}"; \
  docker image inspect f1tenth-sensor-policy:sensor-policy \
    --format "{{.Id}} {{json .Config.Labels}}"; \
  readlink -f /opt/f1tenth/policies/e2e_policy.pt; \
  ls -l /opt/f1tenth/rollback/sensor_policy/run.sh \
    /opt/f1tenth/rollback/sensor_policy/image.id \
    /opt/f1tenth/rollback/sensor_policy/checkpoint.path'
```

Expected: no sensor-policy container is running, and all staged/rollback references resolve.

- [ ] **Step 7: Commit the manifest row**

Review the loader's single new row for image, snapshot, checkpoint, actor-layout, policy-format, and L4T provenance:

```bash
git add deploy/releases/manifest.csv
git commit -m "deploy: record sensor-policy Jetson staging"
```

Do not start `/opt/f1tenth/rollback/sensor_policy/run.sh`. Hand it to the operator with the explicit prerequisite that the VESC 4 A battery-regen maximum and physical safety conditions have been checked outside the Jetson deployment.

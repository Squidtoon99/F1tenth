# Race-Ready 1097-D Sensor Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the 1097-D PPO sensor-policy stack to an 80 A drive / 20 A brake envelope, prove simulator/deploy and rosbag preprocessing parity, and launch a monitored cold-start seed-55 run capped at 2B transitions.

**Architecture:** Keep preprocessing v3 and the promoted Cursor sim-to-real model, changing only the directional current envelope and its derived Warp slew. Treat the policy artifact as the compatibility boundary: trainer, ROS inference, and current gate share 80/20 metadata and reject mismatches. Validate locally with real modules and held-out bag replay before launching one non-resumable PPO run.

**Tech Stack:** Python 3.12, PyTorch/CUDA, NVIDIA Warp, PPO, W&B, ROS 2 Humble Python nodes, pytest, flake8.

## Global Constraints

- Target only the 1097-D `experiments/e2e-ppo` sensor-policy path.
- Preserve all pre-existing user/Cursor worktree changes and do not touch vendored `src/vehicle/**` code.
- Use preprocessing v3, policy format v4, actor layout v2, force mode, and delta steering.
- Use 80 A policy/ROS drive, 20 A policy/ROS brake, 200 A/s physical slew, and 2.5/s normalized Warp slew.
- Keep 25 A hard motor brake and 4 A battery regen as tomorrow's VESC firmware settings; do not claim they are enforced by tonight's code.
- Cold-start seed 55; no init checkpoint or resume checkpoint; 2,000,000,000 transition ceiling.
- Do not build the arm64 runtime image tonight.
- Do not stop a structurally healthy training run for a temporary metric regression.

---

### Task 1: Lock the shared 80/20 current contract

**Files:**
- Modify: `libs/f1tenth_policy/f1tenth_policy/current.py`
- Modify: `libs/f1tenth_policy/test/test_current.py`
- Modify: `src/racing_rl/f1tenth_rl_agent/test/fixtures/actor_layout_v2.json`
- Modify: `libs/f1tenth_policy/test/test_layout_actor.py`

**Interfaces:**
- Consumes: preprocessing-v3 `applied_current_fraction(signed_applied_current_a, i_drive_max_a, i_brake_max_a) -> float`.
- Produces: `TRAINING_I_DRIVE_MAX_A = 80.0`, `TRAINING_I_BRAKE_MAX_A = 20.0`, and `TRAINING_I_SLEW_A_PER_S = 200.0` for artifacts and deploy validation.

- [ ] **Step 1: Change the current-scale test before production constants**

```python
def test_training_current_scale_is_80a_drive_20a_brake_envelope():
    assert TRAINING_I_DRIVE_MAX_A == pytest.approx(80.0)
    assert TRAINING_I_BRAKE_MAX_A == pytest.approx(20.0)
    assert TRAINING_I_SLEW_A_PER_S == pytest.approx(200.0)
    assert TRAINING_I_SLEW_A_PER_S / TRAINING_I_DRIVE_MAX_A == pytest.approx(2.5)
```

Update fraction examples to prove `40 / 80 == 0.5` and `-10 / 20 == -0.5`.

- [ ] **Step 2: Run the focused test and observe the intended failure**

Run:

```bash
PYTHONPATH=libs/f1tenth_policy .venv/bin/python -m pytest \
  libs/f1tenth_policy/test/test_current.py -q
```

Expected: FAIL because the working tree still declares 100 A drive and 10 A brake.

- [ ] **Step 3: Update the shared constants and fixture metadata**

Set the constants to 80/20/200 and update fixture fields and notes to the same directional envelope. Keep `applied_current_fraction` behavior unchanged.

- [ ] **Step 4: Strengthen artifact round-trip assertions**

Add assertions to `test_artifact_roundtrip_validation`:

```python
assert payload["i_drive_max_a"] == 80.0
assert payload["i_brake_max_a"] == 20.0
assert payload["i_slew_a_per_s"] == 200.0
```

- [ ] **Step 5: Run shared policy tests**

```bash
PYTHONPATH=libs/f1tenth_policy .venv/bin/python -m pytest \
  libs/f1tenth_policy/test/test_current.py \
  libs/f1tenth_policy/test/test_layout_actor.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the reviewed contract unit**

```bash
git add libs/f1tenth_policy/f1tenth_policy/current.py \
  libs/f1tenth_policy/test/test_current.py \
  libs/f1tenth_policy/test/test_layout_actor.py \
  src/racing_rl/f1tenth_rl_agent/test/fixtures/actor_layout_v2.json
git commit -m "policy: align sensor current envelope to 80A"
```

### Task 2: Align trainer and Warp configuration

**Files:**
- Modify: `training/config.py`
- Modify: `training/f1tenth_sim/params.py`
- Modify: `training/tests/test_sensor_config.py`
- Modify: `training/tests/test_config_patch.py`
- Modify: `training/tests/test_warp_torch_parity.py`
- Modify: `training/standalone_trainer.py`

**Interfaces:**
- Consumes: shared 80/20/200 artifact defaults from Task 1.
- Produces: resolved environment values `i_drive_max_a=80.0`, `i_brake_max_a=20.0`, `i_slew_a_per_s=200.0`, and `warp_sim.longitudinal_slew_rate_per_s=2.5`.

- [ ] **Step 1: Change default-config assertions first**

```python
def test_default_current_scale_is_80a_drive_20a_brake_envelope():
    env = DEFAULT_CONFIG["env"]
    assert env["i_drive_max_a"] == pytest.approx(80.0)
    assert env["i_brake_max_a"] == pytest.approx(20.0)
    assert env["i_slew_a_per_s"] == pytest.approx(200.0)
    assert env["warp_sim"]["longitudinal_slew_rate_per_s"] == pytest.approx(2.5)
    assert env["warp_sim"]["longitudinal_slew_rate_per_s"] == pytest.approx(
        env["i_slew_a_per_s"] / env["i_drive_max_a"]
    )
```

- [ ] **Step 2: Run the focused config tests and observe failure**

```bash
PYTHONPATH=training:libs/f1tenth_policy .venv/bin/python -m pytest \
  training/tests/test_sensor_config.py \
  training/tests/test_config_patch.py -q
```

Expected: FAIL on the old 100/10/2.0 values.

- [ ] **Step 3: Update defaults and fallback metadata**

Change only the physical envelope and derived slew. Preserve the accepted `f_drive_max=23.0`, `f_brake_max=5.2`, `power_max=320.0`, measured IMU DR, friction DR, and LiDAR settings. Ensure `save_policy_artifact` falls back to the shared/default 80/20/200 values rather than 100/10/200 literals.

- [ ] **Step 4: Run config, artifact, and Warp parity tests**

```bash
PYTHONPATH=training:libs/f1tenth_policy:libs/f1tenth_contract \
  .venv/bin/python -m pytest \
  training/tests/test_sensor_config.py \
  training/tests/test_config_patch.py \
  training/tests/test_warp_torch_parity.py \
  training/tests/test_domain_randomization.py \
  training/tests/test_vehicle_geometry.py -q
```

Expected: PASS without regenerating the committed Torch trajectory unless the test explicitly proves the current-only slew changes its authoritative reference.

- [ ] **Step 5: Commit the trainer/simulator unit**

```bash
git add training/config.py training/f1tenth_sim/params.py \
  training/standalone_trainer.py training/tests/test_sensor_config.py \
  training/tests/test_config_patch.py training/tests/test_warp_torch_parity.py
git commit -m "training: use 80A sensor policy envelope"
```

### Task 3: Align the ROS producer and current gate

**Files:**
- Modify: `src/racing_rl/f1tenth_rl_agent/config/sensor_policy.yaml`
- Modify: `src/control/f1tenth_control/config/rl_current_gate.yaml`
- Modify: `deploy/cars/car01/params.yaml`
- Modify: `src/common/f1tenth_bringup/launch/sensor_policy.launch.py`
- Modify: `src/common/f1tenth_bringup/test/test_sensor_policy_launch.py`
- Modify: `src/racing_rl/f1tenth_rl_agent/test/test_sensor_racer_node.py`
- Modify: `src/racing_rl/f1tenth_rl_agent/test/test_sensor_policy_model.py`
- Modify: `src/racing_rl/f1tenth_rl_agent/test/test_sensor_inference_runtime.py`
- Modify: `src/control/f1tenth_control/test/test_current_gate.py`
- Modify: `src/control/f1tenth_control/test/test_rl_current_gate_node.py`

**Interfaces:**
- Consumes: artifact metadata `i_drive_max_a=80.0`, `i_brake_max_a=20.0`.
- Produces: one command path whose desired and admitted outputs never exceed 80 A drive or 20 A brake and never command both.

- [ ] **Step 1: Change launch/config assertions before defaults**

Replace the old 5 A overlay assertion with:

```python
def test_car_overlay_matches_sensor_policy_artifact_envelope():
    racer = _overlay_node_block("sensor_racer")
    gate = _overlay_node_block("rl_current_gate")
    assert "i_drive_max_a: 80.0" in racer
    assert "i_brake_max_a: 20.0" in racer
    assert "i_drive_max_a: 80.0" in gate
    assert "i_brake_max_a: 20.0" in gate
```

Add real-module mapping cases for longitudinal actions `1.0`, `-1.0`, non-finite input, and mutually exclusive current fields.

- [ ] **Step 2: Run focused ROS-source tests and observe failure**

```bash
PYTHONPATH=libs/f1tenth_policy:src/racing_rl/f1tenth_rl_agent:src/control/f1tenth_control \
  .venv/bin/python -m pytest \
  src/common/f1tenth_bringup/test/test_sensor_policy_launch.py \
  src/racing_rl/f1tenth_rl_agent/test/test_sensor_racer_node.py \
  src/racing_rl/f1tenth_rl_agent/test/test_sensor_policy_model.py \
  src/control/f1tenth_control/test/test_current_gate.py -q
```

Expected: FAIL on old package and overlay limits.

- [ ] **Step 3: Align package, launch, and per-car defaults**

Set sensor-policy producer and gate limits to 80/20, retain `i_brake_safe_a=5.0`, and retain `i_slew_a_per_s=200.0`. Do not change the classical `vesc_actuator` block or vendored VESC code. Ensure node constructor defaults cannot silently revert to old limits when YAML is absent.

- [ ] **Step 4: Prove artifact mismatch fails closed**

Keep tests showing 100/10 and preprocessing-v2 artifacts are rejected by an 80/20 runtime. Verify a matching 80/20 artifact loads.

- [ ] **Step 5: Run the full focused ROS-source suite**

```bash
PYTHONPATH=libs/f1tenth_policy:src/racing_rl/f1tenth_rl_agent:src/control/f1tenth_control \
  .venv/bin/python -m pytest \
  src/common/f1tenth_bringup/test/test_sensor_policy_launch.py \
  src/racing_rl/f1tenth_rl_agent/test/test_sensor_inference_runtime.py \
  src/racing_rl/f1tenth_rl_agent/test/test_sensor_policy_model.py \
  src/racing_rl/f1tenth_rl_agent/test/test_sensor_preprocessing.py \
  src/racing_rl/f1tenth_rl_agent/test/test_sensor_racer_node.py \
  src/control/f1tenth_control/test/test_current_gate.py \
  src/control/f1tenth_control/test/test_rl_current_gate_node.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the ROS alignment unit**

```bash
git add src/racing_rl/f1tenth_rl_agent src/control/f1tenth_control \
  src/common/f1tenth_bringup/launch/sensor_policy.launch.py \
  src/common/f1tenth_bringup/test/test_sensor_policy_launch.py \
  deploy/cars/car01/params.yaml
git commit -m "control: enforce 80A sensor policy limits"
```

### Task 4: Update decision and operator documentation

**Files:**
- Modify: `docs/adr/0026-normalized-current-observation.md`
- Modify: `docs/deployment.md`
- Modify: `training/README.md`
- Modify: `deploy/README.md`

**Interfaces:**
- Consumes: the implemented 80/20/200 contract.
- Produces: exact tonight/tomorrow commands and the distinction between 20 A actor brake, 25 A firmware motor brake, and 4 A firmware battery regen.

- [ ] **Step 1: Replace stale 100/10 and 5/5 statements with current behavior**

Document the directional normalization, artifact rejection, 2.5/s derived slew, no-Docker tonight scope, and tomorrow's manual VESC verification. Avoid claiming firmware settings are enforced by ROS.

- [ ] **Step 2: Resolve the conservative first-motion wording**

The deployed artifact and ROS inference must stay at 80/20 to preserve normalization. Document boxed-wheel validation first, then low-demand commanded-action tests and incremental observed-track escalation; do not instruct operators to load the 80/20 artifact under mismatched 5/5 ROS parameters.

- [ ] **Step 3: Check documentation and commit**

```bash
rg -n "100 A|100A|10 A brake|5 A|5A|2\.0/s|2\.0 normalized" \
  docs/adr/0026-normalized-current-observation.md docs/deployment.md \
  training/README.md deploy/README.md
git diff --check
git add docs/adr/0026-normalized-current-observation.md docs/deployment.md \
  training/README.md deploy/README.md
git commit -m "docs: record 80A sensor policy envelope"
```

Expected: remaining matches refer only to historical evidence or clearly labelled safe-brake behavior.

### Task 5: Verify the complete local code state

**Files:**
- Test only; no production changes unless a real failure is diagnosed.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: evidence that the exact working tree is fit for the training smoke.

- [ ] **Step 1: Run full pure-Python training and library tests**

```bash
PYTHONPATH=training:libs/f1tenth_policy:libs/f1tenth_contract \
  .venv/bin/python -m pytest training/tests libs/f1tenth_policy/test \
  libs/f1tenth_contract/test
```

- [ ] **Step 2: Run calibration tests and lint**

```bash
.venv/bin/python -m pytest calibration
.venv/bin/python -m flake8 training libs/f1tenth_policy calibration \
  src/racing_rl/f1tenth_rl_agent src/control/f1tenth_control
git diff --check
```

- [ ] **Step 3: Run ROS build/test if Docker becomes available; otherwise record the explicit deferral**

```bash
./tools/build.sh -t racing_rl
./tools/test.sh --packages-select \
  f1tenth_rl_agent f1tenth_control f1tenth_bringup
```

Expected tonight: Docker may remain unavailable; this does not replace tomorrow's Jetson image and ROS smoke gate.

### Task 6: Create and validate the exact 2B launch configuration

**Files:**
- Create: `training/configs/ecss_solo_v3_80a_ppo.json`
- Test: `training/tests/test_config_patch.py`

**Interfaces:**
- Consumes: the phase-11 PPO recipe and implemented DEFAULT_CONFIG.
- Produces: a minimal version-controlled config patch for the smoke and long run.

- [ ] **Step 1: Add a test that loads the exact patch**

The test must call `load_config_patch` and assert PPO, ECSS track, enabled DR, 80/20/200, 2.5/s slew, rollout 128, four epochs, 32 env minibatches, entropy 0.01, and policy export every 2,560,000 transitions.

- [ ] **Step 2: Run the test and observe the missing-file failure**

```bash
PYTHONPATH=training:libs/f1tenth_policy .venv/bin/python -m pytest \
  training/tests/test_config_patch.py -q
```

- [ ] **Step 3: Create the minimal patch**

Start from the prior `ecss_solo_v3_100a_ppo.json`, use 80/20/200 and 2.5/s, retain measured DR and PPO settings, and set no opponent strategy. Do not duplicate unchanged defaults without a provenance reason.

- [ ] **Step 4: Validate resolution and CLI precedence**

```bash
cd training
../.venv/bin/python standalone_trainer.py \
  --config configs/ecss_solo_v3_80a_ppo.json --algorithm ppo \
  --num-envs 8 --total-transitions 1024 --opponent none --device cuda \
  --seed 55 --run-id config-validation-80a --no-wandb --no-compile
```

Expected: config snapshot resolves 80/20/200, 2.5/s, PPO, seed 55, and no init/resume checkpoint.

- [ ] **Step 5: Commit the launch configuration**

```bash
git add training/configs/ecss_solo_v3_80a_ppo.json \
  training/tests/test_config_patch.py
git commit -m "training: add race-day 80A PPO recipe"
```

### Task 7: Run smoke, held-out evaluation, and bag replay

**Files:**
- Outputs only: `training/outputs/runs/<smoke-id>/`
- Outputs only: `/home/ubuntu/f1tenth-sim2real-analysis/runs/race_ready_80a/`

**Interfaces:**
- Consumes: exact Task 6 config and current source state.
- Produces: go/no-go evidence for the long run.

- [ ] **Step 1: Run the short cold PPO smoke**

```bash
cd training
WANDB_MODE=disabled ../.venv/bin/python standalone_trainer.py \
  --config configs/ecss_solo_v3_80a_ppo.json --algorithm ppo \
  --num-envs 512 --total-transitions 8388608 --opponent none \
  --device cuda --seed 55 --run-id race-ready-80a-smoke --no-wandb
```

Expected: clean cold-start marker, advancing PPO updates, finite metrics, and at least three immutable policy exports.

- [ ] **Step 2: Evaluate the smoke checkpoint under nominal and unseen DR**

Use the existing phase-11 `phase11_eval_seed.py` with a copied 80/20 config and write new JSON outputs under `runs/race_ready_80a/`. Acceptance: finite observations/actions, zero clipping/OOD, no persistent period-2 oscillation, and nonzero motion. The 8.4M smoke need not meet mature lap-speed thresholds.

- [ ] **Step 3: Replay held-out bags without actuation**

Use the existing `phase11_bag_replay.py` against the new checkpoint and 80/20 normalization. Preserve the JSON report. Reject the launch on non-finite output, persistent ±1 drive/brake alternation, incompatible metadata, or unexpected current-channel clipping/OOD.

- [ ] **Step 4: Record the smoke verdict**

Write a small JSON or Markdown verdict in the external analysis run directory containing config hash, checkpoint hash, source SHA/diff hash, test commands, metrics, and `go_for_long_run`.

### Task 8: Launch and monitor the cold 2B W&B run

**Files:**
- Outputs only: `training/outputs/runs/race-ready-80a-ppo-s55-a001/`
- W&B run: `squidtoon99-ut-dallas/f1tenth-genesis`

**Interfaces:**
- Consumes: Task 7 `go_for_long_run=true` evidence.
- Produces: a live seed-55 PPO run with immutable selection checkpoints and 30-minute health records.

- [ ] **Step 1: Capture provenance and confirm no trainer is already running**

```bash
pgrep -af "standalone_trainer.py" || true
git status --short
git rev-parse HEAD
git diff --binary | sha256sum
nvidia-smi
```

- [ ] **Step 2: Launch the cold run**

```bash
cd training
WANDB_PROJECT=f1tenth-genesis \
WANDB_ENTITY=squidtoon99-ut-dallas \
WANDB_NAME=race-ready-80a-ppo-s55-a001 \
WANDB_TAGS=race-ready,ppo,1097d,prep-v3,80a,seed55,cold-start \
../.venv/bin/python standalone_trainer.py \
  --config configs/ecss_solo_v3_80a_ppo.json --algorithm ppo \
  --num-envs 4096 --total-transitions 2000000000 --opponent none \
  --device cuda --seed 55 --run-id race-ready-80a-ppo-s55-a001 \
  --wandb --wandb-mode online
```

Do not pass `--continuous`; the explicit 2B ceiling provides the requested long-running behavior while retaining a terminal bound.

- [ ] **Step 3: Verify the first checkpoint interval**

Confirm config snapshot values, W&B identity, cold-start markers, GPU utilization, finite metrics, advancing transitions, and policy exports. Stop only for an infrastructure failure or confirmed sustained learner-health failure.

- [ ] **Step 4: Check every 30 minutes**

At each check record timestamp, PID, transitions, last-log age, latest checkpoint and hash, GPU utilization/memory, finite counters, reward/laps/speed, OOB/collision terminations, entropy/KL, saturation, clipping/OOD, and oscillation indicators. Temporary performance regressions are observations, not stop conditions.

- [ ] **Step 5: Diagnose and cold-restart only after a real failure**

If the process exits or stops progressing, preserve the run directory and W&B identity, invoke `superpowers:systematic-debugging`, reproduce the cause, add a regression test, implement the smallest fix, rerun relevant gates, and launch `race-ready-80a-ppo-s55-a002` from scratch. Never use a policy-only checkpoint as `--resume-ckpt`.

- [ ] **Step 6: Prepare tomorrow's checkpoint shortlist**

Do not default to the last checkpoint. Shortlist checkpoints on the Pareto frontier of clean completion, lap time/speed, OOB/collision exposure, tail failures, action saturation, clipping/OOD, oscillation, and unseen-DR robustness for tomorrow's Jetson and physical gates.

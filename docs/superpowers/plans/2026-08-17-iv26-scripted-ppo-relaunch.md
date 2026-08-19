# IV26 Scripted PPO Relaunch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a cold seed-42 PPO run on `IV_2026_SIM` with the prior strong run's scripted-opponent recipe and the race-ready 80A/20A current contract.

**Architecture:** Derive one focused JSON patch from the prior run's resolved settings. Preserve the current simulator implementation, 23N drive-force limit, 320W power limit, 80A drive, 20A brake, and 200A/s physical slew; launch a finite 2B-transition run and verify live telemetry without a separate smoke run.

**Tech Stack:** Python, Warp, PyTorch PPO, W&B, tmux

## Global Constraints

- Do not stage or commit any files.
- Do not warm-start; use seed 42.
- Do not run a separate smoke test.
- Keep `f_drive_max=23.0`, `power_max=320.0`, and `ego_speed_cap_mps=0.0`.
- Keep `i_drive_max_a=80.0`, `i_brake_max_a=20.0`, `i_slew_a_per_s=200.0`, and normalized slew `2.5` per second.
- Use `IV_2026_SIM`, scripted opponent, target 3.5m/s, range 2.0-5.0m/s.
- Stop at 2,000,000,000 transitions and export every 2,560,000 transitions.

---

### Task 1: Exact launch patch

**Files:**
- Create: `training/configs/iv26_scripted_v3_80a_ppo.json`
- Test: `training/tests/test_config_patch.py`

**Interfaces:**
- Consumes: `standalone_trainer.load_config_patch()` and current `DEFAULT_CONFIG`.
- Produces: a validated PPO configuration patch used by the trainer CLI.

- [ ] Add a config-loader assertion covering track, opponent recipe, force/power limits, and current limits.
- [ ] Create the minimal JSON patch by copying the prior run recipe and changing only the approved current envelope and checkpoint cadence.
- [ ] Run `PYTHONPATH=training .venv/bin/python -m pytest -q training/tests/test_config_patch.py` and require success.

### Task 2: Launch and live verification

**Files:**
- Runtime output: `training/outputs/runs/race-ready-iv26-scripted-80a-ppo-s42-a001/`

**Interfaces:**
- Consumes: `training/configs/iv26_scripted_v3_80a_ppo.json`.
- Produces: W&B telemetry, policy checkpoints, and full critic/optimizer resume state.

- [ ] Launch in detached tmux with PPO, 4,096 environments, seed 42, scripted opponent, and 2B transitions.
- [ ] Verify the resolved snapshot contains `IV_2026_SIM`, scripted opponent, 23N/320W, 80A/20A/200A/s, and normalized slew 2.5/s.
- [ ] Verify GPU load, advancing transitions, zero non-finite rates, W&B sync, and the first policy plus resume-state checkpoint.

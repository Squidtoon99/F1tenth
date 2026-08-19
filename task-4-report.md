# Task 4 documentation report

Updated the four planned operator and decision documents without staging or
committing changes:

- `docs/adr/0026-normalized-current-observation.md`
- `docs/deployment.md`
- `training/README.md`
- `deploy/README.md`

The documentation now records the 80 A drive / 20 A policy motor-brake envelope,
200 A/s physical slew, and 2.5/s normalized Warp slew. It distinguishes the
20 A policy/ROS brake denominator from the VESC firmware's 25 A hard motor-brake
and 4 A battery-regen safeguards, which ROS does not enforce or verify.

First-motion guidance retains matching 80/20 artifact and ROS parameters:
boxed-wheel and low-demand commanded-action checks come before low-speed floor
testing and incremental escalation. It explicitly rejects a mismatched 5/5
runtime configuration. Overnight work is training and offline validation only;
the arm64 CUDA image is built and smoke-tested on the Jetson the next morning.

Verification performed:

```text
rg stale-value scan: no stale 100 A, 10 A brake, or 2.0/s statements found;
matches for "5 A" are the intended 25 A firmware ceiling or explicit 5/5
mismatch warnings.
git diff --check: passed.
```

Existing unrelated Cursor documentation edits were preserved.

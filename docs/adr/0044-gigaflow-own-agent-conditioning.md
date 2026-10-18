# 0044 — Own-agent Gigaflow dynamics and reward conditioning

- Status: Accepted
- Date: 2026-10-17
- Supplements: [0043](0043-gigaflow-track-affine-sampling.md)

## Context

Cusumano-Towner et al. 2025 ([arXiv 2502.03349v1](https://arxiv.org/html/2502.03349v1))
condition one reactive policy on per-agent dynamics and reward vectors so
self-play copies of π produce different styles. The baseline actor is 1097-d
lidar+IMU with no opponent pose; the critic is the 392-d privileged vector
including opponents.

## Decision

1. Opt in with `env.gigaflow_conditioning`. Concatenate a 10-d own-agent C
   after the baseline actor and critic: car length/width (fixed 1/10
   embodiment), throttle/steer response ~ X(1.25), accel/speed limits ~
   X(1.5), and reward weights ~ X(2) on collision, boundary, comfort, and
   progress. *Why:* that is the paper's C_dynamics / C_reward idea mapped onto
   this plant; size is not drawn from U(0.8, 7) m.
2. Apply C_dynamics to that agent's `drive_scale`, `steer_gain`, and speed
   cap. Scale the already-computed reward terms by C_reward. The actor never
   receives another car's C or pose. *Why:* other cars are lidar hits; the
   critic keeps the privileged 392-d plus own C.
3. Opponents are copies of π (`--self-play`) with independently sampled C.
   Table A3 knobs live in `training/configs/gigaflow_v2.json`: value clip off,
   cosine LR from 5e-4, γ=0.999, running-max advantage filter η=0.01,
   orthogonal init, spatial CAPS 0.1. *Why:* train the Gigaflow recipe without
   retuning SIM_READY_V2.

## Consequences

Actor obs is 1107-d and critic obs is 402-d only when the flag is on. Default
Lee 1097/392 paths are unchanged. Scripted 3.5 m/s opponents stay out of this
config.

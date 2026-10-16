# 0040 — LiDAR Dreamer world model

- Status: Accepted
- Date: 2026-10-16

## Context

The default sensor path is model-free PPO on 20 Hz Warp steps. Option 7 of the
architecture shortlist is LiDAR Dreamer (Kim et al., 2025): an RSSM with
Cartesian-polar LiDAR encoding and Lightweight State-Space Cells, with the
actor trained in imagination. Deploy keeps the compact encoder+actor. This is
an algorithm swap, not a PPO reward term.

## Decision

1. Add `algorithm=lidar_dreamer` with `model.actor_type=lidar_dreamer_ls3c`.
   Default remains QR-SAC / PPO + `lidar_cnn_gru`.
   *Why:* the world model and imagination loop must not share PPO's clip/GAE
   path.
2. Encode 1081-beam scans as a 3-channel Cartesian-polar chart (range, x, y)
   and replace the RSSM GRU with an LS3C. The actor at deploy is encoder +
   LS3C posterior mode + Gaussian head.
   *Why:* that is the LiDAR Dreamer representation; decoder, reward, continue,
   prior, and imagination critic stay in the training checkpoint.
3. Train the actor on imagined λ-returns (REINFORCE) from the RSSM prior.
   Control stays 20 Hz (`sim_dt=0.005`, `control_interval=10`).
   *Why:* imagination is the sample-efficiency claim; executing a 10 Hz or
   action-repeat policy is rejected.

## Consequences

- Austin arm: `training/configs/austin_e2e_lidar_dreamer.json`.
- Artifacts load through `actor_from_architecture` as `lidar_dreamer_ls3c`.
- Residual mix, action chunks, and QR-SAC stay off this algorithm.

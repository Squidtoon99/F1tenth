# 0041 — TinyLidarNet encoder on raw 1081 beams

- Status: Accepted
- Date: 2026-10-16

## Context

The default LiDAR GRU actor crushes 1081 Hokuyo beams to 32 bins with
AdaptiveAvgPool1d before the recurrent core. TinyLidarNet (Zarrar et al.,
IROS 2024) is a 9-layer PilotNet-style 1D CNN (5 conv + 4 FC) that ran on this
scan at the F1TENTH GP. The isolated claim is the missing 32-bin bottleneck;
GRU, PPO, proprio, and 20 Hz stay.

## Decision

1. Add `model.actor_type=tinylidarnet_gru`. Default remains `lidar_cnn_gru`
   with AdaptiveAvgPool1d(32).
   *Why:* artifacts are architecture-homogeneous; old 32-bin policies must not
   load as TinyLidarNet.
2. TinyLidarNetEncoder reads the raw 1081-beam scan. There is no adaptive pool.
   Its 256-d embedding concatenates with 16-d proprio and still feeds GRU 512
   and the tanh-squashed Gaussian head.
   *Why:* End2Race already showed dropping the GRU hurts safety; this arm is
   encoder-only.
3. Control stays 20 Hz. Plant actions stay 2-d.
   *Why:* this is not action-repeat and not a 10 Hz policy.

## Consequences

- Austin arm: `training/configs/austin_e2e_ppo_baseline_v6_tinylidarnet.json`.
- `sensor_racer` loads the graph through `actor_from_architecture`.

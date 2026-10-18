# 0043 — Sample public tracks with Gigaflow affine variants

- Status: Accepted
- Date: 2026-10-17
- Supplements: [0042](0042-sim-ready-v2-fitted-courtyard-plant.md)

## Context

Cusumano-Towner et al. 2025 ([arXiv 2502.03349v1](https://arxiv.org/html/2502.03349v1))
train on 128 affine variants of 8 maps. They do not publish the transform
matrices. This stack already loads
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) centerlines
and a single occupancy PNG per config. Personal maps (courtyard, courtyard_2,
IV_2026, IV_2026_SIM) are not in that public repo.

## Decision

1. Train Gigaflow on the 23 public racetrack directory names, never courtyard or
   IV. *Why:* those four are personal maps; a track is in only if the public
   repo contains it.
2. Each full env reset draws one public track and one variant id from this
   closed set: `identity`, `mirror` (reverse the centerline and swap left/right
   widths), `width_0.8`, `width_1.2`, `mirror_width_0.8`, `mirror_width_1.2`.
   *Why:* that is the paper's affine-variant idea with the transforms we can
   state; no extra rotations, shears, or occupancy warps.
3. Log the track name and variant id on resample. LiDAR uses the corridor
   polygon of the variant, not Austin's occupancy PNG. The car plant stays the
   frozen SIM_READY_V2 fit from `austin_e2e_ppo_baseline_v6.json`. *Why:* an
   occupancy map does not follow the affine corridor; the fit is not retuned.

## Consequences

`env.gigaflow_track_sampling` opts in. Parallel envs in one `WarpF1tenthEnv`
share the drawn layout; a new draw happens on a full reset. Courtyard and IV
stay available for other configs. No training job is started by this change.

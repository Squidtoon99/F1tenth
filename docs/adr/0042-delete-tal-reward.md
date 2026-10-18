# 0042 — Delete the trajectory-aided speed penalty

- Status: Accepted
- Date: 2026-10-18

## Context

`reward_scales.tal` scored
`scale * (1 - |track_speed - tal_v_ref| - |steer - steer_ref|)`, with
`tal_v_ref` defaulting to `expected_lap_speed_mps` (3.5). Combined with
progress, that two-sided speed match is a cruise target, not a racing
objective. Leaving the scale at 0 still lets a later config turn it back
on.

## Decision

1. Remove the TAL kernel block, `tal_scale` / `tal_v_ref` / `tal_kp_*`
   params, the `tal` extras term, and the `reward_scales.tal` config key.
   *Why:* a later JSON patch must not be able to re-enable the term.
2. Keep `expected_lap_speed_mps` only as the episode-horizon sizer.
   *Why:* it is not a reward.

## Consequences

`validate_config_patch` rejects `reward.reward_scales.tal` as an unknown
key, including `tal: 0`. Progress, collision, passing, and steering terms
are unchanged.

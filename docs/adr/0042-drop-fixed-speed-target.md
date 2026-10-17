# 0042 — Drop the fixed 3.5 m/s speed target

- Status: Accepted
- Date: 2026-10-16

## Context

`expected_lap_speed_mps = 3.5` started as an episode-horizon sizing constant.
TAL reused that same key as a two-sided speed target `v_ref` plus a Stanley
steering half. Combined with progress, the unique max of `progress + TAL` is
exactly 3.5 m/s. Flying eval also spawned at 3.5. The on-car safety envelope
is 7.0 m/s; observation `max_speed` stays 15. 7 m/s is a brake, not a spawn.

Steering-only TAL would still pull toward the scripted Stanley angle and does
not stand alone without the Evans speed term. Courtyard radii cannot take
3.5 m/s (ADR 0030).

## Decision

1. Remove the TAL reward, including the steering half. `reward.reward_scales.tal`
   stays 0 and non-zero values are rejected. There is no `tal_v_ref`. The only
   speed-related reward is centerline progress (`Δs`).
   *Why:* a constant speed target must not be reintroduced by flipping a scale.
2. Delete `expected_lap_speed_mps` and `episode_lap_multiplier`. Episode length
   is a frozen per-track timeout in `TRACK_EPISODE_HORIZON_S` (Austin 360.89 s,
   IV_2026_SIM 242.07 s, ECSS_Floor 105.16 s, courtyard_2 and Galaxy_1 60 s).
   Those seconds are a horizon only, not a reward and not a spawn. Policy and
   reward code must not read that table.
   *Why:* dropping 3.5 must not also shorten episodes in the A/B.
3. Start speeds are per-track, never 7 m/s. Standing eval is always rest.
   Flying eval uses `TRACK_FLYING_SPAWN_MPS` (courtyard_2 0, Galaxy_1 0.5,
   ECSS_Floor 1.0, IV_2026_SIM 1.5, Austin 2.0). Training resets use
   `TRACK_RESET_SPEED_MPS` (courtyard_2 0–1, Galaxy_1 0–1.5, ECSS_Floor 0–2.5,
   IV_2026_SIM 1–3, Austin 0–5). Unknown tracks spawn from rest / 0–1.
   The 64-spawn eval is not randomized.
   *Why:* courtyard is tight (~18 m; ADR 0030). Austin can roll faster than
   courtyard, but 2 m/s is a launch roll, not TAL 3.5 and not the safety brake.
4. Alpha-RPO `centerline_base_action` uses `opponent_target_speed` when no
   speed is passed. `MAX_SPEED` / `env.max_speed = 15` is unchanged. Scripted
   opponent pace is not an ego reward.

## Consequences

- New runs only. Existing TAL / 3.5 checkpoints and live jobs are untouched.
- Action-chunk config keeps `tal: 0`. Overlay JSON that previously set
  `tal: 0.2` now sets `0.0` so they still load.
- Opponent `opponent_target_speed` / range stay the scripted-opponent knobs;
  they are not a horizon constant.

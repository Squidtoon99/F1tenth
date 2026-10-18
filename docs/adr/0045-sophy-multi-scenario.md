# 0045 — GT Sophy multi-scenario PPO without slipstream

- Status: Accepted
- Date: 2026-10-17
- Supplements: [0042](0042-sim-ready-v2-fitted-courtyard-plant.md),
  [0043](0043-gigaflow-track-affine-sampling.md),
  [0044](0044-curvature-limited-scripted-opponent.md)

## Context

Wurman et al., Nature 2022 train GT Sophy on nearby 1v0 / 1v1 / 1v2 / 1v3 / 1v7
plus Sarthe skill scenes (eight-car grid, final chicane). This stack already
mixes solo flying/standing via `scenarios` in the trainer. It did not spawn more
than one extra car, did not tag transitions with a scenario id, and did not
stratify PPO minibatches. Slipstream is out of scope.

## Decision

1. Spawn up to seven extra cars (`MAX_OPPONENTS=7`, grid capped at 8 including
   ego) from the same Warp plant. Privileged opponent features stay the 8-dim
   nearest block; extra cars exist in physics and LiDAR only.
   *Why:* the paper's nearby set and 8-car grid, without changing the
   observation contract or adding slipstream.
2. Nearby scenes randomize along-track pose, speed, and gaps. The grid is a
   2-wide pack. The skill scene is the hardest corner of the drawn public
   track, in light traffic, not Sarthe's chicane.
   *Why:* F1TENTH has no Sarthe map in the public racetrack set.
3. Scripted centerline cars from [0044](0044-curvature-limited-scripted-opponent.md)
   cover every slot at or below the speed that follower can hold. Faster slots
   are a frozen V2 baseline-policy copy, stubbed to the hold speed until
   `env.v2_policy_ckpt` exists. Speed classes relative to ego pace are slower
   0.45, same 0.45, faster 0.10. No 3.5 m/s target.
   *Why:* the old 58 s checkpoint leaves the V2 plant in a few seconds.
4. Each transition stores `scenario_id`. PPO env minibatches round-robin the
   scenario groups so a skill keeps its share. Workers stay many Warp sims, one
   trainer. The plant and maps are SIM_READY_V2 plus the 23 public tracks.
   *Why:* Sophy used one replay table per scenario; we stay on PPO.

## Consequences

`training/configs/sophy_multi_scenario.json` is the opt-in mix. Courtyard and IV
stay out. No slipstream term is added. The long run is a Brev job after the CPU
smoke, not the laptop GPU and not beside the local V2 baseline.

# 0042 — Use the frozen SIM_READY_V2 fit as the training plant

- Status: Accepted
- Date: 2026-10-17
- Supplements: [0033](0033-reduced-bicycle-simulator.md); does not change
  [0028](0028-promote-80-40-current-plant.md) defaults for other scenarios

## Context

The reduced bicycle in [0033](0033-reduced-bicycle-simulator.md) still trains
on the unfitted prior (26.5/23.1 N, coast 0, steer gain 1.10714, lateral scale
11.0). SIM_READY_V2 froze a courtyard bag fit of that same plant. New Austin
and courtyard PPO runs should load those fitted scalars rather than the
default unfitted bicycle. Live run directories and checkpoints are not
converted.

## Decision

1. Set `training/configs/austin_e2e_ppo_baseline_v6.json` and
   `training/configs/courtyard_2_e2e_ppo.json` to the frozen SIM_READY_V2 fit:

   | Parameter | Value |
   | --- | ---: |
   | Drive force at 80 A | 21.4687749217 N |
   | Brake force at 40 A | 9.1391880671 N |
   | Constant coast loss | 4.8047812947 N |
   | Steering gain | 1.0511988660 |
   | Steering lag | 0.09 s, retained prior |
   | Lateral stiffness scale | 5.3293060900 |

   *Why:* this is the selected free/full replay estimate. Austin training,
   including opponents-notal, uses `--config
   training/configs/austin_e2e_ppo_baseline_v6.json`. Courtyard uses
   `--config training/configs/courtyard_2_e2e_ppo.json`.
2. Leave `DEFAULT_CONFIG` and Galaxy on the unfitted 0033 prior. Do not
   convert old checkpoints or edit the live `F1tenth-opponents-notal` tree.
   *Why:* already-running jobs keep the plant they started with; new runs
   pick the fit from the Austin/courtyard JSON.
3. Do not edit the fitted numbers to chase a better score, and do not treat
   this as a physical-deployment certificate.
   *Why:* SIM_READY_V2 already records that angular/radius parity still fails.

## Consequences

New Austin PPO runs that pass `--config
training/configs/austin_e2e_ppo_baseline_v6.json` load the fitted plant
through `VehicleParams.from_config`. Courtyard PPO does the same via its
config. `DEFAULT_CONFIG` and Galaxy keep 26.5/23.1 N, coast 0, gain 1.10714
and lateral scale 11.0. Deployment configuration is unchanged.

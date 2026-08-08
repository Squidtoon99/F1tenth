# Dense-traffic experiment (max 8 vs 10 vs 12)

Compares capacity-aware dense worlds at `max_agents_per_world ∈ {8,10,12}`
**without** changing production/default YAML (`configs/default.yaml`,
`configs/production_h100.yaml` stay at 8).

## What is preserved

- `solo_world_fraction: 0.05`
- Training density bins: `sparse, medium, dense`
- Eval suites still force solo=1 and head-to-head=`pair`→2 cars

## Local CPU validation

Refuses CUDA so it cannot contend with a live `gigaflow view --device cuda`.

```bash
cd gigaflow
# Optional: rebuild capacity-aware atlas variants (never overwrites production cache)
CUDA_VISIBLE_DEVICES= python tools/dense_traffic_experiment.py \
  --rebuild-atlases \
  --output-dir outputs/dense_traffic_experiment
```

Artifacts:

- `outputs/dense_traffic_experiment/dense_traffic_summary.json` — metrics + gates
- `outputs/dense_traffic_experiment/h100_queue.json` — exact H100 commands
- `~/.cache/gigaflow/tracks_dense_max{8,10,12}/` — capacity-rebuilt atlases

## Promotion gates (candidate vs max-8 baseline)

| Gate | Intent |
| --- | --- |
| `cars_per_world_lift` | Dense worlds actually denser (≥1.05× cars) |
| `close_pair_lift` | More close-opponent interactions |
| `spawn_rejects_bounded` | Reject rate ≤15% and ≤1.75× baseline |
| `occlusion_bounded` | LiDAR short-range proxy ≤1.8× baseline |
| `track_visibility_positive` | Some long-range track returns remain |
| `collision_bounded` / `oob_bounded` | ≤2.25× baseline per km |
| `throughput_ok` | Sim ticks/s ≥0.70× baseline |
| `vram_under_budget` | Slot-scaled H100 estimate ≤72 GiB |
| `ppo_finite` / `ppo_kl_bounded` | Short-run PPO health |
| `solo_frac_preserved` / `head_to_head_two_cars` | Mix unchanged |

## Recommendation

**Try max 10 first**, then 12. Reasons: smaller jump from production-8, lower
equal-world VRAM risk (~154 GiB est at 1024×10 vs ~186 GiB at 1024×12), and
cleaner occlusion/spawn trade-offs. Prefer the **shipped, corrected world
counts** (`dense_traffic_h100_max{8,10,12}.yaml`: 448/352/288) — the
`slot_matched_worlds` values below (`≈832` for 10, `≈704` for 12) still exceed
budget at these agent counts and are kept only as a slot-accounting reference,
not a VRAM-safe recommendation.

Local CPU validation (synthetic oval + capacity-rebuilt atlases; CUDA viewer
left untouched) already saw denser realized cars/world at 10 and 12 with
bounded spawn rejects / PPO health. Equal-worlds 1024×8/10/12 all exceed the
soft 72 GiB VRAM gate (~122/154/186 GiB); only the shipped, reduced world
counts (448/352/288) fit.

`dense_traffic.estimate_h100_vram_gib` delegates to
`config.estimate_memory_bytes` — the model validated against measured CUDA
peak — so the VRAM figures above and the shipped world counts in
`dense_traffic_h100_max{8,10,12}.yaml` now come from the same model and cannot
silently disagree.

## H100 queue

See `h100_queue.json` after the tool runs. Do not attach to an active production
trainer; use a fresh run directory. Artifact schema per job:

- `metrics_*.json`, `actor_*.pt`, `config.json`
- `eval_gate/` (solo / head_to_head / dense)
- `dense_traffic_summary.json` (local validation report)

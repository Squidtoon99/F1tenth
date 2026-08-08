# Benchmarks

Prefer the CLI when possible:

```bash
cd gigaflow
gigaflow benchmark --config configs/smoke.yaml --mode sim --steps 50 --device cpu
gigaflow benchmark --config configs/smoke.yaml --mode learner --steps 1 --device cpu
```

## N-agent simulator (legacy script)

```bash
python benchmarks/bench_n_agent_sim.py --steps 50 --device cpu
```

Reports world ticks/s and agent transitions/s for the production Warp path.

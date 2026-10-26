# docs/

- [getting-started.md](getting-started.md) — start here: what the repo is, first-time
  setup (dev container + training venv), gym sim, and contributing a ROS node.
- [context.md](context.md) — domain primer: what F1TENTH is, the GT Sophy RL
  lineage, and the autonomy stack / core problems.
- [development.md](development.md) — setup, build, run, training, and the branching
  model.
- [deployment.md](deployment.md) — image build -> snapshot -> Jetson -> per-car config.
- [observation_contract.md](observation_contract.md) — the shared 390-dim obs/action
  contract and the C++/Python parity policy.
- [adr/](adr/) — architecture decision records (timeline + current state).

Nearby runbooks (not under `docs/`):
[`../training/README.md`](../training/README.md),
[`../sim/README.md`](../sim/README.md),
[`../analysis/README.md`](../analysis/README.md),
[`../deploy/README.md`](../deploy/README.md),
[`../calibration/README.md`](../calibration/README.md).

For the high-level design, see [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

# docs/

- [getting-started.md](getting-started.md) — start here: what the repo is, first-time
  setup (build or pull the seed images), and contributing a ROS node end to end.
- [context.md](context.md) — domain primer: what F1TENTH is, the GT Sophy RL
  lineage, and the autonomy stack / core problems.
- [development.md](development.md) — setup, build, sim, training, pitfalls, and the
  branching model.
- [deployment.md](deployment.md) — image build -> snapshot -> Jetson -> per-car config.
- [observation_contract.md](observation_contract.md) — the shared obs/action contract
  and the C++/Python parity policy.
- [adr/](adr/) — architecture decision records (timeline to the current stack).

Operational runbooks that live next to the code:

- [`../training/README.md`](../training/README.md) — train a new policy (Warp QR-SAC).
- [`../sim/README.md`](../sim/README.md) — gym closed-loop eval vs Warp training sim.
- [`../calibration/README.md`](../calibration/README.md) — sim-to-real / VESC priors.

For the high-level design, see [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

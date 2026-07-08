# AGENTS.md

Operating manual for AI coding agents (and new humans) working in this repo. This
file is the single source of truth for *what must be true* when you change code;
the linked docs explain *why*. Keep it short and current; link out rather than
duplicate.

## Domain context

F1TENTH is an open 1/10th-scale autonomous racing platform (Ackermann car with
LiDAR + IMU + VESC on an NVIDIA Jetson). This monorepo hosts two racing stacks
that share perception, mapping, and localization: a **classical** pipeline
(raceline optimization + pure-pursuit) and an **end-to-end RL** policy. The RL
side follows the GT Sophy approach (Wurman et al., *Nature* 2022): a **QR-SAC**
trainer ([`training/qrsac/`](training/qrsac/)) with a hand-designed observation of
car state, track geometry ahead, and opponent-relative features
([`libs/f1tenth_contract`](libs/f1tenth_contract/)), and sim-to-real via domain
randomization. Everything targets ROS 2 Humble and runs in Docker. Full primer:
[`docs/context.md`](docs/context.md).

## Operating principles

- Verify before acting; prefer the conservative read of an ambiguous request.
- Ask the human over guessing when a decision is genuinely theirs (scope,
  destructive actions, architectural trade-offs).
- You own every line of the diff. Keep diffs small and reviewable.
- Don't ship speculatively — no code for hypothetical future needs.
- For docs/config, read the existing content first and make targeted edits; never
  wholesale-rewrite a file.

## Project layout — where things go

| Path | What lives here |
| --- | --- |
| [`src/`](src/) | The single colcon workspace: all ROS 2 packages, grouped by domain |
| [`libs/`](libs/) | Cross-language shared libraries (the observation/action contract) |
| [`training/`](training/) | RL training (pure Python; never built by colcon, never in the car image) |
| [`sim/`](sim/) | Simulator integration |
| [`deploy/`](deploy/) | Docker/Apptainer images, per-car config, deploy scripts |
| [`tools/`](tools/) | Build/dev helper scripts and lint configs |
| [`docs/`](docs/) | Architecture, development workflow, deployment, ADRs |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module groups under `src/`.

## Golden rules

- **Keep it simple.** Do not add abstractions, layers, config knobs, or helper
  indirection until a second real caller needs them.
- **No mocks.** Test against real modules and real behavior. Don't fake, stub, or
  skip a dependency to avoid exercising it — write a test that genuinely runs
  against the real thing.
- **Match existing patterns** before inventing new ones; prefer editing an
  existing file over adding one.
- **Respect ownership boundaries** in [`CODEOWNERS`](CODEOWNERS). Coordinate before
  touching vendored `src/vehicle/` code (see
  [`src/vehicle/VENDORED.md`](src/vehicle/VENDORED.md)).

## Simplicity & comment discipline

- Don't add docstrings, comments, or type annotations to code you didn't change.
- No ticket/issue IDs in code or comments (a single explicit follow-up `TODO`
  reference is the only exception).
- No historical narration in comments ("previously…", "extracted from…", "now does
  X instead of Y"). Comments describe the code as it is now.
- No near-duplicate comments across related symbols; no verbose docstrings that
  just restate the signature; no unexplained jargon or acronyms that don't already
  appear in the surrounding code.

## Boundaries

| Tier | Rules |
| --- | --- |
| **Always** | Build (`tools/build.sh`) + test (`tools/test.sh`) + lint before proposing a change; keep the diff confined to the module group in scope. |
| **Ask first** | Changing the observation/action contract; editing vendored `src/vehicle/**`; adding a dependency or a new ROS package; altering CI or deploy image definitions. |
| **Never** | Commit heavy artifacts (weights/rosbags/maps/build trees); add company or internal product names; skip/disable tests or lint to go green; self-merge. |

## Build & test

Everything runs inside the dev container (ROS 2 sourced). See
[`docs/development.md`](docs/development.md).

```bash
./tools/dev.sh up                          # build + enter the dev container
./tools/build.sh                           # colcon build everything (--symlink-install)
./tools/build.sh -t control                # build just one module group
./tools/test.sh                            # unit tests + ament lint (vendored pkgs skipped)
./tools/test.sh --packages-select f1tenth_rl_agent   # scope to one package
```

Vendored upstream packages are **built** (our code depends on them) but **not
tested/linted**.

## Testing philosophy

- Real-dependency tests over mocks: exercise the actual module or functionality.
  Don't fake behavior or skip a test to dodge a heavy dependency — set up the real
  dependency so the test genuinely runs.
- Duplicated logic is fenced by a **parity test** (e.g. the Python/C++ observation
  mirror).
- Deterministic seeds (`np.random.default_rng(0)`). CPU-only — there is no GPU CI.

## Code quality & lint

Python is checked by [`.flake8`](.flake8) (max line length 99); C++ by
[`.clang-format`](.clang-format); CI runs `ament_lint`. Keep functions small and
pure where practical.

## Contract change protocol + blast radius

The observation/action layout is a single source of truth in
[`libs/f1tenth_contract`](libs/f1tenth_contract/) with one intentional
duplication: a C++ mirror in
[`observation_layout.hpp`](src/common/f1tenth_common/include/f1tenth_common/observation_layout.hpp).
When you change it:

1. Update the Python contract.
2. Update the C++ mirror.
3. Keep the parity test green
   ([`test_obs_parity.cpp`](src/common/f1tenth_common/test/test_obs_parity.cpp)).

Before changing **any** shared surface (the contract, `f1tenth_interfaces`
messages, `f1tenth_common`), enumerate its downstream consumers — training,
`racing_rl`, and the C++ vehicle node — and verify each. Trace the blast radius
while writing the change, not after. Details:
[`docs/observation_contract.md`](docs/observation_contract.md).

## Documenting work

- **Architectural / significant decisions** -> add a new numbered ADR under
  [`docs/adr/`](docs/adr/) (never rewrite history — supersede). Use
  [`docs/adr/0000-template.md`](docs/adr/0000-template.md). Write an ADR when a
  change alters structure, a public interface/contract, the build/deploy shape, or
  accepts a non-obvious trade-off.
- **Routine work** -> a clear PR (the `.github/PULL_REQUEST_TEMPLATE.md` guides
  this) plus commit messages in the repo's `scope: summary` convention (e.g.
  `contract: …`, `training: …`, `build: …`). See
  [`docs/development.md`](docs/development.md).

## Guardrails

- No company or internal product/project names anywhere (public repo).
- Never commit heavy artifacts — weights, rosbags, outputs, maps, and build trees
  are gitignored; keep them that way.
- `develop` is the trunk; branch as `feature/<ticket>-<slug>` and PR back into it.

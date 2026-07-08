# racing_algo/

Algorithmic racing stack composition. This group mostly wires together shared
modules rather than owning heavy logic:

- localization (`src/localization/`)
- planning / raceline (`src/planning/`)
- control / pure pursuit (`src/control/`)

- `f1tenth_racing_algo/launch/algo.launch.py` — brings the classical pipeline up.

Developed by the algorithmic-racing team independently of `racing_rl/`, but sharing
mapping/localization/perception.

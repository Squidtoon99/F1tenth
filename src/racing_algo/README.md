# racing_algo/

Composition and bringup for the classical (algorithmic) racing stack.

- `f1tenth_racing_algo/` — `algo.launch.py` starts the selected controller from
  `f1tenth_control` plus the RL deadman gate. Localization is started by
  `race.launch.py` (shared with the RL stack).

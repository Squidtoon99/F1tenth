# 0017 — Tighten opponent spawn gaps for sustained 1v1 pressure

- Status: Accepted
- Date: 2026-07-26

## Context

Fixed-champion 1v1 training uses uniform longitudinal spawn gaps between ego and
opponent on reset. The default range was 3–80 m with a 40 m-ahead / 20 m-behind
opponent observation gate. At 4096+ parallel envs and ego speeds in the 5–6 m/s
band, wide spawns plus slower opponents decouple most pairs outside the gate;
`metric/opponent_presence` fell from ~40% early to ~19% late in a 600M run even
though the champion checkpoint drives ~5.4 m/s solo. That late-regime pressure
drop risks the back half of multi-billion-transition runs training in effective
solo conditions, which prior 1v0 arms showed regresses speed and raises
not-moving terminations.

## Decision

1. Change the repo default spawn gap ceiling from 80 m to **35 m**, keeping the
   3 m floor. At 5–6 m/s ego with slow opponents, probe rollouts sustain ~98%
   step-mean presence versus ~40% for 3–80 m; spawn-time visibility reaches 100%
   because no sampled gap exceeds the ahead gate.
2. **Pin 3–80 m explicitly** in the preregistered `numenv8192-deep4m-a001`
   diagnostic JSON so the in-flight parallel-env replay-depth comparison is not
   confounded. Future runs inherit 3–35 m unless a patch overrides it.

*Why 35 m:* it keeps all ahead spawns inside the 40 m gate with margin, preserves
a spread of gaps (≈47% of samples still exceed 20 m), and avoids the contact /
rear-end distortion risk of pushing the floor above 3 m or the ceiling below
~25 m.

## Consequences

Training distribution for new long-horizon and default runs carries higher
sustained opponent presence in the 5–6 m/s regime. Checkpoints from runs before
this change are not directly comparable on opponent-interaction statistics
without noting the gap range. The 8192-deep diagnostic arm remains on 3–80 m
until it completes.

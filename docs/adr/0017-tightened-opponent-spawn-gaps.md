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

The replay buffer stratifies samples 50/50 into windows where the stored critic
opponent block `[384:392)` was non-zero versus all-zero, but it does **not**
synthetically zero opponent features at sample time. Solo-like learner experience
therefore comes only from environment steps where the opponent was actually
outside the observation gate.

## Decision

1. Change the repo default spawn gap ceiling from 80 m to **58 m**, keeping the
   3 m floor. Probe rollouts at ego 5.5 m/s with a 0.5 m/s opponent sustain
   **~70%** step-mean presence (reset ~66%) versus ~40% for 3–80 m — coupled
   enough to beat the late-run 19% collapse while preserving real solo running
   and keeping passing/collision reward rates nearer the validated 600M run.
2. **Pin 3–80 m explicitly** in the preregistered `numenv8192-deep4m-a001`
   diagnostic JSON so the in-flight parallel-env replay-depth comparison is not
   confounded. Future runs inherit 3–58 m unless a patch overrides it.

*Why 58 m:* it lands in the 65–75% presence band at 5–6 m/s ego speed. A 50 m
ceiling (still well inside the ahead gate) measured ~86% presence — too close to
always-on 1v1 — while retaining a spread of gaps (≈64% of samples exceed 20 m).

## Consequences

Training distribution for new long-horizon and default runs carries higher
sustained opponent presence in the 5–6 m/s regime without near-total coupling.
Checkpoints from runs before this change are not directly comparable on
opponent-interaction statistics without noting the gap range. The 8192-deep
diagnostic arm remains on 3–80 m until it completes.

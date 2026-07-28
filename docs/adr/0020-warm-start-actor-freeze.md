# 0020 — Warm-start actor freeze for critic warm-up

- Status: Accepted
- Date: 2026-07-27

## Context

`--init-ckpt` restores actor weights and the actor observation normalizer from a
sensor policy artifact; critics and their normalizer start fresh. The first SAC
policy updates are computed against randomly initialized value estimates and can
destroy a pretrained actor within a few thousand gradient steps — before the
critic has seen enough replay to produce meaningful targets.

QR-SAC uses a fixed entropy coefficient (`model.alpha`), not automatic entropy
tuning. A near-deterministic pretrained policy still receives an entropy bonus
in the policy loss; starting `alpha` low in config limits that push.

`replay_full_reinit` (Lee et al. 2025) resets actor, critics, and optimizers once
when the replay buffer first fills. That would wipe a warm-started actor roughly
two minutes into a run and must be disabled for warm-start probes.

## Decision

1. Add `model.actor_freeze_transitions` (default `0`). While
   `env_transitions < actor_freeze_transitions`, QR-SAC runs critic-only updates:
   the policy forward pass still produces bootstrap targets, but the actor
   optimizer step is skipped.
2. Warm-start probe config sets `replay_full_reinit: false`, `alpha: 0.001` (10×
   below default), and `actor_freeze_transitions: 5_000_000` (~2M critic-only
   steps after the 3M replay buffer fills).
3. No behavioral-cloning anchor term — the freeze window is sufficient for the
   short probe; a BC term would be justified only if freeze alone fails to
   protect the actor.

## Consequences

- Warm-start runs require explicit config (`replay_full_reinit: false`, nonzero
  freeze window when critics start cold). Default cold-start behavior is unchanged.
- `actor_freeze_transitions` is logged at startup and when the actor unfreezes.
- If the probe shows degradation even with freeze + low alpha, the reward stack
  is the likely culprit, not missing pretrain machinery.

# 0014 — Use the paper-literal wall-event scale

- Status: Accepted
- Date: 2026-07-24
- Supersedes: [0013](0013-first-footprint-wall-contact.md) reward scale only

## Context

ADR 0013 interpreted the wall coefficient as a per-metre rate and multiplied it
by the control period. The controlled follow-up experiment instead needs the
paper-literal event value while preserving the first-footprint contact event and
all other training variables.

## Decision

On the terminating first-footprint wall-contact transition, mask progress and
apply

\[
r_{\mathrm{wall}}=-20\,\lVert v_{xy}\rVert.
\]

The event values are -20, -40, and -100 at 1, 2, and 5 m/s. The reward does not
scale with the control period.

## Consequences

Wall geometry, termination, and the remaining Lee training protocol are
unchanged. Reward totals are not directly comparable with runs using ADR 0013's
time-scaled interpretation.

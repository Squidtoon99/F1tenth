# 0013 — Terminate at first projected wall contact

- Status: Accepted
- Date: 2026-07-23
- Supersedes: [0012](0012-lee-core-and-fixed-opponents.md) course-limit reward and
  termination clause only

## Context

The canonical sensor trainer stacked three course-limit costs: a fixed first
crossing penalty, a continuous quadratic off-course penalty, and a quadratic
terminal impact when the full car left the course. That stack did not represent
the wall/barrier atom in Lee et al. and rewarded reducing speed
disproportionately.

The simulator already projects the oriented rectangular footprint into the
local track-normal direction. This gives one deterministic geometric event for
the mapped walls without adding a selectable reward or termination mode.

## Decision

Terminate on the first transition where either projected footprint edge reaches
or crosses its mapped left or right wall. The lateral footprint extent is

\[
h_\perp =
\frac{L}{2}|\sin(\psi-\psi_{\mathrm{track}})|
+\frac{W}{2}|\cos(\psi-\psi_{\mathrm{track}})|.
\]

Contact occurs when \(e_y+h_\perp\geq w_l\) or
\(e_y-h_\perp\leq-w_r\). On that same transition, mask progress and apply only

\[
r_{\mathrm{wall}}=-20\,\Delta t\,\lVert v_{xy}\rVert.
\]

Here \(\Delta t\) is seconds, speed is metres per second, and the coefficient is
20 per metre, producing a dimensionless reward. At the 0.1 s control period the
reward is -2, -4, and -10 at 1, 2, and 5 m/s.

Remove the fixed boundary reward, continuous quadratic OOB reward, terminal OOB
impact, full-car-out behavior, and their compatibility configuration. Preserve
the existing OOB terminal flag and no-bootstrap semantics.

## Consequences

Wall contact is now one canonical physical-wall treatment rather than a matrix
of track-limit modes. Reward totals are not directly comparable with earlier
runs, and first-contact events replace full-car-out terminations as the
behavioral comparison metric. ADR 0012's sensor architecture and fixed-opponent
decisions remain in force.

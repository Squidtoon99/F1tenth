<!-- See AGENTS.md and docs/development.md for conventions. -->

## What & why

<!-- What does this change do, and why? Focus on the "why". -->

## Module group(s) touched

<!-- e.g. control, racing_rl, training, libs/f1tenth_contract -->

## Downstream impact

<!-- If a shared surface changed (the obs/action contract, f1tenth_interfaces
messages, f1tenth_common), list the downstream consumers you checked
(training, racing_rl, the C++ vehicle node) and how you verified them.
Write "none — no shared surface touched" otherwise. -->

## Test evidence

<!-- Paste the relevant output. Keep the diff confined to the paths above. -->

```
# tools/build.sh
# tools/test.sh
```

## Checklist

- [ ] Builds and tests pass (`tools/build.sh`, `tools/test.sh`); lint clean
- [ ] No mocks added — tests use real modules (skip via `importorskip` when a dep is missing)
- [ ] No new unjustified abstraction (KISS)
- [ ] Comment discipline followed (no narration, no ticket IDs, nothing added to untouched code)
- [ ] ADR added under `docs/adr/` if this is an architectural decision
- [ ] No heavy artifacts committed (weights, rosbags, maps, build trees)
- [ ] No company or internal product/project names

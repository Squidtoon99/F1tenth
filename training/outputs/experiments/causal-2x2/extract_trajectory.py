#!/usr/bin/env python3
"""Extract a 0.5M-bucketed speed trajectory from a standalone_trainer run.log."""
import json
import re
import sys

TICK_RE = re.compile(r"transitions=(\d+) replay_inserts")
SPEED_RE = re.compile(r"env: speed=([\d.]+)")


def extract(path: str) -> dict[str, float]:
    buckets: dict[int, float] = {}
    transitions = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = TICK_RE.search(line)
            if m:
                transitions = int(m.group(1))
                continue
            m = SPEED_RE.search(line)
            if m and transitions is not None:
                bucket = round(transitions / 500_000) * 0.5
                buckets[bucket] = float(m.group(1))
    return {f"{k:g}": v for k, v in sorted(buckets.items())}


if __name__ == "__main__":
    run_log, run_id = sys.argv[1], sys.argv[2]
    print(json.dumps({run_id: extract(run_log)}, indent=2))

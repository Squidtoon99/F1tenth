"""Extract the (transitions, speed) trajectory from a standalone_trainer run.log.

Usage: .venv/bin/python extract_trajectory.py <run.log> [--bucket-m 2]

Independent re-extraction tool for verifying reported champion trajectories
(e.g. 642a7a80) from raw log text rather than trusting summarized numbers.
"""
import argparse
import re
import sys
from collections import defaultdict

TICK_RE = re.compile(r"transitions=(\d+) .*?mean_ep_reward=")
SPEED_RE = re.compile(r"\bspeed=([\-0-9.]+)\b")


def parse(path):
    rows = []
    pending_transitions = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = TICK_RE.search(line)
            if m:
                pending_transitions = int(m.group(1))
                continue
            if pending_transitions is not None and "env: speed=" in line:
                sm = SPEED_RE.search(line)
                if sm:
                    rows.append((pending_transitions, float(sm.group(1))))
                pending_transitions = None
    return rows


def bucket(rows, bucket_size):
    buckets = defaultdict(list)
    for t, s in rows:
        buckets[t // bucket_size].append(s)
    out = []
    for b in sorted(buckets):
        vals = buckets[b]
        out.append(
            (b * bucket_size, sum(vals) / len(vals), min(vals), max(vals), len(vals))
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log")
    ap.add_argument(
        "--bucket-m", type=float, default=2.0, help="bucket width in millions of transitions"
    )
    args = ap.parse_args()

    rows = parse(args.run_log)
    if not rows:
        print("No (transitions, speed) rows parsed - check log format", file=sys.stderr)
        sys.exit(1)

    bucket_size = int(args.bucket_m * 1_000_000)
    bucketed = bucket(rows, bucket_size)

    print(f"parsed {len(rows)} samples, last transitions={rows[-1][0]}")
    print(f"{'bucket_start_M':>15} {'mean_speed':>10} {'min':>8} {'max':>8} {'n':>6}")
    for b_start, mean_v, min_v, max_v, n in bucketed:
        print(
            f"{b_start / 1e6:15.1f} {mean_v:10.3f} {min_v:8.3f} {max_v:8.3f} {n:6d}"
        )


if __name__ == "__main__":
    main()

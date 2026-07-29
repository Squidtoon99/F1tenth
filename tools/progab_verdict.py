#!/usr/bin/env python3
"""Analyze progab from-noise A/B at 600M for verdict + interpretability."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))
from run_log_tools import filter_log_text, filter_parsed_rows  # noqa: E402

PCPLUS600_600M = 5.38
PCPLUS600_MIN = 5.05
CONTROL_PROGRESS_BASELINE = 0.510
THR_MPS = 5.0


def parse_log(path: Path) -> list[dict]:
    text = filter_log_text(path.read_text(errors="replace"), path)
    lines = text.splitlines()
    rows: list[dict] = []
    for i, line in enumerate(lines):
        tm = re.search(
            r"transitions=(\d+).*episode_lifespan=([\d.]+)s.*\(n=(\d+)\)",
            line,
        )
        if not tm:
            continue
        t = int(tm.group(1))
        row = {
            "t": t,
            "life": float(tm.group(2)),
            "n": int(tm.group(3)),
            "speed": None,
            "progress": None,
            "progress_ds": None,
        }
        for j in range(i + 1, min(i + 8, len(lines))):
            sm = re.search(r"env: speed=([\d.]+)", lines[j])
            pds = re.search(r"progress_ds=([\d.]+)", lines[j])
            pm = re.search(r"progress=([\d.]+)", lines[j])
            if sm:
                row["speed"] = float(sm.group(1))
            if pds:
                row["progress_ds"] = float(pds.group(1))
            if pm and "rewards:" in lines[j]:
                row["progress"] = float(pm.group(1))
        rows.append(row)
    return filter_parsed_rows(rows, path, trans_key="t")


def nearest(rows: list[dict], target: int) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["t"] - target))


def window_mean(rows: list[dict], lo: int, hi: int, key: str) -> float | None:
    vals = [r[key] for r in rows if lo <= r["t"] <= hi and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def frac_above_speed(rows: list[dict], lo: int, hi: int, thr: float) -> float | None:
    vals = [r["speed"] for r in rows if lo <= r["t"] <= hi and r["speed"] is not None]
    if not vals:
        return None
    return sum(v >= thr for v in vals) / len(vals)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: progab_verdict.py <control.log> <treatment.log>", file=sys.stderr)
        return 2
    control = parse_log(Path(sys.argv[1]))
    treat = parse_log(Path(sys.argv[2]))
    horizon = 600_000_000
    bucket_lo, bucket_hi = 550_000_000, 600_000_000

    ca = window_mean(control, bucket_lo, bucket_hi, "speed")
    cb = window_mean(treat, bucket_lo, bucket_hi, "speed")
    la = window_mean(control, bucket_lo, bucket_hi, "life")
    lb = window_mean(treat, bucket_lo, bucket_hi, "life")
    pa = window_mean(control, bucket_lo, bucket_hi, "progress")
    pb = window_mean(treat, bucket_lo, bucket_hi, "progress")
    fa = frac_above_speed(treat, bucket_lo, bucket_hi, THR_MPS)
    fb = frac_above_speed(treat, bucket_lo, bucket_hi, THR_MPS)
    # control fraction for comparison
    fca = frac_above_speed(control, bucket_lo, bucket_hi, THR_MPS)

    print("=== control validity @600M ===")
    print(f"  speed_50M_mean={ca:.3f} m/s  ref_pcplus600={PCPLUS600_600M}  min_gate={PCPLUS600_MIN}")
    print(f"  lifespan_mean={la:.1f}s")
    print(f"  progress_mean={pa:.3f}")
    valid = ca is not None and ca >= PCPLUS600_MIN
    print(f"  baseline_valid={valid}")

    print("=== A/B @600M (50M bucket mean 550-600M) ===")
    print(f"  control: speed={ca:.3f} life={la:.1f} progress={pa:.3f} frac>={THR_MPS}={fca:.1%}" if ca else "  control: NO_DATA")
    if cb:
        print(f"  treat:   speed={cb:.3f} life={lb:.1f} progress={pb:.3f} frac>={THR_MPS}={fb:.1%}")
    else:
        print("  treat: NO_DATA")

    void = False
    if pb is not None and pa is not None and abs(pb - pa) < 0.02:
        void = True
    if fb is not None and fb < 0.05:
        void = True

    if not valid:
        verdict = "INVALID_BASELINE"
    elif void:
        verdict = "VOID"
    elif cb and ca:
        if cb >= 5.85 and lb and lb >= 200 and (cb >= ca + 0.10 or (lb >= la + 30)):
            verdict = "TREATMENT_WINS"
        elif cb < 5.20 or (lb and lb < 120):
            verdict = "HARM"
        elif abs(cb - ca) < 0.10 and abs(lb - la) < 30:
            verdict = "NULL"
        elif cb > ca and lb and la and lb < 150 and la >= 180:
            verdict = "HARM"
        else:
            verdict = "MIXED"
    else:
        verdict = "INCOMPLETE"

    print(f"=== verdict={verdict} void={void} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Physics substep cadence fidelity vs 20×0.005 s reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gigaflow_f1tenth.sim.stages import MAX_PHYSICS_SUBSTEPS

ROOT = Path(__file__).resolve().parents[1]


def _load_cadence_bench():
    path = ROOT / "tools" / "bench_physics_cadence.py"
    spec = importlib.util.spec_from_file_location("bench_physics_cadence", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_max_physics_substeps_bound():
    assert MAX_PHYSICS_SUBSTEPS == 20


def test_cadence_fidelity_selects_passing_interval():
    mod = _load_cadence_bench()
    report = mod.run_fidelity(device="cpu", steps=30, worlds=1, agents=2)
    assert report["ok"]
    assert report["selected"] is not None
    selected = int(report["selected"]["control_interval"])
    assert selected in {4, 5, 10, 20}
    by_iv = {c["control_interval"]: c for c in report["candidates"]}
    assert by_iv[20]["passed"]
    passing = [c["control_interval"] for c in report["candidates"] if c["passed"]]
    assert selected == min(passing)

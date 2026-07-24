"""Fixed-opponent / solo benchmark helpers."""

from __future__ import annotations

import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = TRAINING_DIR / "analysis"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from tournament import race_metrics_from_rollout  # noqa: E402


def test_race_metrics_from_rollout_aggregates_terms():
    import torch

    extras = [
        {
            "metrics": {
                "progress_ds": torch.tensor([0.5]),
                "oob_mask": torch.tensor([1.0]),
                "car_collision": torch.tensor([0.0]),
            },
            "rewards": {"terms": {}},
            "termination": {},
        },
        {
            "metrics": {
                "progress_ds": torch.tensor([0.25]),
                "oob_mask": torch.tensor([0.0]),
                "car_collision": torch.tensor([1.0]),
            },
            "rewards": {"terms": {}},
            "termination": {},
        },
    ]
    metrics = race_metrics_from_rollout(extras)
    assert metrics["progress_m"] == 0.75
    assert metrics["oob_steps"] == 1.0
    assert metrics["collisions"] == 1.0
    assert metrics["lifespan_steps"] == 2.0

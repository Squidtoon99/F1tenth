"""Shared pytest setup for the gigaflow suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_global_rng():
    """Seed the process-global RNGs so no test inherits another test's stream.

    Actor initialization, action sampling, and minibatch permutation all draw
    from the global torch generator, so without this a test's numeric outcome
    depends on which tests ran before it.
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(0)

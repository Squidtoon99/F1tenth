"""Running sensor normalizer: Welford correctness, identity, clipping."""

from __future__ import annotations

import numpy as np
import torch

from gigaflow_f1tenth.normalization import SensorNormalizer


def test_fresh_normalizer_is_identity():
    norm = SensorNormalizer(dim=8)
    obs = torch.randn(4, 8)
    assert torch.equal(norm(obs), obs)


def test_update_matches_numpy_mean_std_across_batches():
    rng = np.random.default_rng(0)
    dim = 16
    norm = SensorNormalizer(dim=dim)
    batches = [rng.normal(loc=3.0, scale=2.0, size=(5, dim)).astype(np.float32) for _ in range(6)]
    for b in batches:
        norm.update(torch.as_tensor(b))
    all_obs = np.concatenate(batches, axis=0)
    expected_mean = all_obs.mean(axis=0)
    expected_std = all_obs.std(axis=0)
    assert np.allclose(norm.mean.numpy(), expected_mean, atol=1e-3)
    assert np.allclose(norm.std().numpy(), expected_std, atol=1e-3)
    assert float(norm.count.item()) == float(all_obs.shape[0])


def test_normalize_is_clipped_to_unit_interval():
    norm = SensorNormalizer(dim=4, clip_sigma=2.0)
    norm.update(torch.zeros(100, 4))
    norm.update(torch.ones(100, 4) * 1e-3)  # tiny nonzero variance
    huge = torch.full((1, 4), 1000.0)
    out = norm(huge)
    assert torch.all(out.abs() <= 1.0 + 1e-6)


def test_normalize_centers_and_scales_within_bounds():
    norm = SensorNormalizer(dim=1, clip_sigma=5.0)
    rng = np.random.default_rng(1)
    obs = rng.normal(loc=10.0, scale=3.0, size=(4096, 1)).astype(np.float32)
    norm.update(torch.as_tensor(obs))
    probe = torch.as_tensor([[10.0], [13.0], [7.0]], dtype=torch.float32)
    out = norm(probe).numpy()
    # Mean maps near zero; +/-1 std maps to a small positive/negative fraction
    # of the clip range (1/clip_sigma), not clipped since 1 std << 5 std.
    assert abs(out[0, 0]) < 0.05
    assert 0.0 < out[1, 0] < 0.3
    assert -0.3 < out[2, 0] < 0.0


def test_to_dict_load_dict_roundtrip():
    norm = SensorNormalizer(dim=6)
    norm.update(torch.randn(10, 6) * 5.0 + 2.0)
    state = norm.to_dict()
    restored = SensorNormalizer(dim=6)
    restored.load_dict(state)
    assert torch.allclose(restored.mean, norm.mean)
    assert torch.allclose(restored.m2, norm.m2)
    assert float(restored.count.item()) == float(norm.count.item())
    obs = torch.randn(3, 6)
    assert torch.allclose(restored(obs), norm(obs))

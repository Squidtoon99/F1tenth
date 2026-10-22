"""VRAM-accounting and MaxPool64/512 production defaults after the A40 OOM."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from gigaflow_f1tenth.config import (
    CNN_POOL_BINS,
    CNN_POOL_MODE,
    CNN_PROJECTION_DIM,
    LEGACY_CNN_POOL_BINS,
    LEGACY_CNN_POOL_MODE,
    configure_cuda_allocator,
    estimate_memory_bytes,
    load_config,
    ppo_minibatch_backward_bytes,
)
from gigaflow_f1tenth.model import build_actor

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "configs" / "production_rtxpro6000_clean_grace.yaml"
DEFAULT = ROOT / "configs" / "default.yaml"
SMOKE = ROOT / "configs" / "smoke.yaml"
GIB = 1024 ** 3
A40_DEVICE_GIB = 44.42
PRO6000_DEVICE_GIB = 96.0


def test_package_defaults_are_maxpool64_512d():
    assert CNN_POOL_MODE == "max"
    assert CNN_POOL_BINS == 64
    assert CNN_PROJECTION_DIM == 512
    assert LEGACY_CNN_POOL_MODE == "avg"
    assert LEGACY_CNN_POOL_BINS == 32


def test_production_rtxpro6000_is_maxpool64_512d_at_256_worlds():
    cfg = load_config(PRODUCTION)
    assert cfg.agents.cnn_pool_mode == "max"
    assert cfg.agents.cnn_pool_bins == 64
    assert cfg.agents.cnn_projection_dim == 512
    assert cfg.worlds.num_worlds == 256
    assert cfg.worlds.max_agents_per_world == 8
    assert cfg.ppo.rollout_length == 256
    actor = build_actor(cfg)
    assert isinstance(actor.encoder.pool, nn.AdaptiveMaxPool1d)
    assert actor.encoder.pool_bins == 64
    assert actor.encoder.projection.in_features == 64 * 64
    assert actor.encoder.projection.out_features == 512
    relus = [m for m in actor.encoder.conv if isinstance(m, nn.ReLU)]
    assert relus and all(m.inplace for m in relus)
    cond_relus = [m for m in actor.condition_encoder.net if isinstance(m, nn.ReLU)]
    assert cond_relus and all(m.inplace for m in cond_relus)


def test_default_and_smoke_use_maxpool64():
    default_cfg = load_config(DEFAULT)
    assert default_cfg.agents.cnn_pool_mode == "max"
    assert default_cfg.agents.cnn_pool_bins == 64
    assert default_cfg.agents.cnn_projection_dim == 512
    assert estimate_memory_bytes(default_cfg) <= (
        default_cfg.profiling.estimate_bytes_budget
    )
    cfg = load_config(SMOKE)
    assert cfg.agents.cnn_pool_mode == "max"
    assert cfg.agents.cnn_pool_bins == 64
    actor = build_actor(cfg)
    assert isinstance(actor.encoder.pool, nn.AdaptiveMaxPool1d)
    assert all(m.inplace for m in actor.encoder.conv if isinstance(m, nn.ReLU))


def test_yaml_presets_declare_maxpool64():
    skip = {"gpu_smoke.yaml", "smoke.yaml", "convergence_smoke.yaml"}
    for path in sorted(ROOT.joinpath("configs").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "agents" not in raw:
            continue
        agents = raw["agents"]
        assert agents.get("cnn_pool_mode", CNN_POOL_MODE) == "max", path.name
        assert agents.get("cnn_pool_bins", CNN_POOL_BINS) == 64, path.name
        if path.name not in skip and not path.name.startswith("dense_traffic_cpu"):
            assert agents.get("cnn_projection_dim", CNN_PROJECTION_DIM) == 512, path.name


def test_configure_cuda_allocator_is_idempotent(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    configure_cuda_allocator()
    first = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    assert "expandable_segments:True" in first
    configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == first
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    configure_cuda_allocator()
    conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    assert "max_split_size_mb:128" in conf
    assert "expandable_segments:True" in conf


def test_caps_spatial_second_graph_is_in_the_minibatch_estimate():
    cfg = load_config(PRODUCTION)
    with_caps = ppo_minibatch_backward_bytes(cfg)
    no_caps = ppo_minibatch_backward_bytes(
        replace(cfg, ppo=replace(cfg.ppo, caps_spatial_coef=0.0))
    )
    assert cfg.ppo.caps_spatial_coef > 0.0
    assert with_caps > no_caps


def test_a40_128_and_pro6000_256_estimates_stay_ordered():
    cfg256 = load_config(PRODUCTION)
    cfg128 = replace(cfg256, worlds=replace(cfg256.worlds, num_worlds=128))
    est128 = estimate_memory_bytes(cfg128)
    est256 = estimate_memory_bytes(cfg256)
    assert est256 > est128
    # A40 has 44.42 GiB device memory; estimate is torch allocated, not smi.
    assert est128 / GIB < A40_DEVICE_GIB
    # 256-world PRO 6000 budget must stay under 96 GiB allocated (smi is higher).
    assert est256 / GIB < PRO6000_DEVICE_GIB
    assert est256 <= cfg256.profiling.estimate_bytes_budget


def test_reconstruct_logp_clone_is_independent_of_batch_storage():
    from gigaflow_f1tenth.trainer import build_trainer

    cfg = load_config(SMOKE)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    batch = trainer.collect_rollout()
    prepared = trainer.reconstruct_prepared(batch)
    assert prepared.reconstruct_logp is not None
    assert prepared.reconstruct_logp.data_ptr() != batch.old_logp.data_ptr()
    torch.testing.assert_close(prepared.reconstruct_logp, batch.old_logp)
    batch.old_logp.add_(5.0)
    assert not torch.allclose(prepared.reconstruct_logp, batch.old_logp)

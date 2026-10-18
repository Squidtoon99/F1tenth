"""Production CUDA steps must stay device-resident (no per-step host sync)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(SMOKE)


def test_cuda_step_forbids_host_materialization(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device-residency gate")
    from dataclasses import replace

    worlds = replace(cfg.worlds, device="cuda", num_worlds=2, max_agents_per_world=2)
    cfg_cuda = replace(cfg, worlds=worlds)
    atlas = make_synthetic_oval_atlas(max_agents=2)
    sim = build_simulator(cfg_cuda, atlas, "cuda")
    assert sim.buffers.device_str == "cuda"

    hits: list[str] = []
    real_cpu = torch.Tensor.cpu
    real_numpy = torch.Tensor.numpy

    def _cpu(self, *args, **kwargs):
        hits.append("cpu")
        return real_cpu(self, *args, **kwargs)

    def _numpy(self, *args, **kwargs):
        hits.append("numpy")
        return real_numpy(self, *args, **kwargs)

    n = world_slot_layout(cfg_cuda).num_slots
    actions = torch.zeros((n, 2), device="cuda", dtype=torch.float32)
    actions[:, 0] = 0.2
    # Warmup compile outside the residency window.
    sim.step(actions)

    torch.Tensor.cpu = _cpu  # type: ignore[method-assign]
    torch.Tensor.numpy = _numpy  # type: ignore[method-assign]
    try:
        out = sim.step(actions)
        assert out["sensor_obs"].is_cuda
        assert out["rewards"].is_cuda
        assert out["compact_state"].is_cuda
        assert torch.isfinite(out["rewards"]).all()
        assert torch.isfinite(out["sensor_obs"]).all()
        # Overflow flag stays on device; callers may .item() outside the step.
        assert hasattr(out["broadphase_overflow"], "is_cuda")
        assert out["broadphase_overflow"].is_cuda or out["broadphase_overflow"].is_cpu
    finally:
        torch.Tensor.cpu = real_cpu  # type: ignore[method-assign]
        torch.Tensor.numpy = real_numpy  # type: ignore[method-assign]

    assert hits == [], f"host sync during CUDA step: {hits}"


def test_cuda_collect_and_reconstruct_device_resident(cfg, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device-residency gate")
    from dataclasses import replace

    worlds = replace(cfg.worlds, device="cuda", num_worlds=2, max_agents_per_world=2)
    ppo = replace(cfg.ppo, rollout_length=4, minibatch_size=8, num_epochs=1, amp=False)
    cfg_cuda = replace(cfg, worlds=worlds, ppo=ppo)
    atlas = make_synthetic_oval_atlas(max_agents=2)
    trainer = build_trainer(
        cfg_cuda, atlas=atlas, device="cuda", run_dir=tmp_path / "run"
    )
    trainer.setup()
    hits: list[str] = []
    real_cpu = torch.Tensor.cpu
    real_numpy = torch.Tensor.numpy

    def _cpu(self, *args, **kwargs):
        # Allow one-time metadata paths outside the timed loop by tracking count.
        hits.append("cpu")
        return real_cpu(self, *args, **kwargs)

    def _numpy(self, *args, **kwargs):
        hits.append("numpy")
        return real_numpy(self, *args, **kwargs)

    # Warmup
    trainer.collect_rollout()

    torch.Tensor.cpu = _cpu  # type: ignore[method-assign]
    torch.Tensor.numpy = _numpy  # type: ignore[method-assign]
    try:
        batch = trainer.collect_rollout()
        prepared = trainer.reconstruct_prepared(batch, verify_parity=True)
        assert batch.state.is_cuda
        assert prepared.sensor_obs.is_cuda
        assert prepared.ego_state.is_cuda
        assert trainer.profile["reconstruction_digest_mismatches"] == 0.0
    finally:
        torch.Tensor.cpu = real_cpu  # type: ignore[method-assign]
        torch.Tensor.numpy = real_numpy  # type: ignore[method-assign]

    assert hits == [], f"host sync during CUDA collect/reconstruct: {hits}"

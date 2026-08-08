"""estimate_memory_bytes must match the tensors training actually allocates."""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.buffers import allocate_rollout_buffer
from gigaflow_f1tenth.config import (
    BYTES_PER_FLOAT32,
    CUDA_LIBRARY_WORKSPACE_BYTES,
    ESTIMATE_SAFETY_MARGIN,
    actor_parameter_count,
    config_from_dict,
    config_to_dict,
    critic_parameter_count,
    estimate_memory_bytes,
    full_rollout_scoring_bytes,
    load_config,
    model_state_bytes,
    ppo_minibatch_backward_bytes,
    ppo_update_bytes,
    prepared_inputs_bytes,
    rollout_buffer_bytes,
)
from gigaflow_f1tenth.critic import build_critic
from gigaflow_f1tenth.model import build_actor
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
GPU_SMOKE = ROOT / "configs" / "gpu_smoke.yaml"
PREPARED_ATLAS = Path.home() / ".cache" / "gigaflow" / "tracks"


def _tensor_bytes(tensors) -> int:
    """Summed storage of distinct tensors (aliases counted once)."""
    sizes = {t.data_ptr(): t.numel() * t.element_size() for t in tensors}
    return sum(sizes.values())


def test_rollout_buffer_bytes_matches_every_allocated_tensor():
    """Any field added to the buffer must show up in the estimate."""
    cfg = load_config(SMOKE)
    buf = allocate_rollout_buffer(cfg, "cpu")
    tensors = [v for v in vars(buf).values() if isinstance(v, torch.Tensor)]
    assert _tensor_bytes(tensors) == rollout_buffer_bytes(cfg)


def test_rollout_buffer_stores_no_full_width_observations():
    cfg = load_config(SMOKE)
    buf = allocate_rollout_buffer(cfg, "cpu")
    obs_dim = cfg.agents.sensor_obs_dim
    for name, value in vars(buf).items():
        if isinstance(value, torch.Tensor):
            assert obs_dim not in tuple(value.shape), f"{name} stores 1097-D obs"


def test_prepared_inputs_bytes_matches_reconstruction():
    cfg = load_config(SMOKE)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    batch = trainer.collect_rollout()
    prepared = trainer.reconstruct_prepared(batch, verify_parity=True)
    # ego_state concatenates the track preview onto the compact state, so it is
    # a fresh [T, S, D] tensor; if it ever aliases batch.state again the
    # estimate would double count (or drop) a full-width tensor.
    assert prepared.ego_state.data_ptr() != batch.state.data_ptr()
    measured = _tensor_bytes(
        [
            prepared.sensor_obs,
            prepared.ego_state,
            prepared.other_agents,
            prepared.other_mask,
            prepared.bootstrap_values,
        ]
    )
    assert measured == prepared_inputs_bytes(cfg)


def test_estimate_counts_observations_for_the_whole_rollout():
    """A minibatch-sized observation term understates the reconstruction 128x."""
    cfg = load_config(SMOKE)
    tiny_minibatch = replace(cfg, ppo=replace(cfg.ppo, minibatch_size=8))
    agent_steps = (
        cfg.worlds.num_worlds
        * cfg.worlds.max_agents_per_world
        * cfg.ppo.rollout_length
    )
    full_obs = agent_steps * cfg.agents.sensor_obs_dim * BYTES_PER_FLOAT32
    assert prepared_inputs_bytes(tiny_minibatch) > full_obs


def test_estimate_is_the_sum_of_its_reported_terms():
    cfg = load_config(SMOKE)
    terms = (
        rollout_buffer_bytes(cfg)
        + prepared_inputs_bytes(cfg)
        + ppo_update_bytes(cfg)
        + model_state_bytes(cfg)
        + CUDA_LIBRARY_WORKSPACE_BYTES
        + max(full_rollout_scoring_bytes(cfg), ppo_minibatch_backward_bytes(cfg))
    )
    assert estimate_memory_bytes(cfg) == int(ESTIMATE_SAFETY_MARGIN * terms)


def test_activation_terms_dominate_the_buffers_at_scale():
    """The activation term is what the estimate exists to guard, not the buffers.

    A rollout is scored one flattened batch at a time, so every agent-step pays
    the CNN's live set — an order of magnitude more than the tensors it reads.
    """
    cfg = load_config(ROOT / "configs" / "production_rtx4080.yaml")
    buffers = (
        rollout_buffer_bytes(cfg) + prepared_inputs_bytes(cfg) + ppo_update_bytes(cfg)
    )
    assert full_rollout_scoring_bytes(cfg) > 5 * buffers


def test_parameter_counts_match_the_built_modules():
    """Model state is estimated from shapes; drift from the modules is silent."""
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    critic = build_critic(cfg)
    assert actor_parameter_count(cfg) == sum(p.numel() for p in actor.parameters())
    assert critic_parameter_count(cfg) == sum(p.numel() for p in critic.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA peak-memory gate")
@pytest.mark.skipif(
    not (PREPARED_ATLAS / "manifest.json").exists(),
    reason="prepared track atlas cache not present",
)
def test_preview_atlas_stays_small_on_device():
    """D1 memory gate: only the sampler's fields move to device, not the atlas.

    The full pinned atlas carries a corridor EDT grid the simulator already
    copies to device itself via Warp (outside torch's allocator); mirroring
    the whole atlas into a torch-resident view here would silently double
    hundreds of MB inside torch's own peak-memory accounting instead.
    """
    cfg = load_config(SMOKE)
    cfg_dict = config_to_dict(cfg)
    cfg_dict["tracks"]["manifest_path"] = str(PREPARED_ATLAS)
    cfg = config_from_dict(cfg_dict)
    trainer = build_trainer(cfg, device="cuda", run_dir=None)
    trainer.setup()
    preview_atlas = trainer._preview_atlas
    assert preview_atlas.centerline_xy.is_cuda
    assert preview_atlas.tangents_xy.is_cuda
    assert preview_atlas.cum_length.is_cuda
    device_bytes = sum(
        getattr(preview_atlas, name).element_size()
        * getattr(preview_atlas, name).numel()
        for name in (
            "offsets", "centerline_xy", "tangents_xy",
            "widths_rl", "cum_length", "lengths",
        )
    )
    # The full atlas's EDT/LUT grids are hundreds of MB; the preview-relevant
    # fields for the full 23-track pinned set are under 2 MB.
    assert device_bytes < 2_000_000
    # The much larger EDT/LUT arrays must stay off the torch-resident view.
    assert not isinstance(preview_atlas.edt_distance, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA peak-memory gate")
@pytest.mark.parametrize(
    "worlds,agents,rollout,minibatch",
    [(8, 2, 16, 64), (16, 4, 64, 512)],
)
def test_measured_cuda_peak_stays_under_the_estimate(
    worlds, agents, rollout, minibatch
):
    """GPU validation gate at two scales 16x apart in agent-steps.

    One scale can be matched by a wrong model with a compensating constant; the
    estimate has to hold as worlds, agents, and rollout length all grow.
    """
    base = load_config(GPU_SMOKE)
    cfg = replace(
        base,
        worlds=replace(
            base.worlds, num_worlds=worlds, max_agents_per_world=agents
        ),
        ppo=replace(base.ppo, rollout_length=rollout, minibatch_size=minibatch),
    )
    gc.collect()
    torch.cuda.empty_cache()
    outside = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    trainer = build_trainer(cfg, device="cuda", run_dir=None)
    trainer.setup()
    # First update allocates optimizer state and library workspaces; both are in
    # the estimate, so the gate reads the absolute peak of the second one.
    trainer.train_update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    trainer.train_update()
    torch.cuda.synchronize()
    measured = torch.cuda.max_memory_allocated() - outside
    estimate = estimate_memory_bytes(cfg)
    assert measured <= 0.9 * estimate, (
        f"measured CUDA peak {measured} leaves under 10% headroom in estimate "
        f"{estimate} at {worlds}x{agents} worlds, rollout {rollout}"
    )
    # An estimate that is merely large is not a prediction; keep it usable.
    assert measured >= 0.4 * estimate, (
        f"estimate {estimate} overshoots measured peak {measured} so far that "
        "it would reject configurations that fit"
    )

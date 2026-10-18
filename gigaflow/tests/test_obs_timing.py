"""Action-time observation timing, replay parity, and compact-actor parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.buffers import STATE_INDEX
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.trainer import ReconstructionParityError, build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"

# Channels added to the compact state so proprioception is reconstructible.
REPLAY_ONLY_CHANNELS = (
    "executed_long_0",
    "executed_long_1",
    "executed_steer_0",
    "executed_steer_1",
    "executed_steer_2",
    "executed_steer_3",
    "omega_0",
    "omega_1",
    "omega_2",
    "omega_3",
)


def _short_rollout_trainer(steps: int = 6):
    """Short rollout with two live cars per world, so opponent LiDAR is covered."""
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            max_agents_per_world=4,
            density_bins=("dense",),
            solo_world_fraction=0.0,
        ),
        ppo=replace(cfg.ppo, rollout_length=steps, num_epochs=1, minibatch_size=8),
    )
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    return trainer


def test_collect_does_not_double_build_lidar():
    cfg = load_config(SMOKE)
    ppo = replace(cfg.ppo, rollout_length=4, num_epochs=1, minibatch_size=16)
    cfg = replace(cfg, ppo=ppo)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    assert trainer.sim is not None

    builds = {"n": 0}
    real = trainer.sim.rebuild_sensors

    def counted():
        builds["n"] += 1
        return real()

    trainer.sim.rebuild_sensors = counted  # type: ignore[method-assign]
    # Also count internal launch path used by step().
    real_launch = trainer.sim._launch_sensors

    def counted_launch():
        builds["n"] += 1
        return real_launch()

    trainer.sim._launch_sensors = counted_launch  # type: ignore[method-assign]

    batch = trainer.collect_rollout()
    # One sensor build per control step inside sim.step (not a collect-time rebuild).
    assert builds["n"] == cfg.ppo.rollout_length
    assert batch.obs_digest.shape == (cfg.ppo.rollout_length, trainer.layout.num_slots)
    assert int(batch.obs_digest.abs().sum()) != 0


def test_replayed_observations_are_bitwise_identical_to_collection():
    """Dropping stored obs is only safe if replay reproduces every bit."""
    trainer = _short_rollout_trainer()
    collected = []
    real = trainer.sim.action_observation

    def recording():
        obs = real()
        collected.append(obs.clone())
        return obs

    trainer.sim.action_observation = recording  # type: ignore[method-assign]
    try:
        batch = trainer.collect_rollout()
    finally:
        trainer.sim.action_observation = real  # type: ignore[method-assign]

    replayed = trainer._replay_observations(batch)
    active = batch.state[..., STATE_INDEX["active"]] > 0.5
    per_world = active[0].view(-1, trainer.layout.max_agents_per_world).sum(dim=-1)
    assert int(per_world.max()) >= 2, "need an opponent in view to cover car hits"
    reference = torch.stack(collected, dim=0)
    assert torch.equal(replayed[active], reference[active])


@pytest.mark.parametrize("channel", REPLAY_ONLY_CHANNELS)
def test_replay_gate_rejects_a_perturbed_proprio_channel(channel):
    """Each added channel is load-bearing, and a mismatch is fatal not advisory."""
    trainer = _short_rollout_trainer(steps=4)
    batch = trainer.collect_rollout()
    trainer.reconstruct_prepared(batch, verify_parity=True)
    assert trainer.profile["reconstruction_digest_mismatches"] == 0.0
    batch.state[..., STATE_INDEX[channel]] += 0.5
    with pytest.raises(ReconstructionParityError):
        trainer.reconstruct_prepared(batch, verify_parity=True)


def test_action_obs_matches_pre_step_state_sensors():
    cfg = load_config(SMOKE)
    ppo = replace(cfg.ppo, rollout_length=2, num_epochs=1, minibatch_size=8)
    cfg = replace(cfg, ppo=ppo)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    assert trainer.sim is not None
    live = trainer.sim.action_observation().clone()
    again = trainer.sim.rebuild_sensors().clone()
    assert torch.allclose(live, again, atol=0.0, rtol=0.0)


def test_compact_actor_matches_full_batch_on_active_rows():
    cfg = load_config(SMOKE)
    ppo = replace(cfg.ppo, rollout_length=4, num_epochs=1, minibatch_size=16, amp=False)
    cfg = replace(cfg, ppo=ppo)
    torch.manual_seed(0)
    compact = build_trainer(cfg, device="cpu", run_dir=None)
    compact.setup()
    full = build_trainer(cfg, device="cpu", run_dir=None)
    full.setup()
    assert compact.actor is not None and full.actor is not None
    full.actor.load_state_dict(compact.actor.state_dict())
    full.ppo.carry_hidden = compact.ppo.carry_hidden.clone()  # type: ignore[union-attr]
    full.ppo.carry_reset = compact.ppo.carry_reset.clone()  # type: ignore[union-attr]
    full.sim.restore_state(compact.sim.pack_state())
    full.sim.rebuild_sensors()
    full.disable_compact_actor = True

    torch.manual_seed(1)
    b_c = compact.collect_rollout()
    torch.manual_seed(1)
    # Align RNG for squashed-Gaussian sampling.
    full.ppo.carry_hidden = b_c.gru_start.clone()
    full.ppo.carry_reset = torch.zeros_like(b_c.reset_mask[0])
    # Re-seed and collect with full actor path from same starting state is hard;
    # instead compare one-step compact vs full on identical inputs.
    n = compact.layout.num_slots
    obs = compact.sim.action_observation().clone()
    cond = compact._condition_tensor()
    hidden = torch.zeros(n, cfg.agents.gru_hidden_dim)
    reset = torch.zeros(n, dtype=torch.bool)
    active = compact.sim.buffers.torch_arrays.active > 0
    # Deterministic actions: stochastic sampling advances RNG by batch size, so
    # compact vs full-slot draws are not comparable under the same seed.
    with torch.no_grad():
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        out_c = compact.actor.forward(
            obs.index_select(0, idx),
            cond.index_select(0, idx),
            hidden.index_select(0, idx),
            reset_mask=reset.index_select(0, idx),
            deterministic=True,
        )
        out_f = full.actor.forward(
            obs, cond, hidden, reset_mask=reset, deterministic=True
        )
    assert torch.allclose(out_c.actions, out_f.actions[idx], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_c.log_prob, out_f.log_prob[idx], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_c.hidden, out_f.hidden[idx], atol=1e-5, rtol=1e-5)
    a1, l1, h1, p1 = compact._actor_step_compact(obs, cond, hidden, reset, active)
    inactive = ~active
    if bool(inactive.any()):
        assert torch.count_nonzero(a1[inactive]) == 0
        assert torch.count_nonzero(l1[inactive]) == 0

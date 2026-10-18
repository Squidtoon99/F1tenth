"""Spatial CAPS on the real gigaflow PPO (temporal/grad CAPS stay off)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.buffers import CompactRolloutBatch
from gigaflow_f1tenth.config import AGENT_STATE_DIM, LIDAR_DIM, load_config
from gigaflow_f1tenth.critic import CRITIC_EGO_STATE_DIM, build_critic
from gigaflow_f1tenth.model import build_actor
from gigaflow_f1tenth.ppo import (
    PreparedPPOInputs,
    build_ppo,
    caps_spatial_loss,
    caps_temporal_loss,
    evaluate_actions_sequence,
    perturb_lidar_obs,
)
from gigaflow_f1tenth.rewards import sample_private_styles, styles_to_condition_batch

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _cfg(*, caps_spatial_coef: float, caps_temporal_coef: float = 0.0):
    cfg = load_config(SMOKE)
    return replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=1, max_agents_per_world=2),
        ppo=replace(
            cfg.ppo,
            rollout_length=4,
            num_epochs=1,
            minibatch_size=8,
            amp=False,
            adaptive_filter_enabled=False,
            target_kl=0.0,
            actor_kl_soft=0.0,
            actor_kl_hard=0.0,
            actor_kl_stop_enabled=False,
            caps_spatial_coef=float(caps_spatial_coef),
            caps_spatial_std=0.05,
            caps_temporal_coef=float(caps_temporal_coef),
            caps_coef=0.0,
            grad_caps_coef=0.0,
            total_updates=4,
        ),
        ablations=replace(cfg.ablations, adaptive_filter_enabled=False),
    )


def _real_batch(cfg, actor):
    torch.manual_seed(0)
    t = cfg.ppo.rollout_length
    s = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    obs_dim = cfg.agents.sensor_obs_dim
    rng = np.random.default_rng(0)
    styles = sample_private_styles(cfg, s, rng)
    cond = torch.as_tensor(styles_to_condition_batch(styles), dtype=torch.float32)
    cond = cond.unsqueeze(0).expand(t, s, -1).contiguous()
    sensor = torch.as_tensor(rng.random((t, s, obs_dim), dtype=np.float32))
    valid = torch.ones(t, s, dtype=torch.bool)
    done = torch.zeros(t, s, dtype=torch.bool)
    timeout = torch.zeros(t, s, dtype=torch.bool)
    reset = torch.zeros(t, s, dtype=torch.bool)
    gru_start = actor.initial_hidden(s, "cpu")
    actions = torch.zeros(t, s, 2)
    old_logp = torch.zeros(t, s)
    pre_tanh = torch.zeros(t, s, 2)
    with torch.no_grad():
        hidden = gru_start.clone()
        for step in range(t):
            out = actor.forward(
                sensor[step],
                cond[step],
                hidden,
                reset_mask=reset[step],
            )
            actions[step].copy_(out.actions)
            old_logp[step].copy_(out.log_prob)
            pre_tanh[step].copy_(out.pre_tanh)
            hidden = out.hidden
    batch = CompactRolloutBatch(
        state=torch.zeros(t, s, AGENT_STATE_DIM),
        next_state=torch.zeros(t, s, AGENT_STATE_DIM),
        actions=actions,
        rewards=torch.zeros(t, s),
        valid=valid,
        done=done,
        timeout=timeout,
        reset_mask=reset,
        track_id=torch.zeros(s, dtype=torch.int32),
        condition=cond,
        sensor_noise_seed=torch.zeros(t, s, dtype=torch.int64),
        episode_id=torch.zeros(t, s, dtype=torch.int32),
        episode_step=torch.zeros(t, s, dtype=torch.int32),
        gru_start=gru_start,
        old_logp=old_logp,
        obs_digest=torch.zeros(t, s, dtype=torch.int64),
        pre_tanh=pre_tanh,
    )
    n_others = max(cfg.worlds.max_agents_per_world - 1, 1)
    prepared = PreparedPPOInputs(
        sensor_obs=sensor,
        ego_state=torch.zeros(t, s, CRITIC_EGO_STATE_DIM),
        other_agents=torch.zeros(t, s, n_others, AGENT_STATE_DIM),
        other_mask=torch.ones(t, s, n_others, dtype=torch.bool),
        bootstrap_values=torch.zeros(t, s),
    )
    return batch, prepared


def test_ppo_defaults_enable_spatial_and_temporal_caps():
    cfg = load_config(SMOKE)
    assert cfg.ppo.caps_spatial_coef == pytest.approx(0.1)
    assert cfg.ppo.caps_spatial_std == pytest.approx(0.05)
    assert cfg.ppo.caps_temporal_coef == pytest.approx(0.02)
    assert cfg.ppo.caps_coef == pytest.approx(0.0)
    assert cfg.ppo.grad_caps_coef == pytest.approx(0.0)


def test_spatial_caps_perturbs_lidar_only():
    obs = torch.zeros(2, 4, LIDAR_DIM + 16)
    obs[..., :LIDAR_DIM] = 1.0
    obs[..., LIDAR_DIM:] = 2.0
    torch.manual_seed(0)
    pert = perturb_lidar_obs(obs, 0.05)
    assert torch.equal(pert[..., LIDAR_DIM:], obs[..., LIDAR_DIM:])
    assert not torch.equal(pert[..., :LIDAR_DIM], obs[..., :LIDAR_DIM])


def test_caps_temporal_penalizes_action_jumps():
    smooth = torch.zeros(3, 8, 2)
    smooth[:, :, 0] = torch.linspace(-0.1, 0.1, 8)
    bang = torch.zeros(3, 8, 2)
    bang[:, 0::2, 0] = 1.0
    bang[:, 1::2, 0] = -1.0
    assert float(caps_temporal_loss(smooth).item()) < 0.01
    assert float(caps_temporal_loss(bang).item()) > 0.5
    mask = torch.zeros(3, 7)
    masked = caps_temporal_loss(bang, mask)
    assert float(masked.item()) == pytest.approx(0.0)


def test_spatial_caps_loss_positive_on_real_actor():
    cfg = _cfg(caps_spatial_coef=0.1)
    actor = build_actor(cfg)
    batch, prepared = _real_batch(cfg, actor)
    obs = prepared.sensor_obs.transpose(0, 1).contiguous()
    act = batch.actions.transpose(0, 1).contiguous()
    cond = batch.condition.transpose(0, 1).contiguous()
    reset = batch.reset_mask.transpose(0, 1).contiguous()
    pre = batch.pre_tanh.transpose(0, 1).contiguous()
    scored = evaluate_actions_sequence(
        actor,
        obs,
        cond,
        batch.gru_start,
        act,
        reset_mask=reset,
        pre_tanh=pre,
        return_mean=True,
    )
    assert len(scored) >= 4
    torch.manual_seed(1)
    obs_pert = perturb_lidar_obs(obs, cfg.ppo.caps_spatial_std)
    pert = evaluate_actions_sequence(
        actor,
        obs_pert,
        cond,
        batch.gru_start,
        act,
        reset_mask=reset,
        pre_tanh=pre,
        return_mean=True,
    )
    loss = caps_spatial_loss(scored[3], pert[3])
    assert float(loss.item()) > 0.0


def test_ppo_update_adds_spatial_term_only_when_coef_positive():
    cfg_on = _cfg(caps_spatial_coef=0.1)
    actor_on = build_actor(cfg_on)
    critic_on = build_critic(cfg_on)
    learner_on = build_ppo(cfg_on, actor_on, critic_on, device="cpu")
    batch_on, prepared_on = _real_batch(cfg_on, actor_on)
    stats_on = learner_on.update(batch_on, prepared_on)
    assert stats_on.caps_spatial_loss > 0.0
    assert cfg_on.ppo.caps_coef == pytest.approx(0.0)
    assert cfg_on.ppo.grad_caps_coef == pytest.approx(0.0)

    cfg_off = _cfg(caps_spatial_coef=0.0)
    actor_off = build_actor(cfg_off)
    critic_off = build_critic(cfg_off)
    learner_off = build_ppo(cfg_off, actor_off, critic_off, device="cpu")
    batch_off, prepared_off = _real_batch(cfg_off, actor_off)
    stats_off = learner_off.update(batch_off, prepared_off)
    assert stats_off.caps_spatial_loss == pytest.approx(0.0)
    assert stats_off.caps_temporal_loss == pytest.approx(0.0)


def test_ppo_update_adds_temporal_term_when_coef_positive():
    cfg_on = _cfg(caps_spatial_coef=0.0, caps_temporal_coef=0.02)
    actor_on = build_actor(cfg_on)
    critic_on = build_critic(cfg_on)
    learner_on = build_ppo(cfg_on, actor_on, critic_on, device="cpu")
    batch_on, prepared_on = _real_batch(cfg_on, actor_on)
    stats_on = learner_on.update(batch_on, prepared_on)
    assert stats_on.caps_temporal_loss > 0.0
    assert stats_on.caps_spatial_loss == pytest.approx(0.0)

    cfg_off = _cfg(caps_spatial_coef=0.0, caps_temporal_coef=0.0)
    actor_off = build_actor(cfg_off)
    critic_off = build_critic(cfg_off)
    learner_off = build_ppo(cfg_off, actor_off, critic_off, device="cpu")
    batch_off, prepared_off = _real_batch(cfg_off, actor_off)
    stats_off = learner_off.update(batch_off, prepared_off)
    assert stats_off.caps_temporal_loss == pytest.approx(0.0)

"""Focused tests for live GRU hidden carry across collection, opponents, eval."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from evaluation import actor_is_recurrent, deterministic_rollout
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.opponents import MixedOpponentController, PolicyOpponent
from f1tenth_env.sensors import ACTOR_OBS_DIM
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM, LIDAR_DIM, PROPRIO_DIM
from standalone_trainer import (
    REPLAY_CHECKPOINT_INTERVAL,
    TrajectoryReplayBuffer,
    build_env_cfg,
)

OBS_DIM = LIDAR_DIM + PROPRIO_DIM
ACT_DIM = 2
HIDDEN = [32, 32]
DEVICE = torch.device("cpu")


def _gru_actor(
    seed: int = 0,
    *,
    hidden_sizes=None,
    lidar_pool_bins: int = 16,
):
    torch.manual_seed(seed)
    return make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=list(hidden_sizes or HIDDEN),
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=lidar_pool_bins,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    ).to(DEVICE)


def _env_matched_gru_actor(seed: int = 0):
    """Match DEFAULT_CONFIG PolicyOpponent architecture for hot-load tests."""
    return _gru_actor(
        seed,
        hidden_sizes=DEFAULT_CONFIG["model"]["actor_hidden_layers"],
        lidar_pool_bins=int(DEFAULT_CONFIG["model"]["lidar_pool_bins"]),
    )


def _configure_cpu():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=DEVICE,
        eps=1e-12,
    )


def _make_policy_env(*, num_envs: int = 2, opponent_strategy: str = "policy"):
    _configure_cpu()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = opponent_strategy
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": num_envs},
    )
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def test_actor_is_recurrent_detects_gru_only():
    gru = _gru_actor(1)

    class _FeedForwardStub(nn.Module):
        def forward(self, obs, deterministic=False, with_logprob=True):
            del deterministic, with_logprob
            return obs[..., :ACT_DIM], None

    assert actor_is_recurrent(gru)
    assert not actor_is_recurrent(_FeedForwardStub())


def test_policy_opponent_carries_and_resets_hidden():
    actor = _gru_actor(2)
    opp = PolicyOpponent(actor=actor, device=DEVICE)
    assert opp._recurrent
    obs = torch.randn(3, OBS_DIM)
    with torch.no_grad():
        a0 = opp.act_observation(obs)
        h_after = opp._hidden.clone()
        a1 = opp.act_observation(obs)
    assert a0.shape == (3, ACT_DIM)
    assert h_after.shape == (3, GRU_HIDDEN_DIM)
    assert not torch.allclose(h_after, torch.zeros_like(h_after))
    # Second step from carried state differs from a fresh zero-state step.
    fresh = PolicyOpponent(actor=copy.deepcopy(actor), device=DEVICE)
    with torch.no_grad():
        a_fresh = fresh.act_observation(obs)
    assert not torch.allclose(a1, a_fresh, atol=1e-5)

    # Reset after a0-state: clone a fresh opponent at h_after, then mask-reset.
    opp2 = PolicyOpponent(actor=copy.deepcopy(actor), device=DEVICE)
    with torch.no_grad():
        opp2.act_observation(obs)
    h_mid = opp2._hidden.clone()
    mask = torch.tensor([True, False, True])
    opp2.reset(mask)
    assert torch.allclose(opp2._hidden[0], torch.zeros(GRU_HIDDEN_DIM))
    assert torch.allclose(opp2._hidden[2], torch.zeros(GRU_HIDDEN_DIM))
    assert torch.allclose(opp2._hidden[1], h_mid[1])


def test_policy_opponent_load_snapshot_zeros_all_hidden():
    actor_a = _gru_actor(3)
    actor_b = _gru_actor(4)
    opp = PolicyOpponent(actor=actor_a, device=DEVICE)
    obs = torch.randn(2, OBS_DIM)
    with torch.no_grad():
        opp.act_observation(obs)
    assert opp._hidden.abs().sum() > 0
    opp.load_snapshot(
        actor_b.state_dict(),
        torch.zeros(OBS_DIM),
        torch.ones(OBS_DIM),
        actor_architecture=actor_b.actor_architecture,
    )
    assert torch.allclose(opp._hidden, torch.zeros_like(opp._hidden))


def test_learner_opponent_hidden_isolation():
    learner = _gru_actor(5)
    opponent_actor = _gru_actor(6)
    opp = PolicyOpponent(actor=opponent_actor, device=DEVICE)
    obs_l = torch.randn(2, OBS_DIM)
    obs_o = torch.randn(2, OBS_DIM)
    h_l = learner.initial_hidden(2, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        _, _, h_l = learner.step(obs_l, h_l, deterministic=True, with_logprob=False)
        opp.act_observation(obs_o)
    assert h_l.data_ptr() != opp._hidden.data_ptr()
    # Mutating opponent hidden must not touch the learner carry.
    before = h_l.clone()
    opp._hidden.zero_()
    assert torch.equal(h_l, before)
    assert h_l.abs().sum() > 0


def test_mixed_reset_mask_zeros_only_masked_policy_rows():
    actor = _gru_actor(6)
    policy = PolicyOpponent(actor=actor, device=DEVICE)
    ctrl = MixedOpponentController(policy)
    obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        policy.act_observation(obs)
    before = policy._hidden.clone()
    mask = torch.tensor([False, True, False, True])
    ctrl.reset(mask)
    assert torch.allclose(policy._hidden[1], torch.zeros(GRU_HIDDEN_DIM))
    assert torch.allclose(policy._hidden[3], torch.zeros(GRU_HIDDEN_DIM))
    assert torch.allclose(policy._hidden[0], before[0])
    assert torch.allclose(policy._hidden[2], before[2])


def test_collection_stores_pre_action_hidden_and_zeros_done_rows():
    actor = _gru_actor(7)
    num_envs = 2
    replay = TrajectoryReplayBuffer(
        capacity=num_envs * 64,
        actor_obs_dim=OBS_DIM,
        critic_obs_dim=392,
        act_dim=ACT_DIM,
        n_step=7,
        gamma=0.99,
        num_envs=num_envs,
        device=DEVICE,
        checkpoint_interval=REPLAY_CHECKPOINT_INTERVAL,
        hidden_dim=GRU_HIDDEN_DIM,
    )
    learner_h = actor.initial_hidden(num_envs, device=DEVICE, dtype=torch.float32)
    actor_obs = torch.randn(num_envs, OBS_DIM)
    critic_obs = torch.randn(num_envs, 392)
    stored_pre = []
    for t in range(REPLAY_CHECKPOINT_INTERVAL + 1):
        pre = learner_h.detach().clone()
        with torch.no_grad():
            actions, _, learner_h = actor.step(
                actor_obs, learner_h, deterministic=True, with_logprob=False
            )
        done = torch.zeros(num_envs)
        if t == 3:
            done[0] = 1.0
        replay.add(actor_obs, critic_obs, actions, torch.zeros(num_envs), done, hidden=pre)
        if t % REPLAY_CHECKPOINT_INTERVAL == 0:
            stored_pre.append(pre.clone())
        if done.any():
            learner_h[done.bool()] = 0
        actor_obs = torch.randn(num_envs, OBS_DIM)
        critic_obs = torch.randn(num_envs, 392)

    assert torch.allclose(
        replay.hidden[0, 0].float(), stored_pre[0][0], atol=1e-3
    )
    assert torch.allclose(
        replay.hidden[1, 0].float(), stored_pre[0][1], atol=1e-3
    )
    # After done on env 0 at t=3, subsequent carry for that row restarted at 0.
    assert learner_h[0].abs().sum() > 0  # stepped again after reset
    # Env 1 never done: hidden should differ from a fresh zero init after many steps.
    assert learner_h[1].abs().sum() > 0


def test_deterministic_rollout_gru_parity(warp_runtime):
    del warp_runtime
    env = _make_policy_env(num_envs=2, opponent_strategy="scripted")
    try:
        actor = _gru_actor(8)
        for p in actor.parameters():
            p.data.mul_(0.01)

        first = deterministic_rollout(
            env,
            actor,
            lambda o: o,
            num_steps=6,
            control_interval=env.control_interval,
            clip_actions=1.0,
            seed=11,
            with_sensors=True,
        )
        second = deterministic_rollout(
            env,
            actor,
            lambda o: o,
            num_steps=6,
            control_interval=env.control_interval,
            clip_actions=1.0,
            seed=11,
            with_sensors=True,
        )
        assert first["finite"] and second["finite"]
        assert torch.allclose(first["total_reward"], second["total_reward"], atol=1e-5)
        assert torch.allclose(
            first["final_observation"], second["final_observation"], atol=1e-5
        )
    finally:
        env.close()


def test_warp_policy_opponent_resets_hidden_on_refresh_and_done(warp_runtime):
    del warp_runtime
    env = _make_policy_env(num_envs=2, opponent_strategy="policy")
    try:
        assert env._policy_opponent is not None
        assert env._policy_opponent._recurrent
        actor = _env_matched_gru_actor(9)
        mean = torch.zeros(ACTOR_OBS_DIM)
        var = torch.ones(ACTOR_OBS_DIM)
        env.refresh_opponent_policy(
            actor.state_dict(),
            mean,
            var,
            actor_architecture=actor.actor_architecture,
        )
        # Refresh zeros then re-fills actions once from a fresh hidden.
        h_refresh = env._policy_opponent._hidden.clone()
        assert h_refresh.shape == (2, GRU_HIDDEN_DIM)
        assert h_refresh.abs().sum() > 0

        obs, _ = env.reset(seed=3, with_sensors=True)
        h0 = env._policy_opponent._hidden.clone()
        assert h0.abs().sum() > 0

        actions = torch.zeros(2, 2)
        for _ in range(4):
            obs, _, done, _ = env.step(
                actions, n_steps=env.control_interval, with_sensors=True
            )
        h_mid = env._policy_opponent._hidden.clone()
        assert not torch.allclose(h_mid, h0, atol=1e-5)
        # Mid-episode refresh must drop carry (restart from zero + one fill).
        env.refresh_opponent_policy(
            actor.state_dict(),
            mean,
            var,
            actor_architecture=actor.actor_architecture,
        )
        assert not torch.allclose(env._policy_opponent._hidden, h_mid, atol=1e-5)
        # Full env reset zeros then re-fills from zero state.
        env.reset(seed=3, with_sensors=True)
        assert not torch.allclose(env._policy_opponent._hidden, h_mid, atol=1e-5)
    finally:
        env.close()


def test_mixed_selfplay_policy_hidden_independent_of_scripted_rows(warp_runtime):
    del warp_runtime
    env = _make_policy_env(num_envs=4, opponent_strategy="mixed")
    try:
        actor = _env_matched_gru_actor(10)
        env.refresh_opponent_policy(
            actor.state_dict(),
            torch.zeros(ACTOR_OBS_DIM),
            torch.ones(ACTOR_OBS_DIM),
            actor_architecture=actor.actor_architecture,
        )
        env.reset(seed=5, with_sensors=True)
        h_before = env._policy_opponent._hidden.clone()
        # Advance a few steps; all rows get prefetch (kernel may ignore scripted).
        for _ in range(3):
            env.step(
                torch.zeros(4, 2),
                n_steps=env.control_interval,
                with_sensors=True,
            )
        h_after = env._policy_opponent._hidden.clone()
        assert h_after.shape == (4, GRU_HIDDEN_DIM)
        assert not torch.allclose(h_after, h_before)
        # Partial reset: only masked rows clear+refill; others retain carry.
        mask = torch.tensor([True, False, True, False], device=DEVICE)
        env.reset(mask, with_sensors=True)
        assert torch.allclose(env._policy_opponent._hidden[1], h_after[1], atol=1e-5)
        assert torch.allclose(env._policy_opponent._hidden[3], h_after[3], atol=1e-5)
        assert not torch.allclose(
            env._policy_opponent._hidden[0], h_after[0], atol=1e-5
        )
        assert not torch.allclose(
            env._policy_opponent._hidden[2], h_after[2], atol=1e-5
        )
    finally:
        env.close()

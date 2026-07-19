from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from evaluation import deterministic_rollout
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from qrsac import SquashedGaussianMLPActor


def _make_env(num_envs=16, opponent=False, device="cpu"):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device(device),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    if opponent:
        cfg["env"]["opponent_strategy"] = "scripted"
        cfg["reward"]["reward_scales"]["passing"] = 0.5
        cfg["reward"]["reward_scales"]["collision"] = 1.0
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def test_warp_env_end_to_end():
    env = _make_env()
    try:
        obs, _ = env.reset()
        assert obs.shape == (16, 390)
        assert torch.isfinite(obs).all()
        assert torch.equal(obs[:, 384:], torch.zeros_like(obs[:, 384:]))

        rewards = []
        for _ in range(40):
            actions = torch.zeros(16, 2)
            actions[:, 0] = 0.5
            actions[:, 1] = 0.1
            obs, reward, _, extras = env.step(
                actions, n_steps=env.control_interval
            )
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
            rewards.append(reward.mean().item())

        assert sum(rewards[-10:]) > sum(rewards[:10])
        assert float(extras["metrics"]["speed_xy"].mean()) > 0.5
    finally:
        env.close()


def test_warp_env_partial_reset_is_finite():
    env = _make_env(num_envs=8)
    try:
        obs, _ = env.reset()
        assert torch.isfinite(obs).all()
        mask = torch.zeros(8, dtype=torch.bool)
        mask[::2] = True
        env.reset(mask)
        assert torch.isfinite(env.obs_buf).all()
    finally:
        env.close()


def test_warp_observation_slots_match_state_and_wheels():
    env = _make_env(num_envs=4)
    try:
        action = torch.tensor([[0.4, 0.2]]).repeat(4, 1)
        env.step(action, n_steps=env.control_interval)
        obs, _, _, _ = env.step(action, n_steps=env.control_interval)
        state = env.read_state()
        wheels = env.read_wheel_state()

        assert torch.allclose(obs[:, :2], state["base_lin_vel"][:, :2])
        assert torch.allclose(obs[:, 2], state["base_ang_vel"][:, 2])
        assert torch.allclose(obs[:, 3:5], state["base_lin_acc"][:, :2])
        assert torch.allclose(obs[:, 5:7], env.last_actions)
        assert torch.allclose(obs[:, 372:380], wheels["tyre_slip"])
        assert torch.allclose(obs[:, 380:384], wheels["tyre_load"])
        assert torch.equal(obs[:, 384:390], torch.zeros_like(obs[:, 384:390]))
    finally:
        env.close()


def test_obs_tyre_load_slip_locked_and_nonzero():
    # Guards the tyre-load/slip -> observation path (contract slices OBS_TYRE_SLIP
    # / OBS_TYRE_LOAD): obs must equal read_wheel_state, and after driving through a
    # steady corner the loads must transfer off the static 1.0 and slip must be
    # nonzero (i.e. real physics is flowing into the model input, not stubbed).
    env = _make_env(num_envs=4)
    try:
        action = torch.tensor([[0.6, 0.8]]).repeat(4, 1)
        for _ in range(30):
            obs, _, _, _ = env.step(action, n_steps=env.control_interval)
        wheels = env.read_wheel_state()
        assert torch.allclose(obs[:, 372:380], wheels["tyre_slip"])
        assert torch.allclose(obs[:, 380:384], wheels["tyre_load"])

        load = obs[:, 380:384]
        slip = obs[:, 372:380]
        assert torch.isfinite(load).all() and torch.isfinite(slip).all()
        # Load transfer moved at least one wheel meaningfully off the static ratio.
        assert (load - 1.0).abs().max() > 0.05
        # Cornering under drive produces nonzero slip ratio and slip angle.
        assert slip.abs().max() > 1e-3
    finally:
        env.close()


def test_obs_contact_flag_is_track_boundary_proximity():
    # obs[11] (OBS_CONTACT_FLAG) is the track-boundary proximity flag
    # (boundary_dist < contact_margin_m) shared with the C++ deploy path and the
    # contract -- NOT the 1v1 box-collision flag. It must equal the boundary
    # metric thresholded at the margin on every tick.
    env = _make_env(num_envs=4)
    try:
        margin = float(env.obs_cfg.get("contact_margin_m", 0.08))
        action = torch.tensor([[0.5, 1.0]]).repeat(4, 1)
        for _ in range(60):
            obs, _, _, extras = env.step(action, n_steps=env.control_interval)
            boundary = extras["metrics"]["boundary_dist"]
            assert torch.equal(obs[:, 11], (boundary < margin).float())
    finally:
        env.close()


def test_obs_contact_flag_branches_on_margin():
    # Deterministic both-branch coverage at reset (car on track, boundary_dist>0):
    # a zero margin gives flag 0, a margin wider than the track gives flag 1.
    def _flag_after_reset(margin):
        rt.configure(
            float_dtype=torch.float32,
            int_dtype=torch.int32,
            dev=torch.device("cpu"),
            eps=1e-12,
        )
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["env"]["domain_randomization"]["enabled"] = False
        cfg["obs"]["contact_margin_m"] = margin
        env = F1tenthEnv(
            num_envs=4,
            env_cfg=cfg["env"],
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        )
        try:
            obs, _ = env.reset(seed=0)
            return obs[:, 11].clone()
        finally:
            env.close()

    assert torch.equal(_flag_after_reset(0.0), torch.zeros(4))
    assert torch.equal(_flag_after_reset(100.0), torch.ones(4))


def test_observation_ping_pong_preserves_previous_tick():
    env = _make_env(num_envs=4)
    try:
        first, _, _, _ = env.step(
            torch.full((4, 2), 0.1), n_steps=env.control_interval
        )
        first_snapshot = first.clone()
        second, _, _, _ = env.step(
            torch.full((4, 2), 0.2), n_steps=env.control_interval
        )
        assert env.step_launch_count == 3
        assert first.data_ptr() != second.data_ptr()
        assert torch.equal(first, first_snapshot)
        assert not torch.equal(first, second)
    finally:
        env.close()


def test_with_sensors_true_returns_dict_shapes():
    env = _make_env(num_envs=4)
    try:
        obs, _ = env.reset(seed=0, with_sensors=True)
        assert set(obs) == {"frenet", "actor", "lidar", "imu"}
        assert obs["frenet"].shape == (4, 390)
        assert obs["actor"].shape == (4, 1093)
        assert obs["lidar"].shape == (4, 1081)
        assert obs["imu"].shape == (4, 6)
        out, _, _, _ = env.step(
            torch.zeros(4, 2),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert set(out) == {"frenet", "actor", "lidar", "imu"}
        assert env.step_launch_count == 4
        assert torch.isfinite(out["lidar"]).all()
        assert torch.isfinite(out["imu"]).all()
        assert torch.isfinite(out["actor"]).all()
    finally:
        env.close()


def test_with_sensors_false_unchanged_flat_obs_and_launch_count():
    env_off = _make_env(num_envs=4)
    env_on = _make_env(num_envs=4)
    try:
        flat, _ = env_off.reset(seed=11, with_sensors=False)
        bundled, _ = env_on.reset(seed=11, with_sensors=True)
        assert isinstance(flat, torch.Tensor)
        assert flat.shape == (4, 390)
        assert torch.equal(flat, bundled["frenet"])

        actions = torch.full((4, 2), 0.15)
        flat_step, reward_off, done_off, _ = env_off.step(
            actions, n_steps=env_off.control_interval, with_sensors=False
        )
        bundled_step, reward_on, done_on, _ = env_on.step(
            actions, n_steps=env_on.control_interval, with_sensors=True
        )
        assert env_off.step_launch_count == 3
        assert env_on.step_launch_count == 4
        assert torch.equal(flat_step, bundled_step["frenet"])
        assert torch.equal(reward_off, reward_on)
        assert torch.equal(done_off, done_on)
    finally:
        env_off.close()
        env_on.close()


def test_partial_reset_does_not_change_unmasked_rows():
    env = _make_env(num_envs=8)
    try:
        before_obs = env.obs_buf.clone()
        before_state = env.read_state()["base_pos"].clone()
        mask = torch.zeros(8, dtype=torch.bool)
        mask[::2] = True
        env.reset(mask)
        after_state = env.read_state()["base_pos"]
        assert torch.equal(env.obs_buf[~mask], before_obs[~mask])
        assert torch.equal(after_state[~mask], before_state[~mask])
        assert not torch.equal(after_state[mask], before_state[mask])
    finally:
        env.close()


def test_batch_prefix_is_deterministic():
    small = _make_env(num_envs=4)
    large = _make_env(num_envs=9)
    try:
        small.reset(seed=23)
        large.reset(seed=23)
        generator = torch.Generator().manual_seed(5)
        for _ in range(12):
            actions = torch.rand(9, 2, generator=generator) * 2.0 - 1.0
            small.step(actions[:4], n_steps=small.control_interval)
            large.step(actions, n_steps=large.control_interval)
        assert torch.equal(
            small.read_state()["base_pos"],
            large.read_state()["base_pos"][:4],
        )
        assert torch.equal(small.obs_buf, large.obs_buf[:4])
    finally:
        small.close()
        large.close()


def test_different_reset_seeds_diverge():
    env = _make_env(num_envs=8)
    try:
        first, _ = env.reset(seed=1)
        first = first.clone()
        second, _ = env.reset(seed=2)
        assert not torch.equal(first, second)
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_step_uses_current_torch_stream():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cuda"),
        eps=1.0e-12,
    )
    env = _make_env(num_envs=32, device="cuda")
    try:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            obs, reward, _, _ = env.step(
                torch.zeros(32, 2, device="cuda"),
                n_steps=env.control_interval,
            )
            finite = torch.isfinite(obs).all() & torch.isfinite(reward).all()
        stream.synchronize()
        assert bool(finite)
    finally:
        env.close()
        rt.configure(
            float_dtype=torch.float32,
            int_dtype=torch.int32,
            dev=torch.device("cpu"),
            eps=1.0e-12,
        )


@pytest.mark.parametrize("opponent", [False, True])
def test_deterministic_rollout_repeats_trajectory(opponent):
    env = _make_env(num_envs=2, opponent=opponent)
    try:
        actor = SquashedGaussianMLPActor(
            env.num_obs, 2, [8], nn.ReLU, 1.0
        )
        for parameter in actor.parameters():
            parameter.data.zero_()

        positions = []

        def record(_step, rollout_env, _before, reward, _done, _extras):
            assert torch.isfinite(reward).all()
            positions.append(
                rollout_env.read_state()["base_pos"].detach().clone()
            )

        first = deterministic_rollout(
            env,
            actor,
            lambda obs: obs,
            num_steps=5,
            control_interval=env.control_interval,
            clip_actions=1.0,
            seed=7,
            callback=record,
        )
        first_positions = torch.stack(positions)
        positions.clear()
        second = deterministic_rollout(
            env,
            actor,
            lambda obs: obs,
            num_steps=5,
            control_interval=env.control_interval,
            clip_actions=1.0,
            seed=7,
            callback=record,
        )

        assert first["finite"]
        assert second["finite"]
        assert torch.equal(first_positions, torch.stack(positions))
    finally:
        env.close()

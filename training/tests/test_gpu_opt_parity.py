"""Parity gates for LiDAR LUT, opponent cull, march cap, and compile-time GRU."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from config import DEFAULT_CONFIG
from conftest import make_straight_track
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.sensors import LEGACY_LIDAR_MARCH_STEPS, beam_trig_lut
from f1tenth_policy.actor import make_actor
from f1tenth_policy.layout import ACTOR_OBS_DIM, LIDAR_DIM, PROPRIO_DIM
from ppo.ppo import PPOTrainer

from test_warp_lidar import (
    _CAR_LENGTH,
    _CAR_WIDTH,
    _NUM_BEAMS,
    _configure_cpu,
    _launch_lidar,
    _sensor_params,
)


def test_beam_trig_lut_matches_per_beam_sincos():
    sensor = DEFAULT_CONFIG["sensor"]
    fov = math.radians(float(sensor["fov_deg"]))
    n = int(sensor["num_beams"])
    angle_min = -0.5 * fov
    increment = fov / float(n - 1)
    beam_cos, beam_sin = beam_trig_lut(angle_min, increment, n)
    yaws = np.linspace(-math.pi, math.pi, 16, dtype=np.float64)
    mount = 0.03
    worst = 0.0
    for yaw in yaws:
        heading = yaw + mount
        ch = math.cos(heading)
        sh = math.sin(heading)
        for beam_id in range(n):
            beam = angle_min + beam_id * increment
            legacy_x = math.cos(heading + beam)
            legacy_y = math.sin(heading + beam)
            lut_x = ch * float(beam_cos[beam_id]) - sh * float(beam_sin[beam_id])
            lut_y = sh * float(beam_cos[beam_id]) + ch * float(beam_sin[beam_id])
            worst = max(worst, abs(legacy_x - lut_x), abs(legacy_y - lut_y))
    assert worst < 1e-5


def test_default_max_march_steps_is_bounded():
    assert DEFAULT_CONFIG["sensor"]["max_march_steps"] == 128
    assert LEGACY_LIDAR_MARCH_STEPS == 512


def test_march_128_matches_512_on_straight_corridor(warp_runtime):
    del warp_runtime
    _configure_cpu()
    centerline, width_left, width_right = make_straight_track()
    kwargs = dict(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(0.0, 0.0),
        ego_yaw=0.0,
    )
    ranges_128, _, _ = _launch_lidar(
        sensor=_sensor_params(max_march_steps=128), **kwargs
    )
    ranges_512, _, _ = _launch_lidar(
        sensor=_sensor_params(max_march_steps=512), **kwargs
    )
    close = torch.isclose(ranges_128, ranges_512, rtol=0.0, atol=1e-4)
    assert float(close.float().mean()) >= 0.999


def test_opponent_circle_cull_is_lossless(warp_runtime):
    del warp_runtime
    _configure_cpu()
    centerline, width_left, width_right = make_straight_track()
    solo, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(0.0, 0.0),
        ego_yaw=0.0,
    )
    far, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(0.0, 0.0),
        ego_yaw=0.0,
        opponent_pose=(40.0, 40.0),
        opponent_yaw=0.0,
    )
    assert torch.equal(solo, far)
    occluded, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(0.0, 0.0),
        ego_yaw=0.0,
        opponent_pose=(2.0, 0.0),
        opponent_yaw=0.0,
    )
    center = int(_NUM_BEAMS // 2)
    assert float(occluded[0, center]) < float(solo[0, center]) - 0.2
    radius = 0.5 * math.hypot(_CAR_LENGTH, _CAR_WIDTH)
    assert radius > 0.1


def _make_cuda_env(num_envs=2):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cuda"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["opponent_strategy"] = None
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA trajectory parity")
def test_thousand_step_lidar_scan_stays_finite_and_stable():
    env = _make_cuda_env(num_envs=2)
    try:
        env.reset(seed=7, with_sensors=True)
        first = None
        last = None
        for step in range(1000):
            actions = torch.zeros(2, 2, device=env.device)
            actions[:, 0] = 0.25
            out, reward, _, _ = env.step(
                actions, n_steps=env.control_interval, with_sensors=True
            )
            scan = out["lidar"]
            assert torch.isfinite(scan).all()
            assert torch.isfinite(reward).all()
            if step == 0:
                first = scan.detach().clone()
            last = scan.detach().clone()
        assert first is not None and last is not None
        agree = torch.isclose(first, last, rtol=0.0, atol=1e-3).float().mean()
        assert float(agree) < 0.99
        hit = (last < 29.9).float().mean()
        assert float(hit) > 0.5
    finally:
        env.close()


def test_compile_keeps_segment_gru():
    actor = make_actor(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=2,
        hidden_sizes=[32, 32],
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=64,
        lidar_dim=LIDAR_DIM,
        proprio_dim=PROPRIO_DIM,
    )
    trainer = PPOTrainer(
        actor,
        critic_obs_dim=8,
        actor_normalizer=None,
        critic_normalizer=None,
        device="cpu",
        compile=True,
        compile_mode="default",
        rollout_steps=4,
        num_epochs=1,
        env_minibatch_size=2,
    )
    assert trainer.actor.use_segment_gru is True
    compiled_seq = getattr(
        trainer._evaluate_actions_sequence, "_torchdynamo_orig_callable", None
    )
    assert compiled_seq is None

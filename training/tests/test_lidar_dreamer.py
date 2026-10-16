from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, control_dt_s
from dreamer import DreamerTrainer, WorldModelHeads, imagine_actor_critic
from f1tenth_policy.artifact import (
    build_sensor_artifact_payload,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.dreamer import (
    LidarDreamerActor,
    cartesian_polar_lidar,
    deploy_actor_state_dict,
    gru_rssm_parameter_count,
)
from f1tenth_policy.layout import (
    ACTOR_OBS_DIM,
    CONTROL_HZ,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_DREAMER_ARCHITECTURE_NAME,
    NUM_ACTIONS,
)
from f1tenth_policy.normalizer import ObsNormalizer
from ppo import PPOTrainer
from standalone_trainer import (
    build_config,
    make_policy_network,
    parse_args,
    validate_algorithm_config,
    validate_config_patch,
)

HIDDEN = [32, 32]


def _tiny_actor(**kwargs):
    params = dict(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=NUM_ACTIONS,
        hidden_sizes=HIDDEN,
        act_limit=1.0,
        deter_dim=16,
        stoch_dim=4,
        embed_dim=32,
    )
    params.update(kwargs)
    return LidarDreamerActor(**params)


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return build_config(args, patch=patch or {}, explicit=explicit)


def test_cartesian_polar_is_range_and_xy():
    ranges = torch.tensor([[1.0, 2.0, 3.0]])
    chart = cartesian_polar_lidar(
        ranges,
        angle_min=0.0,
        angle_increment=0.5 * torch.pi,
    )
    assert chart.shape == (1, 3, 3)
    assert torch.allclose(chart[0, 0], ranges[0])
    x = ranges[0] * torch.cos(
        torch.tensor([0.0, 0.5 * torch.pi, torch.pi])
    )
    y = ranges[0] * torch.sin(
        torch.tensor([0.0, 0.5 * torch.pi, torch.pi])
    )
    assert torch.allclose(chart[0, 1], x, atol=1e-5)
    assert torch.allclose(chart[0, 2], y, atol=1e-5)


def test_full_scan_uses_layout_angles():
    ranges = torch.linspace(1.0, 2.0, LIDAR_DIM).unsqueeze(0)
    chart = cartesian_polar_lidar(ranges)
    assert chart.shape == (1, 3, LIDAR_DIM)
    mid = LIDAR_DIM // 2
    angle = LIDAR_ANGLE_MIN + LIDAR_ANGLE_INCREMENT * mid
    assert chart[0, 1, mid].item() == pytest.approx(
        ranges[0, mid].item() * torch.cos(torch.tensor(angle)).item(),
        rel=1e-5,
    )


def test_ls3c_rssm_has_fewer_params_than_gru():
    from f1tenth_policy.dreamer import LightweightStateSpaceCell

    hidden = 256
    inp = 32 + NUM_ACTIONS
    cell = LightweightStateSpaceCell(inp, hidden)
    ls3c_n = sum(int(p.numel()) for p in cell.parameters())
    gru_n = gru_rssm_parameter_count(inp, hidden)
    assert ls3c_n < gru_n


def test_arm_config_is_lidar_dreamer_not_ppo():
    path = Path(__file__).resolve().parents[1] / (
        "configs/austin_e2e_lidar_dreamer.json"
    )
    patch = json.loads(path.read_text(encoding="utf-8"))
    validate_config_patch(patch)
    cfg = _resolve(["--algorithm", "lidar_dreamer"], patch=patch)
    assert cfg["algorithm"] == "lidar_dreamer"
    assert cfg["model"]["actor_type"] == LIDAR_DREAMER_ARCHITECTURE_NAME
    assert cfg["env"]["action_horizon"] == 1
    dt = control_dt_s(cfg["env"])
    assert dt == pytest.approx(1.0 / CONTROL_HZ)
    cfg["model"]["actor_hidden_layers"] = HIDDEN
    cfg["model"]["critic_hidden_layers"] = HIDDEN
    cfg["dreamer"]["deter_dim"] = 16
    cfg["dreamer"]["stoch_dim"] = 4
    cfg["dreamer"]["embed_dim"] = 32
    cfg["dreamer"]["hidden_sizes"] = HIDDEN
    actor = make_policy_network(cfg)
    assert isinstance(actor, LidarDreamerActor)
    assert actor.actor_architecture["encoding"] == "cartesian_polar"
    assert actor.actor_architecture["recurrent"] == "ls3c"
    assert actor.actor_architecture["deploy"] == "encoder_actor"
    validate_algorithm_config(cfg, num_envs=8)


def test_lidar_dreamer_rejected_as_unknown_without_flag():
    with pytest.raises(ValueError, match="lidar_dreamer"):
        validate_algorithm_config(
            {
                "algorithm": "dreamer",
                "env": {"num_actions": 2, "action_horizon": 1},
            },
            num_envs=8,
        )


def test_step_matches_deploy_hidden_api():
    actor = _tiny_actor()
    obs = torch.rand(3, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(3)
    action, logp, next_hidden = actor.step(
        obs, hidden, deterministic=False, with_logprob=True
    )
    assert action.shape == (3, NUM_ACTIONS)
    assert logp.shape == (3,)
    assert next_hidden.shape == hidden.shape
    restored = actor.__class__(
        obs_dim=actor.obs_dim,
        act_dim=actor.act_dim,
        hidden_sizes=HIDDEN,
        deter_dim=16,
        stoch_dim=4,
        embed_dim=32,
    )
    restored.load_state_dict(deploy_actor_state_dict(actor))
    restored.eval()
    actor.eval()
    h0 = actor.initial_hidden(1)
    a1, _, _ = actor.step(obs[:1], h0, deterministic=True, with_logprob=False)
    a2, _, _ = restored.step(obs[:1], h0, deterministic=True, with_logprob=False)
    assert torch.allclose(a1, a2, atol=1e-5)


def test_artifact_is_encoder_actor_only():
    actor = _tiny_actor()
    world = WorldModelHeads(actor, hidden_sizes=HIDDEN)
    payload = build_sensor_artifact_payload(
        actor_state_dict=deploy_actor_state_dict(actor),
        obs_norm=ObsNormalizer(ACTOR_OBS_DIM, device=torch.device("cpu")).state_dict(),
        actor_architecture=actor.actor_architecture,
        env_transitions=0,
        actor_obs_dim=ACTOR_OBS_DIM,
        critic_obs_dim=392,
        actor_layout_version=2,
        action_dim=NUM_ACTIONS,
        action_scale=1.0,
        config_version=4,
        policy_format_version=4,
        longitudinal_mode="force",
        steering_action_mode="delta",
        steering_delta_max_rad=3.141592653589793 / 120.0,
        f_drive_max=26.5,
        f_brake_max=23.1,
        i_drive_max_a=80.0,
        i_brake_max_a=40.0,
        i_slew_a_per_s=200.0,
        control_hz=CONTROL_HZ,
        simulator_id="f1tenth-torch",
        simulator_version=1,
        training_protocol={"algorithm": "lidar_dreamer"},
    )
    validate_sensor_policy_artifact(
        payload, expected_architecture=actor.actor_architecture
    )
    keys = payload["actor"]
    assert any(k.startswith("encoder.") for k in keys)
    assert any(k.startswith("ls3c.") for k in keys)
    assert not any("decoder" in k for k in keys)
    assert not any("reward_head" in k for k in keys)
    assert not any(k.startswith("prior_") for k in keys)
    world_keys = set(world.state_dict())
    assert world_keys.isdisjoint(keys)


def test_imagination_trains_actor_without_ppo_clip():
    torch.manual_seed(0)
    actor = _tiny_actor()
    world = WorldModelHeads(actor, hidden_sizes=HIDDEN)
    critic = nn.Sequential(nn.Linear(actor.deter_dim + actor.stoch_dim, 1))
    hidden = actor.initial_hidden(4)
    metrics = imagine_actor_critic(
        actor,
        world,
        lambda feat: critic(feat).squeeze(-1),
        hidden,
        horizon=3,
        gamma=0.99,
        lambda_gae=0.95,
        entropy_coef=0.0,
    )
    assert "clip_ratio" in metrics
    assert float(metrics["clip_ratio"]) == 0.0
    metrics["actor_loss"].backward()
    assert actor.mu_layer.weight.grad is not None
    assert actor.ls3c.in_proj.weight.grad is None


def test_dreamer_trainer_observe_is_not_ppo():
    torch.manual_seed(1)
    actor = _tiny_actor()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, device=torch.device("cpu"))
    trainer = DreamerTrainer(
        actor,
        normalizer,
        device="cpu",
        batch_length=2,
        imagination_horizon=2,
        world_hidden_sizes=HIDDEN,
    )
    assert not isinstance(trainer, PPOTrainer)
    n = 4
    obs = torch.rand(n, ACTOR_OBS_DIM)
    critic_obs = torch.rand(n, 392)
    trainer.initialize(obs, critic_obs)
    for _ in range(2):
        actions = trainer.act(obs, critic_obs)
        assert actions.shape == (n, NUM_ACTIONS)
        obs = torch.rand(n, ACTOR_OBS_DIM)
        metrics = trainer.observe(
            obs,
            critic_obs,
            torch.zeros(n),
            torch.zeros(n, dtype=torch.bool),
        )
    assert metrics is not None
    assert metrics["clip_ratio"] == 0.0
    assert "world_loss" in metrics
    assert "approx_kl" not in metrics
    deploy = trainer.deploy_state_dict()
    assert not any("decoder" in k for k in deploy)


def test_default_algorithm_is_not_lidar_dreamer():
    assert DEFAULT_CONFIG["algorithm"] != "lidar_dreamer"
    assert DEFAULT_CONFIG["model"]["actor_type"] != LIDAR_DREAMER_ARCHITECTURE_NAME

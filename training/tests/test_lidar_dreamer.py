from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, control_dt_s
from dreamer import DreamerTrainer, WorldModelHeads, imagine_actor_critic
from dreamer.world_model import world_model_losses
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
    ActorModels,
    build_config,
    load_resume_state,
    make_policy_network,
    parse_args,
    save_resume_state,
    validate_algorithm_config,
    validate_config_patch,
)

try:
    import warp  # noqa: F401

    HAS_WARP = True
except ImportError:
    HAS_WARP = False

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
    assert cfg["env"]["opponent_strategy"] == "scripted"
    assert cfg["schedule"]["total_transitions"] == 256000000
    assert cfg["env"]["action_horizon"] == 1
    assert cfg["env"]["num_actions"] == NUM_ACTIONS
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


def _tiny_trainer(**kwargs):
    actor = _tiny_actor()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, device=torch.device("cpu"))
    params = dict(
        device="cpu",
        batch_length=2,
        imagination_horizon=2,
        world_hidden_sizes=HIDDEN,
        replay_capacity=32,
        replay_batch_size=8,
    )
    params.update(kwargs)
    return DreamerTrainer(actor, normalizer, **params), actor, normalizer


def _fill_window(trainer, n):
    obs = torch.rand(n, ACTOR_OBS_DIM)
    critic_obs = torch.rand(n, 392)
    trainer.initialize(obs, critic_obs)
    metrics = None
    for _ in range(trainer.batch_length):
        trainer.act(obs, critic_obs)
        obs = torch.rand(n, ACTOR_OBS_DIM)
        metrics = trainer.observe(
            obs,
            critic_obs,
            torch.zeros(n),
            torch.zeros(n, dtype=torch.bool),
        )
    return metrics


def _has_adam_moments(optimizer):
    return any("exp_avg" in state for state in optimizer.state.values())


def _clone_params(module):
    return [p.detach().cpu().clone() for p in module.parameters()]


def _adam_exp_avg(optimizer):
    for state in optimizer.state.values():
        if "exp_avg" in state:
            return state["exp_avg"].detach().cpu().clone()
    raise AssertionError("optimizer has no Adam moments")


def test_replay_batch_shapes():
    torch.manual_seed(0)
    trainer, _actor, _norm = _tiny_trainer(batch_length=4, replay_batch_size=8)
    n = 3
    metrics = _fill_window(trainer, n)
    assert metrics is not None
    shape = trainer.last_batch_obs_shape
    assert shape is not None
    assert len(shape) == 3
    assert shape[0] == n
    assert shape[1] == 4
    assert shape[2] == ACTOR_OBS_DIM
    assert trainer._replay.size == n


def test_train_samples_latents_eval_uses_mean():
    torch.manual_seed(0)
    actor = _tiny_actor()
    obs = torch.rand(8, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(8)
    _, _, sampled_a = actor.step(
        obs, hidden, deterministic=False, with_logprob=False
    )
    _, _, sampled_b = actor.step(
        obs, hidden, deterministic=False, with_logprob=False
    )
    z_a = actor.unpack_hidden(sampled_a)[1]
    z_b = actor.unpack_hidden(sampled_b)[1]
    assert not torch.allclose(z_a, z_b)
    _h, z_mean, post_mean, _std, _embed = actor.rssm_observe(
        obs, hidden, deterministic=True
    )
    assert torch.allclose(z_mean, post_mean)
    _, _, eval_hidden = actor.step(
        obs, hidden, deterministic=True, with_logprob=False
    )
    z_eval = actor.unpack_hidden(eval_hidden)[1]
    assert torch.allclose(z_eval, post_mean)

    world = WorldModelHeads(actor, hidden_sizes=HIDDEN)
    h, z, prev_a = actor.unpack_hidden(hidden)
    action = torch.zeros(8, NUM_ACTIONS)
    torch.manual_seed(0)
    _h1, z1, _pm1, _ps1 = world.imagine_step(actor, h, z, action)
    torch.manual_seed(1)
    _h2, z2, _pm2, _ps2 = world.imagine_step(actor, h, z, action)
    assert not torch.allclose(z1, z2)
    _h3, z3, mean3, _ps = world.imagine_step(
        actor, h, z, action, deterministic=True
    )
    assert torch.allclose(z3, mean3)

    seq_obs = obs.unsqueeze(1).expand(-1, 2, -1).contiguous()
    seq_act = prev_a.unsqueeze(1).expand(-1, 2, -1).contiguous()
    rewards = torch.zeros(8, 2)
    dones = torch.zeros(8, 2, dtype=torch.bool)
    torch.manual_seed(0)
    out_a = world_model_losses(
        actor, world, seq_obs, seq_act, rewards, dones, hidden
    )
    torch.manual_seed(1)
    out_b = world_model_losses(
        actor, world, seq_obs, seq_act, rewards, dones, hidden
    )
    z_wm_a = actor.unpack_hidden(out_a["posterior_hidden"])[1]
    z_wm_b = actor.unpack_hidden(out_b["posterior_hidden"])[1]
    assert not torch.allclose(z_wm_a, z_wm_b)


def test_dreamer_resume_roundtrip(tmp_path):
    torch.manual_seed(0)
    src, src_actor, src_actor_norm = _tiny_trainer()
    src_critic_norm = ObsNormalizer(392, device=torch.device("cpu"))
    n = 4
    metrics = _fill_window(src, n)
    assert metrics is not None
    src_critic_norm.update(torch.rand(n, 392))
    assert _has_adam_moments(src.world_optimizer)
    assert _has_adam_moments(src.actor_optimizer)
    assert _has_adam_moments(src.critic_optimizer)

    ckpt = tmp_path / "resume_state.pt"
    save_resume_state(
        ckpt,
        algorithm="lidar_dreamer",
        models=ActorModels(actor=src_actor),
        trainer=src,
        actor_normalizer=src_actor_norm,
        critic_normalizer=src_critic_norm,
        env_transitions=256000000,
        gradient_updates=3,
        vector_ticks=64,
        sampled_replay_rows=12,
        device=torch.device("cpu"),
    )
    src_actor_params = _clone_params(src_actor)
    src_world_params = _clone_params(src.world)
    src_critic_params = _clone_params(src.critic)
    src_world_m = _adam_exp_avg(src.world_optimizer)
    src_actor_m = _adam_exp_avg(src.actor_optimizer)
    src_critic_m = _adam_exp_avg(src.critic_optimizer)
    src_actor_mean = src_actor_norm.mean.clone()
    src_critic_mean = src_critic_norm.mean.clone()

    torch.manual_seed(1)
    dst, dst_actor, dst_actor_norm = _tiny_trainer()
    dst_critic_norm = ObsNormalizer(392, device=torch.device("cpu"))
    assert not torch.equal(src_actor_params[0], _clone_params(dst_actor)[0])

    payload = load_resume_state(
        str(ckpt),
        algorithm="lidar_dreamer",
        models=ActorModels(actor=dst_actor),
        trainer=dst,
        actor_normalizer=dst_actor_norm,
        critic_normalizer=dst_critic_norm,
        device=torch.device("cpu"),
    )
    assert payload["env_transitions"] == 256000000
    for got, expected in zip(_clone_params(dst_actor), src_actor_params):
        assert torch.equal(got, expected)
    for got, expected in zip(_clone_params(dst.world), src_world_params):
        assert torch.equal(got, expected)
    for got, expected in zip(_clone_params(dst.critic), src_critic_params):
        assert torch.equal(got, expected)
    assert torch.equal(_adam_exp_avg(dst.world_optimizer), src_world_m)
    assert torch.equal(_adam_exp_avg(dst.actor_optimizer), src_actor_m)
    assert torch.equal(_adam_exp_avg(dst.critic_optimizer), src_critic_m)
    assert torch.equal(dst_actor_norm.mean, src_actor_mean)
    assert torch.equal(dst_critic_norm.mean, src_critic_mean)
    deploy = dst.deploy_state_dict()
    assert any(k.startswith("encoder.") for k in deploy)
    assert not any("decoder" in k for k in deploy)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "decoder.0.0.weight" in saved["world"] or any(
        "decoder" in key for key in saved["world"]
    )
    assert not any("decoder" in key for key in deploy)


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_cpu_smoke_one_update_few_env_steps(warp_runtime):
    del warp_runtime
    from test_warp_env import _make_env

    path = Path(__file__).resolve().parents[1] / (
        "configs/austin_e2e_lidar_dreamer.json"
    )
    patch = json.loads(path.read_text(encoding="utf-8"))
    cfg = _resolve(["--algorithm", "lidar_dreamer"], patch=patch)
    assert cfg["algorithm"] == "lidar_dreamer"
    assert control_dt_s(cfg["env"]) == pytest.approx(0.05)
    assert cfg["env"]["action_horizon"] == 1
    assert cfg["env"]["num_actions"] == 2

    actor = _tiny_actor()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, device=torch.device("cpu"))
    trainer = DreamerTrainer(
        actor,
        normalizer,
        device="cpu",
        batch_length=2,
        imagination_horizon=2,
        world_hidden_sizes=HIDDEN,
        replay_capacity=8,
        replay_batch_size=2,
    )
    assert isinstance(trainer, DreamerTrainer)
    assert not isinstance(trainer, PPOTrainer)

    env = _make_env(num_envs=2)
    try:
        raw, _ = env.reset(seed=0, with_sensors=True)
        actor_obs = raw["actor"].to(torch.float32)
        trainer.initialize(actor_obs)
        metrics = None
        for _ in range(2):
            actions = trainer.act(actor_obs)
            assert actions.shape == (2, 2)
            raw, reward, done, _extras = env.step(
                actions,
                n_steps=env.control_interval,
                with_sensors=True,
            )
            actor_obs = raw["actor"].to(torch.float32)
            metrics = trainer.observe(actor_obs, None, reward, done)
        assert metrics is not None
        for key in (
            "world_loss",
            "recon_loss",
            "kl_loss",
            "policy_loss",
            "value_loss",
        ):
            assert math.isfinite(metrics[key]), key
        assert metrics["clip_ratio"] == 0.0
        deploy = trainer.deploy_state_dict()
        assert any(k.startswith("encoder.") for k in deploy)
        assert any(k.startswith("ls3c.") for k in deploy)
        assert not any("decoder" in k for k in deploy)
    finally:
        env.close()

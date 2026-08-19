"""Focused tests for the recurrent LiDAR CNN + GRU actor and format-4 artifacts."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from qrsac import Models, QuantileCritic, SquashedGaussianLidarGRUActor, make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM, LIDAR_DIM, PROPRIO_DIM
from standalone_trainer import (
    SENSOR_POLICY_FORMAT_VERSION,
    ObsNormalizer,
    actor_architecture_from_module,
    make_policy_network,
    reference_actor_from_architecture,
    save_policy_artifact,
    validate_model_architecture,
    validate_sensor_policy_artifact,
)

OBS_DIM = LIDAR_DIM + PROPRIO_DIM
ACT_DIM = 2
HIDDEN = [32, 32]


def _make_gru_actor(seed: int = 0) -> SquashedGaussianLidarGRUActor:
    torch.manual_seed(seed)
    return make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=HIDDEN,
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )


def _stepwise_sequence(
    actor,
    obs,
    hidden,
    reset_mask=None,
    deterministic=False,
    with_logprob=True,
):
    actions = []
    logps = []
    h = hidden
    for t in range(obs.shape[1]):
        step_reset = None if reset_mask is None else reset_mask[:, t]
        action, logp, h = actor.step(
            obs[:, t],
            h,
            reset_mask=step_reset,
            deterministic=deterministic,
            with_logprob=with_logprob,
        )
        actions.append(action)
        if with_logprob:
            logps.append(logp)
    action_seq = torch.stack(actions, dim=1)
    logp_seq = torch.stack(logps, dim=1) if with_logprob else None
    return action_seq, logp_seq, h


def test_default_config_uses_recurrent_actor_schema():
    assert DEFAULT_CONFIG["policy_format_version"] == SENSOR_POLICY_FORMAT_VERSION == 4
    assert DEFAULT_CONFIG["obs"]["num_actor_obs"] == OBS_DIM == 1097
    assert DEFAULT_CONFIG["obs"]["num_obs"] == 392
    assert DEFAULT_CONFIG["model"]["actor_type"] == "lidar_cnn_gru"
    assert DEFAULT_CONFIG["model"]["actor_hidden_layers"] == [1024, 1024, 1024]
    assert DEFAULT_CONFIG["model"]["critic_hidden_layers"] == [1024, 1024, 1024]
    validate_model_architecture(DEFAULT_CONFIG)
    actor = make_policy_network(DEFAULT_CONFIG)
    assert isinstance(actor, SquashedGaussianLidarGRUActor)
    assert actor.gru_hidden_dim == 512
    assert actor.actor_architecture["name"] == "lidar_cnn_gru"
    assert actor.actor_architecture["gru_hidden_dim"] == 512
    assert actor.actor_architecture["obs_dim"] == 1097
    assert actor.actor_architecture["proprio_dim"] == 16


def test_pool64_projection512_policy_factory():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["lidar_pool_bins"] = 64
    cfg["model"]["lidar_projection_dim"] = 512

    validate_model_architecture(cfg)
    actor = make_policy_network(cfg)

    assert actor.pool_bins == 64
    assert actor.projection_dim == 512
    assert actor.gru_hidden_dim == 512
    assert actor.hidden_sizes == (1024, 1024, 1024)


def test_actor_shapes_step_and_sequence():
    actor = _make_gru_actor(1)
    batch, steps = 3, 5
    obs = torch.randn(batch, OBS_DIM)
    hidden = actor.initial_hidden(batch)
    assert hidden.shape == (batch, GRU_HIDDEN_DIM)

    action, logp, next_h = actor.step(obs, hidden, deterministic=True)
    assert action.shape == (batch, ACT_DIM)
    assert logp.shape == (batch,)
    assert next_h.shape == (batch, GRU_HIDDEN_DIM)

    seq = torch.randn(batch, steps, OBS_DIM)
    h0 = actor.initial_hidden(batch)
    actions, logps, h_t = actor.forward_sequence(seq, h0, deterministic=True)
    assert actions.shape == (batch, steps, ACT_DIM)
    assert logps.shape == (batch, steps)
    assert h_t.shape == (batch, GRU_HIDDEN_DIM)

    zero_h_action, _ = actor(obs, deterministic=True, with_logprob=False)
    assert zero_h_action.shape == (batch, ACT_DIM)


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(
    "reset_mask",
    [
        None,
        torch.zeros(3, 7, dtype=torch.bool),
        torch.tensor(
            [
                [True, False, False, True, False, False, True],
                [False, False, True, False, False, False, False],
                [False, False, False, False, True, False, False],
            ],
            dtype=torch.bool,
        ),
    ],
    ids=["none", "all-false", "mixed-first-middle-last"],
)
def test_batched_sequence_matches_stepwise(reset_mask, deterministic):
    actor = _make_gru_actor(2)
    actor.eval()
    batch, steps = 3, 7
    torch.manual_seed(11)
    seq = torch.randn(batch, steps, OBS_DIM)
    h0 = torch.randn(batch, GRU_HIDDEN_DIM)

    with torch.no_grad():
        torch.manual_seed(23)
        seq_actions, seq_logp, seq_h = actor.forward_sequence(
            seq,
            h0.clone(),
            reset_mask=reset_mask,
            deterministic=deterministic,
            with_logprob=True,
        )
        torch.manual_seed(23)
        step_actions, step_logps, step_h = _stepwise_sequence(
            actor,
            seq,
            h0.clone(),
            reset_mask=reset_mask,
            deterministic=deterministic,
            with_logprob=True,
        )

    assert torch.allclose(seq_actions, step_actions, atol=1e-6)
    assert torch.allclose(seq_logp, step_logps, atol=1e-6)
    assert torch.allclose(seq_h, step_h, atol=1e-6)


def test_batched_sequence_encodes_once_and_uses_static_cell_recurrence():
    actor = _make_gru_actor(3)
    batch, steps = 3, 7
    obs = torch.randn(batch, steps, OBS_DIM)
    hidden = torch.randn(batch, GRU_HIDDEN_DIM)
    reset = torch.zeros(batch, steps, dtype=torch.bool)
    reset[0, 0] = True
    reset[1, 2] = True
    reset[2, 5] = True
    conv_calls = 0
    gru_calls = 0

    def count_conv(*_):
        nonlocal conv_calls
        conv_calls += 1

    def count_gru(*_):
        nonlocal gru_calls
        gru_calls += 1

    conv_handle = actor.encoder.conv.register_forward_hook(count_conv)
    gru_handle = actor.gru.register_forward_hook(count_gru)
    try:
        actor.forward_sequence(
            obs,
            hidden,
            reset_mask=reset,
            deterministic=True,
            with_logprob=False,
        )
    finally:
        conv_handle.remove()
        gru_handle.remove()

    assert conv_calls == 1
    assert gru_calls == 0


def test_batched_sequence_gradient_parity_with_mixed_resets():
    batched = _make_gru_actor(4)
    stepwise = copy.deepcopy(batched)
    batch, steps = 2, 5
    obs_data = torch.randn(batch, steps, OBS_DIM)
    hidden_data = torch.randn(batch, GRU_HIDDEN_DIM)
    reset = torch.tensor(
        [
            [True, False, False, True, False],
            [False, False, True, False, True],
        ],
        dtype=torch.bool,
    )
    batched_obs = obs_data.clone().requires_grad_()
    batched_hidden = hidden_data.clone().requires_grad_()
    reference_obs = obs_data.clone().requires_grad_()
    reference_hidden = hidden_data.clone().requires_grad_()

    actions, logp, hidden = batched.forward_sequence(
        batched_obs,
        batched_hidden,
        reset_mask=reset,
        deterministic=True,
        with_logprob=True,
    )
    reference_actions, reference_logp, reference_h = _stepwise_sequence(
        stepwise,
        reference_obs,
        reference_hidden,
        reset_mask=reset,
        deterministic=True,
        with_logprob=True,
    )
    loss = actions.square().mean() + logp.mean() + hidden.square().mean()
    reference_loss = (
        reference_actions.square().mean()
        + reference_logp.mean()
        + reference_h.square().mean()
    )
    loss.backward()
    reference_loss.backward()

    assert torch.allclose(actions, reference_actions, atol=1e-6)
    assert torch.allclose(logp, reference_logp, atol=1e-6)
    assert torch.allclose(hidden, reference_h, atol=1e-6)
    assert torch.allclose(batched_obs.grad, reference_obs.grad, atol=1e-6)
    assert torch.allclose(batched_hidden.grad, reference_hidden.grad, atol=1e-6)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        batched.named_parameters(), stepwise.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert torch.allclose(
            parameter.grad,
            reference_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        ), name


def test_hidden_carry_and_reset_mask():
    actor = _make_gru_actor(3)
    actor.eval()
    batch = 2
    torch.manual_seed(12)
    obs_a = torch.randn(batch, OBS_DIM)
    obs_b = torch.randn(batch, OBS_DIM)
    h0 = actor.initial_hidden(batch)

    with torch.no_grad():
        _, _, h1 = actor.step(obs_a, h0, deterministic=True, with_logprob=False)
        action_carry, _, h2 = actor.step(
            obs_b, h1, deterministic=True, with_logprob=False
        )
        action_fresh, _, h_fresh = actor.step(
            obs_b, h0, deterministic=True, with_logprob=False
        )
        assert not torch.allclose(action_carry, action_fresh, atol=1e-5)
        assert not torch.allclose(h2, h_fresh, atol=1e-5)

        reset = torch.tensor([True, False])
        action_reset, _, h_reset = actor.step(
            obs_b, h1, reset_mask=reset, deterministic=True, with_logprob=False
        )
        assert torch.allclose(action_reset[0], action_fresh[0], atol=1e-6)
        assert torch.allclose(h_reset[0], h_fresh[0], atol=1e-6)
        assert torch.allclose(action_reset[1], action_carry[1], atol=1e-6)
        assert torch.allclose(h_reset[1], h2[1], atol=1e-6)

        seq = torch.stack([obs_a, obs_b], dim=1)
        reset_seq = torch.tensor([[False, True], [False, False]], dtype=torch.bool)
        actions, _, h_seq = actor.forward_sequence(
            seq, h0, reset_mask=reset_seq, deterministic=True, with_logprob=False
        )
        assert torch.allclose(actions[0, 1], action_fresh[0], atol=1e-6)
        assert torch.allclose(actions[1, 1], action_carry[1], atol=1e-6)
        assert torch.allclose(h_seq[0], h_fresh[0], atol=1e-6)
        assert torch.allclose(h_seq[1], h2[1], atol=1e-6)


def test_gradient_flows_cnn_gru_head():
    actor = _make_gru_actor(4)
    actor.train()
    batch, steps = 2, 4
    obs = torch.randn(batch, steps, OBS_DIM, requires_grad=False)
    hidden = actor.initial_hidden(batch)
    actions, logp, _ = actor.forward_sequence(
        obs, hidden, deterministic=False, with_logprob=True
    )
    loss = actions.pow(2).mean() + logp.mean()
    loss.backward()

    assert actor.encoder.conv[0].weight.grad is not None
    assert actor.encoder.conv[0].weight.grad.abs().sum() > 0
    assert actor.gru.weight_ih_l0.grad is not None
    assert actor.gru.weight_ih_l0.grad.abs().sum() > 0
    assert actor.gru.weight_hh_l0.grad is not None
    assert actor.gru.weight_hh_l0.grad.abs().sum() > 0
    assert actor.mu_layer.weight.grad is not None
    assert actor.mu_layer.weight.grad.abs().sum() > 0
    assert actor.log_std_layer.weight.grad is not None
    assert actor.net[0].weight.grad is not None


def test_artifact_round_trip_and_old_format_rejection(tmp_path):
    from evaluation import load_sensor_actor_bundle

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["actor_hidden_layers"] = HIDDEN
    cfg["model"]["critic_hidden_layers"] = [16, 16]
    cfg["model"]["lidar_pool_bins"] = 16
    cfg["model"]["num_quantiles"] = 4

    actor = make_policy_network(cfg)
    critic = QuantileCritic(cfg["obs"]["num_obs"], ACT_DIM, [16, 16], 4)
    models = Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )
    normalizer = ObsNormalizer(OBS_DIM, torch.device("cpu"))
    normalizer.update(torch.randn(5, OBS_DIM))

    path = save_policy_artifact(models, 42, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    loaded_actor, loaded_normalizer, loaded_arch, _ = load_sensor_actor_bundle(
        path,
        torch.device("cpu"),
        expected_actor_obs_dim=OBS_DIM,
        expected_action_dim=ACT_DIM,
        expected_layout_version=2,
        expected_critic_obs_dim=392,
        require_obs_norm=True,
    )
    assert payload["policy_format_version"] == 4
    assert payload["actor_architecture"]["name"] == "lidar_cnn_gru"
    assert payload["actor_architecture"]["gru_hidden_dim"] == 512
    assert payload["actor_obs_dim"] == 1097
    assert "gru.weight_ih_l0" in payload["actor"]
    assert loaded_actor.actor_architecture == loaded_arch
    assert loaded_normalizer.count == normalizer.count

    arch = actor_architecture_from_module(actor)
    validate_sensor_policy_artifact(
        payload,
        expected_actor_obs_dim=OBS_DIM,
        expected_action_dim=ACT_DIM,
        expected_layout_version=2,
        expected_architecture=arch,
        expected_critic_obs_dim=392,
    )
    restored = reference_actor_from_architecture(arch)
    restored.load_state_dict(payload["actor"], strict=True)
    obs = torch.randn(3, OBS_DIM)
    with torch.no_grad():
        expected, _ = actor(obs, deterministic=True, with_logprob=False)
        actual, _ = restored(obs, deterministic=True, with_logprob=False)
    assert torch.allclose(expected, actual, atol=1e-6)

    old = copy.deepcopy(payload)
    old["policy_format_version"] = 3
    with pytest.raises(ValueError, match="policy_format_version"):
        validate_sensor_policy_artifact(
            old,
            expected_actor_obs_dim=OBS_DIM,
            expected_action_dim=ACT_DIM,
            expected_layout_version=2,
            expected_architecture=arch,
        )

    other = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[8, 8],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=32,
    )
    stale_arch = actor_architecture_from_module(other)
    stale = copy.deepcopy(payload)
    stale["actor"] = other.state_dict()
    stale["actor_architecture"] = stale_arch
    with pytest.raises(ValueError, match="actor_architecture mismatch"):
        validate_sensor_policy_artifact(
            stale,
            expected_actor_obs_dim=OBS_DIM,
            expected_action_dim=ACT_DIM,
            expected_layout_version=2,
            expected_architecture=arch,
        )

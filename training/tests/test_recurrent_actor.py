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


def test_batched_sequence_encodes_once_and_runs_one_packed_gru():
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
    assert gru_calls == 1


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


def _gru_sequence_cell_loop(actor, features, hidden, reset_mask):
    """Pre-cuDNN per-timestep gru_cell path; used as the parity reference."""
    if reset_mask is None:
        out, h_n = actor.gru(features, hidden.unsqueeze(0).contiguous())
        return out, h_n.squeeze(0)
    outs = []
    h = hidden
    for t in range(features.shape[1]):
        h = actor._apply_reset_mask(h, reset_mask[:, t])
        h = actor.gru_cell(features[:, t], h)
        outs.append(h)
    return torch.stack(outs, dim=1), h


def _assert_close(actual, expected, name, rtol=1e-5, atol=1e-5):
    assert actual.shape == expected.shape, f"{name} shape {tuple(actual.shape)}"
    max_abs = (actual - expected).abs().max().item()
    denom = expected.abs().max().clamp(min=1e-8).item()
    rel = max_abs / denom
    assert rel <= rtol or max_abs <= atol, (
        f"{name}: max_abs={max_abs:.3e} rel={rel:.3e}"
    )


def _packed_vs_cell_case(device, seed=7, batch=4, steps=11):
    torch.manual_seed(seed)
    actor = _make_gru_actor(seed).to(device)
    actor.train()
    obs = torch.randn(batch, steps, OBS_DIM, device=device)
    hidden = torch.randn(batch, GRU_HIDDEN_DIM, device=device)
    reset = torch.zeros(batch, steps, dtype=torch.bool, device=device)
    reset[0, 0] = True
    reset[0, 4] = True
    reset[1, 2] = True
    reset[1, 10] = True
    reset[2, 7] = True
    reset[3, :] = False
    actions = torch.tanh(torch.randn(batch, steps, ACT_DIM, device=device)) * 0.9
    pre_tanh = torch.randn(batch, steps, ACT_DIM, device=device)

    packed = copy.deepcopy(actor)
    cell = copy.deepcopy(actor)

    def cell_sequence(features, hidden_s, reset_mask):
        return _gru_sequence_cell_loop(cell, features, hidden_s, reset_mask)

    cell._gru_sequence = cell_sequence

    obs_p = obs.clone().requires_grad_()
    hid_p = hidden.clone().requires_grad_()
    obs_c = obs.clone().requires_grad_()
    hid_c = hidden.clone().requires_grad_()

    logp_p, h_p, ent_p = packed.evaluate_actions_sequence(
        obs_p, hid_p, actions, reset_mask=reset, pre_tanh_actions=pre_tanh
    )
    logp_c, h_c, ent_c = cell.evaluate_actions_sequence(
        obs_c, hid_c, actions, reset_mask=reset, pre_tanh_actions=pre_tanh
    )
    act_p, lp_p, hf_p = packed.forward_sequence(
        obs, hidden, reset_mask=reset, deterministic=True, with_logprob=True
    )
    act_c, lp_c, hf_c = cell.forward_sequence(
        obs, hidden, reset_mask=reset, deterministic=True, with_logprob=True
    )
    loss_p = logp_p.square().mean() + ent_p.mean() + h_p.square().mean()
    loss_c = logp_c.square().mean() + ent_c.mean() + h_c.square().mean()
    loss_p.backward()
    loss_c.backward()

    _assert_close(logp_p, logp_c, "logp")
    _assert_close(ent_p, ent_c, "entropy")
    _assert_close(h_p, h_c, "eval_hidden")
    _assert_close(act_p, act_c, "actions")
    _assert_close(lp_p, lp_c, "forward_logp")
    _assert_close(hf_p, hf_c, "forward_hidden")
    _assert_close(obs_p.grad, obs_c.grad, "obs_grad")
    _assert_close(hid_p.grad, hid_c.grad, "hidden_grad")
    for (name, p_param), (_, c_param) in zip(
        packed.named_parameters(), cell.named_parameters(), strict=True
    ):
        assert p_param.grad is not None and c_param.grad is not None, name
        _assert_close(p_param.grad, c_param.grad, f"grad:{name}")


def test_packed_gru_parity_with_cell_loop_mid_sequence_resets():
    _packed_vs_cell_case(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_packed_gru_parity_with_cell_loop_cuda():
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        _packed_vs_cell_case(torch.device("cuda"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


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
    assert payload["policy_format_version"] == 4
    assert payload["actor_architecture"]["name"] == "lidar_cnn_gru"
    assert payload["actor_architecture"]["gru_hidden_dim"] == 512
    assert payload["actor_obs_dim"] == 1097
    assert "gru.weight_ih_l0" in payload["actor"]

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

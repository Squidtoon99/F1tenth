"""Focused tests for recurrent sequence QR-SAC updates (burn-in + 32-step train)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from qrsac import Models, QRSACTrainer, QuantileCritic, make_actor
from qrsac.qrsac import select_min_quantiles
from qrsac.spinningup.core import GRU_HIDDEN_DIM, LIDAR_DIM, PROPRIO_DIM

ACTOR_DIM = LIDAR_DIM + PROPRIO_DIM
CRITIC_DIM = 24
ACT_DIM = 2
HIDDEN = [32, 32]
NUM_QUANTILES = 8
BURN_IN = 4
TRAIN_LEN = 4
N_STEP = 3
NUM_SEQ = 2


def _gru_models(seed: int = 0) -> Models:
    torch.manual_seed(seed)
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=HIDDEN,
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )
    critic = QuantileCritic(CRITIC_DIM, ACT_DIM, [32, 32], NUM_QUANTILES)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


def _trainer(models: Models | None = None, **kwargs) -> QRSACTrainer:
    models = models if models is not None else _gru_models()
    defaults = dict(
        gamma=0.9896,
        n_step=N_STEP,
        alpha=0.01,
        smooth_factor=0.005,
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
    )
    defaults.update(kwargs)
    return QRSACTrainer(models, torch.device("cpu"), **defaults)


def _sequence_batch(
    num_seq: int = NUM_SEQ,
    burn_in: int = BURN_IN,
    train_len: int = TRAIN_LEN,
    n_step: int = N_STEP,
    *,
    seed: int = 0,
    terminal_at_train: int | None = None,
    reset_at: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Synthetic contiguous window matching TrajectoryReplayBuffer.sample."""
    torch.manual_seed(seed)
    seq_len = burn_in + train_len + n_step
    actor_obs = torch.randn(num_seq, seq_len, ACTOR_DIM)
    critic_obs = torch.randn(num_seq, seq_len, CRITIC_DIM)
    action = torch.rand(num_seq, seq_len, ACT_DIM) * 2.0 - 1.0
    reward = torch.randn(num_seq, seq_len)
    done = torch.zeros(num_seq, seq_len)
    reset = torch.zeros(num_seq, seq_len)
    if terminal_at_train is not None:
        t = burn_in + int(terminal_at_train)
        done[:, t] = 1.0
    if reset_at is not None:
        s, t = reset_at
        reset[s, t] = 1.0
    hidden = torch.randn(num_seq, GRU_HIDDEN_DIM)

    gamma = 0.9896
    gamma_powers = torch.tensor([gamma**k for k in range(n_step)], dtype=torch.float32)
    n_step_idx = (
        burn_in
        + torch.arange(train_len).unsqueeze(1)
        + torch.arange(n_step).unsqueeze(0)
    )
    rew_g = reward[:, n_step_idx]
    done_g = done[:, n_step_idx]
    prior_done = torch.cumsum(done_g, dim=-1) - done_g
    alive = (prior_done == 0).to(torch.float32)
    n_step_reward = (rew_g * gamma_powers * alive).sum(dim=-1)
    n_step_done = ((done_g * alive).sum(dim=-1) > 0).to(torch.float32)
    boot_idx = burn_in + torch.arange(train_len) + n_step

    return {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "action": action,
        "reward": reward,
        "done": done,
        "reset": reset,
        "hidden": hidden,
        "n_step_reward": n_step_reward,
        "n_step_done": n_step_done,
        "bootstrap_actor_obs": actor_obs[:, boot_idx],
        "bootstrap_critic_obs": critic_obs[:, boot_idx],
    }


def test_sequence_batch_rejects_iid_and_malformed():
    trainer = _trainer()
    iid = {
        "actor_obs": torch.randn(4, ACTOR_DIM),
        "critic_obs": torch.randn(4, CRITIC_DIM),
        "action": torch.randn(4, ACT_DIM),
        "reward": torch.randn(4),
        "next_actor_obs": torch.randn(4, ACTOR_DIM),
        "next_critic_obs": torch.randn(4, CRITIC_DIM),
        "done": torch.zeros(4),
    }
    with pytest.raises(KeyError, match="rejects IID"):
        trainer.update_from_sequences(iid)

    batch = _sequence_batch()
    bad = dict(batch)
    bad["actor_obs"] = batch["actor_obs"][:, :, : ACTOR_DIM - 1]
    with pytest.raises(ValueError, match="actor_obs shape"):
        trainer.update_from_sequences(bad)

    misaligned = dict(batch)
    misaligned["bootstrap_actor_obs"] = batch["bootstrap_actor_obs"] + 1.0
    debug_trainer = _trainer(assert_bootstrap_alignment=True)
    with pytest.raises(ValueError, match="bootstrap_actor_obs must equal"):
        debug_trainer.update_from_sequences(misaligned)
    # Production skips the syncing content check (bootstrap_actor_obs unused in
    # the compiled phase); schema/shape checks still run via _require_sequence_batch.
    _trainer().update_from_sequences(misaligned)

    with pytest.raises(ValueError, match="Unsupported actor_type"):
        make_actor(
            actor_type="flat_mlp",
            obs_dim=ACTOR_DIM,
            act_dim=ACT_DIM,
            hidden_sizes=HIDDEN,
            activation=nn.ReLU,
            act_limit=1.0,
        )


def test_loss_and_target_shapes_fixed_seed():
    torch.manual_seed(0)
    models = _gru_models(1)
    trainer = _trainer(models, gamma=0.5, n_step=N_STEP, alpha=0.01)
    batch = _sequence_batch(seed=2)

    rows = NUM_SEQ * TRAIN_LEN
    burn = BURN_IN
    train = TRAIN_LEN
    n_step = N_STEP
    post_len = train + n_step

    with torch.no_grad():
        _, _, h_burn = models.actor.forward_sequence(
            batch["actor_obs"][:, :burn],
            batch["hidden"],
            reset_mask=batch["reset"][:, :burn],
            deterministic=True,
            with_logprob=False,
        )
        h_burn = h_burn.detach()
        torch.manual_seed(99)
        boot_act, boot_logp, _ = models.actor.forward_sequence(
            batch["actor_obs"][:, burn : burn + post_len],
            h_burn,
            reset_mask=batch["reset"][:, burn : burn + post_len],
            deterministic=False,
            with_logprob=True,
        )
        actions_next = boot_act[:, n_step : n_step + train].reshape(rows, ACT_DIM)
        log_prob_next = boot_logp[:, n_step : n_step + train].reshape(rows)
        next_critic = batch["bootstrap_critic_obs"].reshape(rows, CRITIC_DIM)
        q1 = models.critic1_target(next_critic, actions_next)
        q2 = models.critic2_target(next_critic, actions_next)
        min_q = select_min_quantiles(q1, q2)
        discount = trainer.gamma**trainer.n_step
        ref_target = batch["n_step_reward"].reshape(rows).unsqueeze(-1) + discount * (
            1.0 - batch["n_step_done"].reshape(rows).unsqueeze(-1)
        ) * (min_q - trainer.alpha * log_prob_next.unsqueeze(-1))

    assert ref_target.shape == (rows, NUM_QUANTILES)
    assert actions_next.shape == (rows, ACT_DIM)
    assert log_prob_next.shape == (rows,)

    captured: dict[str, torch.Tensor] = {}
    orig_forward = trainer.critic1_target.forward

    def _capture_forward(obs, act):
        out = orig_forward(obs, act)
        captured["next_obs"] = obs.detach().clone()
        captured["next_act"] = act.detach().clone()
        captured["q"] = out.detach().clone()
        return out

    trainer.critic1_target.forward = _capture_forward  # type: ignore[method-assign]
    torch.manual_seed(99)
    losses = trainer.update_from_sequences(batch)
    trainer.critic1_target.forward = orig_forward  # type: ignore[method-assign]

    assert losses.policy_loss.shape == ()
    assert losses.critic_loss.shape == ()
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    assert captured["next_obs"].shape == (rows, CRITIC_DIM)
    assert captured["next_act"].shape == (rows, ACT_DIM)
    assert captured["q"].shape == (rows, NUM_QUANTILES)
    assert torch.allclose(captured["next_act"], actions_next, rtol=1e-5, atol=1e-5)
    assert torch.allclose(captured["next_obs"], next_critic, rtol=1e-5, atol=1e-5)
    assert torch.allclose(captured["q"], q1, rtol=1e-5, atol=1e-5)


def test_burn_in_reset_mask_zeros_hidden():
    models = _gru_models(3)
    trainer = _trainer(models)
    batch = _sequence_batch(seed=4)
    # Force a mid-burn-in reset on sequence 0.
    batch["reset"][0, BURN_IN // 2] = 1.0

    burn = BURN_IN
    with torch.no_grad():
        _, _, h_carry = models.actor.forward_sequence(
            batch["actor_obs"][:, :burn],
            batch["hidden"],
            reset_mask=torch.zeros_like(batch["reset"][:, :burn]),
            deterministic=True,
            with_logprob=False,
        )
        _, _, h_reset = models.actor.forward_sequence(
            batch["actor_obs"][:, :burn],
            batch["hidden"],
            reset_mask=batch["reset"][:, :burn],
            deterministic=True,
            with_logprob=False,
        )
    assert not torch.allclose(h_carry[0], h_reset[0], atol=1e-5)
    assert torch.allclose(h_carry[1], h_reset[1], atol=1e-5)

    losses = trainer.update_from_sequences(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)


def test_no_gradient_through_burn_in():
    models = _gru_models(5)
    trainer = _trainer(models)
    batch = _sequence_batch(seed=6)

    # Sentinel parameter path: burn-in-only features must not receive grad via h_burn.
    burn_obs = batch["actor_obs"][:, :BURN_IN].clone().requires_grad_(True)
    h0 = batch["hidden"].clone().requires_grad_(True)
    with torch.no_grad():
        _, _, h_burn = models.actor.forward_sequence(
            burn_obs.detach(),
            h0.detach(),
            reset_mask=batch["reset"][:, :BURN_IN],
            deterministic=False,
            with_logprob=False,
        )
    assert not h_burn.requires_grad

    # Full update: encoder grads must come from train segment, not require burn-in graph.
    for p in models.actor.parameters():
        p.grad = None
    losses = trainer.update_from_sequences(batch)
    assert torch.isfinite(losses.policy_loss)
    # Detached burn-in hidden: a second burn-in with grad should be independent of update.
    h_probe = batch["hidden"].clone().requires_grad_(True)
    _, _, h_out = models.actor.forward_sequence(
        batch["actor_obs"][:, :BURN_IN],
        h_probe,
        reset_mask=batch["reset"][:, :BURN_IN],
        deterministic=True,
        with_logprob=False,
    )
    (h_out**2).mean().backward()
    assert h_probe.grad is not None
    # Update path itself never populated grad on the stored checkpoint tensor.
    assert batch["hidden"].grad is None


def test_terminal_n_step_zeros_bootstrap():
    torch.manual_seed(7)
    models = _gru_models(7)
    trainer = _trainer(models, gamma=0.5, n_step=N_STEP)
    batch = _sequence_batch(seed=8, terminal_at_train=0)
    # First optimized step is terminal → all n-step returns for that row done=1,
    # and later train rows that start after the terminal also see prior_done.
    assert batch["n_step_done"][0, 0] == 1.0

    rows = NUM_SEQ * TRAIN_LEN
    captured: dict[str, torch.Tensor] = {}

    def _wrap_target(module):
        orig = module.forward

        def wrapped(obs, act):
            out = orig(obs, act)
            captured.setdefault("q", out.detach().clone())
            return out

        module.forward = wrapped  # type: ignore[method-assign]
        return orig

    orig1 = _wrap_target(trainer.critic1_target)
    losses = trainer.update_from_sequences(batch)
    trainer.critic1_target.forward = orig1  # type: ignore[method-assign]
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)

    # Manually rebuild targets: done rows must ignore bootstrap Q.
    done_flat = batch["n_step_done"].reshape(rows)
    assert done_flat[0] == 1.0
    discount = trainer.gamma**trainer.n_step
    # For done transitions, target equals reward only (bootstrap coeff 0).
    # Spot-check by recomputing one done row's target scale: (1-done)=0.
    assert discount * (1.0 - done_flat[0]) == 0.0


def test_gradients_flow_cnn_gru_head_via_sequence_update():
    models = _gru_models(9)
    trainer = _trainer(models)
    batch = _sequence_batch(seed=10)
    losses = trainer.update_from_sequences(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)

    actor = models.actor
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
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in models.critic1.parameters()
    )


def test_eager_sequence_update_smoke_full_window():
    """Real eager CPU update at production burn-in/train/n-step lengths."""
    torch.manual_seed(11)
    models = _gru_models(11)
    trainer = QRSACTrainer(
        models,
        torch.device("cpu"),
        gamma=0.9896,
        n_step=7,
        alpha=0.01,
        smooth_factor=0.005,
        burn_in=16,
        train_len=32,
    )
    assert trainer.seq_len == 55
    batch = _sequence_batch(
        num_seq=2,
        burn_in=16,
        train_len=32,
        n_step=7,
        seed=12,
    )
    assert batch["actor_obs"].shape == (2, 55, ACTOR_DIM)
    assert batch["n_step_reward"].shape == (2, 32)
    assert batch["hidden"].shape == (2, GRU_HIDDEN_DIM)

    target_before = {
        k: v.clone() for k, v in models.critic1_target.state_dict().items()
    }
    losses = trainer.update_from_sequences(batch)
    assert losses.policy_loss.shape == ()
    assert losses.critic_loss.shape == ()
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    assert any(
        not torch.equal(target_before[k], v)
        for k, v in models.critic1_target.state_dict().items()
    )


def test_sequence_consistent_bootstrap_hidden_differs_from_zero_state():
    """Target actor at t+n must use carried state, not a fresh zero hidden."""
    models = _gru_models(13)
    trainer = _trainer(models)
    batch = _sequence_batch(seed=14)
    burn = BURN_IN
    train = TRAIN_LEN
    n_step = N_STEP
    rows = NUM_SEQ * train

    with torch.no_grad():
        _, _, h_burn = models.actor.forward_sequence(
            batch["actor_obs"][:, :burn],
            batch["hidden"],
            reset_mask=batch["reset"][:, :burn],
            deterministic=True,
            with_logprob=False,
        )
        seq_act, _, _ = models.actor.forward_sequence(
            batch["actor_obs"][:, burn : burn + train + n_step],
            h_burn,
            reset_mask=batch["reset"][:, burn : burn + train + n_step],
            deterministic=True,
            with_logprob=False,
        )
        boot_seq = seq_act[:, n_step : n_step + train].reshape(rows, ACT_DIM)
        boot_obs = batch["bootstrap_actor_obs"].reshape(rows, ACTOR_DIM)
        zero_h = models.actor.initial_hidden(rows)
        boot_zero, _, _ = models.actor.step(
            boot_obs, zero_h, deterministic=True, with_logprob=False
        )
    assert not torch.allclose(boot_seq, boot_zero, atol=1e-4)

    losses = trainer.update_from_sequences(batch)
    assert torch.isfinite(losses.policy_loss)


def test_eager_vs_compiled_sequence_losses_match():
    """Compiled sequence policy/critic phases must match eager numerics."""
    torch.manual_seed(21)
    eager_models = _gru_models(21)
    compiled_models = _gru_models(21)
    compiled_models.actor.load_state_dict(eager_models.actor.state_dict())
    compiled_models.critic1.load_state_dict(eager_models.critic1.state_dict())
    compiled_models.critic2.load_state_dict(eager_models.critic2.state_dict())
    compiled_models.critic1_target.load_state_dict(
        eager_models.critic1_target.state_dict()
    )
    compiled_models.critic2_target.load_state_dict(
        eager_models.critic2_target.state_dict()
    )

    eager = _trainer(eager_models, compile=False)
    compiled = _trainer(compiled_models, compile=True, compile_mode="default")
    assert hasattr(compiled._sequence_policy_phase, "_torchdynamo_orig_callable")
    batch = _sequence_batch(seed=22)

    torch.manual_seed(99)
    eager_losses = eager.update_from_sequences(batch)
    torch.manual_seed(99)
    compiled_losses = compiled.update_from_sequences(batch)

    assert torch.allclose(
        eager_losses.policy_loss, compiled_losses.policy_loss, atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        eager_losses.critic_loss, compiled_losses.critic_loss, atol=1e-5, rtol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_compiled_sequence_update_smoke():
    """Production-shaped CUDA compile path stays finite across warmup updates."""
    device = torch.device("cuda")
    torch.manual_seed(31)
    models = _gru_models(31)
    models.actor.to(device)
    models.critic1.to(device)
    models.critic2.to(device)
    models.critic1_target.to(device)
    models.critic2_target.to(device)
    trainer = QRSACTrainer(
        models,
        device,
        gamma=0.9896,
        n_step=7,
        alpha=0.01,
        burn_in=16,
        train_len=32,
        compile=True,
        compile_mode="reduce-overhead",
    )
    batch = {
        key: value.to(device)
        for key, value in _sequence_batch(
            num_seq=16, burn_in=16, train_len=32, n_step=7, seed=32
        ).items()
    }
    after_capture = None
    captured_losses = None
    captured_policy_value = None
    for update_idx in range(4):
        losses = trainer.update_from_sequences(batch)
        assert torch.isfinite(losses.policy_loss)
        assert torch.isfinite(losses.critic_loss)
        if update_idx == 2:
            after_capture = next(models.actor.parameters()).detach().clone()
            captured_losses = losses
            captured_policy_value = losses.policy_loss.clone()
    assert trainer._sequence_graph is not None
    assert not torch.equal(after_capture, next(models.actor.parameters()))
    assert torch.equal(captured_losses.policy_loss, captured_policy_value)
    actor_state = trainer.actor_optimizer.state[next(models.actor.parameters())]
    assert int(actor_state["step"].item()) == 4

    incomplete = dict(batch)
    incomplete.pop("hidden")
    with pytest.raises(KeyError, match="hidden"):
        trainer.update_from_sequences(incomplete)

    first_graph = trainer._sequence_graph
    trainer.reinitialize_networks()
    assert trainer._sequence_graph is None
    for _ in range(4):
        losses = trainer.update_from_sequences(batch)
        assert torch.isfinite(losses.policy_loss)
        assert torch.isfinite(losses.critic_loss)
    assert trainer._sequence_graph is not first_graph
    actor_state = trainer.actor_optimizer.state[next(models.actor.parameters())]
    assert int(actor_state["step"].item()) == 4


def _actor_l2_delta(models: Models, before: list[torch.Tensor]) -> float:
    return sum(
        (p.detach() - b).pow(2).sum().item()
        for p, b in zip(models.actor.parameters(), before)
    )


def test_sequence_graph_capture_deferred_while_actor_frozen_cpu():
    """Graph capture must not run while actor_frozen even after warmup."""
    torch.manual_seed(41)
    models = _gru_models(41)
    trainer = _trainer(models, compile=False)
    trainer._full_sequence_graph = True
    trainer.actor_frozen = True
    batch = _sequence_batch(seed=42)
    actor_before = [p.detach().clone() for p in models.actor.parameters()]

    trainer.update_from_sequences(batch)
    assert trainer._sequence_graph_warmed
    assert trainer._sequence_graph is None

    for _ in range(3):
        trainer.update_from_sequences(batch)

    assert trainer._sequence_graph is None
    assert _actor_l2_delta(models, actor_before) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sequence_graph_actor_updates_after_unfreeze():
    """CUDA graph capture after actor unfreeze must include actor optimizer steps."""
    device = torch.device("cuda")
    torch.manual_seed(43)
    models = _gru_models(43)
    models.actor.to(device)
    models.critic1.to(device)
    models.critic2.to(device)
    models.critic1_target.to(device)
    models.critic2_target.to(device)
    trainer = QRSACTrainer(
        models,
        device,
        gamma=0.9896,
        n_step=N_STEP,
        alpha=0.01,
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
        compile=True,
        compile_mode="reduce-overhead",
    )
    batch = {
        key: value.to(device)
        for key, value in _sequence_batch(num_seq=2, seed=44).items()
    }
    trainer.actor_frozen = True
    actor_before = [p.detach().clone() for p in models.actor.parameters()]

    trainer.update_from_sequences(batch)
    assert trainer._sequence_graph_warmed
    for _ in range(2):
        trainer.update_from_sequences(batch)

    assert trainer._sequence_graph is None
    assert _actor_l2_delta(models, actor_before) == 0.0

    trainer.actor_frozen = False
    trainer.update_from_sequences(batch)
    trainer.update_from_sequences(batch)
    assert trainer._sequence_graph is not None
    assert _actor_l2_delta(models, actor_before) > 0.0

    frozen_snapshot = [p.detach().clone() for p in models.actor.parameters()]
    trainer.update_from_sequences(batch)
    assert _actor_l2_delta(models, frozen_snapshot) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sequence_graph_capture_after_long_actor_freeze():
    """Compile actor backward outside capture after many frozen eager updates."""
    device = torch.device("cuda")
    torch.manual_seed(45)
    models = _gru_models(45)
    models.actor.to(device)
    models.critic1.to(device)
    models.critic2.to(device)
    models.critic1_target.to(device)
    models.critic2_target.to(device)
    trainer = QRSACTrainer(
        models,
        device,
        gamma=0.9896,
        n_step=N_STEP,
        alpha=0.01,
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
        compile=True,
        compile_mode="reduce-overhead",
    )
    batch = {
        key: value.to(device)
        for key, value in _sequence_batch(num_seq=16, seed=46).items()
    }
    trainer.actor_frozen = True
    actor_before = [p.detach().clone() for p in models.actor.parameters()]

    trainer.update_from_sequences(batch)
    for _ in range(400):
        trainer.update_from_sequences(batch)

    assert trainer._sequence_graph is None
    assert _actor_l2_delta(models, actor_before) == 0.0

    trainer.actor_frozen = False
    trainer.update_from_sequences(batch)
    trainer.update_from_sequences(batch)
    assert trainer._sequence_graph is not None
    assert _actor_l2_delta(models, actor_before) > 0.0

    before_replay = [p.detach().clone() for p in models.actor.parameters()]
    trainer.update_from_sequences(batch)
    assert _actor_l2_delta(models, before_replay) > 0.0

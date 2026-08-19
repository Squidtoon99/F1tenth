"""Tests for format-4 LiDAR-CNN+GRU deploy loader and training parity."""

from __future__ import annotations

import copy
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
from f1tenth_policy import assert_artifact_current_limits_match

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.policy_model import (
    SquashedGaussianLidarGRUActor,
    load_sensor_actor,
    load_sensor_obs_norm,
    make_sensor_actor,
    validate_sensor_policy_artifact,
)


def _default_architecture():
    actor = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    return copy.deepcopy(actor.actor_architecture)


def _sensor_payload(actor: SquashedGaussianLidarGRUActor, **extra):
    architecture = actor.actor_architecture
    payload = {
        "actor": actor.state_dict(),
        "obs_norm": {
            "mean": torch.zeros(si.NUM_OBS),
            "var": torch.ones(si.NUM_OBS),
            "count": torch.tensor(100.0),
        },
        "obs_dim": si.NUM_OBS,
        "actor_obs_dim": si.NUM_OBS,
        "critic_obs_dim": 392,
        "actor_layout_version": si.ACTOR_LAYOUT_VERSION,
        "action_dim": si.NUM_ACTIONS,
        "policy_format_version": si.POLICY_FORMAT_VERSION,
        "observation_preprocessing_version": si.OBS_PREPROCESSING_VERSION,
        "artifact_scope": si.ARTIFACT_SCOPE_SIM_TRAINING,
        "actor_architecture": architecture,
        "longitudinal_mode": "force",
        "steering_action_mode": si.STEERING_ACTION_MODE,
        "steering_delta_max_rad": float(si.STEERING_DELTA_MAX_RAD),
        "control_hz": si.CONTROL_HZ,
        "i_drive_max_a": 80.0,
        "i_brake_max_a": 20.0,
        "i_slew_a_per_s": 200.0,
    }
    payload.update(extra)
    return payload


def test_format4_checkpoint_round_trip():
    actor = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sensor.pt")
        torch.save(_sensor_payload(actor), path)
        loaded = load_sensor_actor(path, "actor", torch.device("cpu"))
        norm = load_sensor_obs_norm(
            path, torch.device("cpu"), si.OBS_NORM_EPS, si.OBS_NORM_CLIP
        )
    obs = torch.randn(3, si.NUM_OBS)
    with torch.no_grad():
        expected, _ = actor(obs, deterministic=True, with_logprob=False)
        actual, _ = loaded(obs, deterministic=True, with_logprob=False)
        normed = norm.normalize(obs)
        from_norm, _ = loaded(normed, deterministic=True, with_logprob=False)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert norm is not None
    assert from_norm.shape == (3, si.NUM_ACTIONS)


def test_sensor_policy_current_envelope_matches_runtime():
    actor = make_sensor_actor(_default_architecture())
    payload = _sensor_payload(actor)

    assert_artifact_current_limits_match(payload, 80.0, 20.0)


def test_sensor_policy_rejects_old_current_envelope():
    actor = make_sensor_actor(_default_architecture())
    payload = _sensor_payload(actor, i_drive_max_a=100.0, i_brake_max_a=10.0)

    with pytest.raises(ValueError, match="does not match artifact training scale"):
        assert_artifact_current_limits_match(payload, 80.0, 20.0)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"policy_format_version": 3}, "policy_format_version"),
        ({"observation_preprocessing_version": 2}, "observation_preprocessing_version"),
        ({"artifact_scope": "deployable"}, "artifact_scope"),
        ({}, "obs_norm"),
        ({}, "i_drive_max_a"),
    ],
)
def test_sensor_loader_rejects_invalid_artifacts(mutation, match):
    actor = make_sensor_actor(_default_architecture())
    payload = _sensor_payload(actor)
    if match == "obs_norm":
        del payload["obs_norm"]
    elif match == "i_drive_max_a":
        del payload["i_drive_max_a"]
    else:
        payload.update(mutation)
    with pytest.raises(ValueError, match=match):
        validate_sensor_policy_artifact(
            payload, expected_architecture=actor.actor_architecture
        )


def test_sensor_loader_rejects_architecture_mismatch():
    actor = make_sensor_actor(_default_architecture())
    payload = _sensor_payload(actor)
    bad = copy.deepcopy(payload)
    bad["actor_architecture"] = {
        **actor.actor_architecture,
        "pool_bins": 16,
    }
    with pytest.raises(ValueError, match="actor_architecture mismatch"):
        validate_sensor_policy_artifact(
            bad, expected_architecture=actor.actor_architecture
        )


def test_delta_sensor_loader_rejects_absolute_or_wrong_delta_metadata():
    actor = make_sensor_actor(_default_architecture())
    absolute = _sensor_payload(actor, steering_action_mode="absolute")
    with pytest.raises(ValueError, match="steering_action_mode"):
        validate_sensor_policy_artifact(
            absolute,
            expected_architecture=actor.actor_architecture,
            expected_steering_action_mode="delta",
            expected_steering_delta_max_rad=0.05235987755982988,
        )
    delta = _sensor_payload(
        actor,
        steering_action_mode="delta",
        steering_delta_max_rad=0.04,
    )
    with pytest.raises(ValueError, match="steering_delta_max_rad"):
        validate_sensor_policy_artifact(
            delta,
            expected_architecture=actor.actor_architecture,
            expected_steering_action_mode="delta",
            expected_steering_delta_max_rad=0.05235987755982988,
        )


def _repo_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )


def _real_sensor_checkpoint() -> str:
    path = os.path.join(
        _repo_root(),
        "training",
        "outputs",
        "runs",
        "7210e365",
        "checkpoints",
        "policy_271360000.pt",
    )
    if not os.path.isfile(path):
        pytest.skip(f"ignored real checkpoint not present: {path}")
    return path


def test_deploy_lidar_gru_actor_matches_training_core():
    training_root = os.path.join(_repo_root(), "training")
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    qrsac = pytest.importorskip("qrsac")
    TrainActor = qrsac.spinningup.core.SquashedGaussianLidarGRUActor

    deploy = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    train = TrainActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    deploy.load_state_dict(train.state_dict())
    rng = torch.manual_seed(0)
    obs = torch.randn(4, si.NUM_OBS, generator=rng)
    with torch.no_grad():
        d_out, _ = deploy(obs, deterministic=True, with_logprob=False)
        t_out, _ = train(obs, deterministic=True, with_logprob=False)
    assert torch.allclose(d_out, t_out, atol=1e-6)


@pytest.mark.parametrize("deterministic", [True, False])
def test_deploy_sequence_matches_training_core_with_resets(deterministic):
    training_root = os.path.join(_repo_root(), "training")
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    qrsac = pytest.importorskip("qrsac")
    TrainActor = qrsac.spinningup.core.SquashedGaussianLidarGRUActor

    deploy = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    train = TrainActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    deploy.load_state_dict(train.state_dict())
    torch.manual_seed(5)
    obs = torch.randn(3, 7, si.NUM_OBS)
    hidden = torch.randn(3, si.GRU_HIDDEN_DIM)
    reset = torch.tensor(
        [
            [True, False, False, True, False, False, True],
            [False, False, True, False, False, False, False],
            [False, False, False, False, True, False, False],
        ],
        dtype=torch.bool,
    )

    torch.manual_seed(7)
    deploy_action, deploy_logp, deploy_hidden = deploy.forward_sequence(
        obs,
        hidden,
        reset_mask=reset,
        deterministic=deterministic,
        with_logprob=True,
    )
    torch.manual_seed(7)
    train_action, train_logp, train_hidden = train.forward_sequence(
        obs,
        hidden,
        reset_mask=reset,
        deterministic=deterministic,
        with_logprob=True,
    )

    assert torch.allclose(deploy_action, train_action, atol=1e-6)
    assert torch.allclose(deploy_logp, train_logp, atol=1e-6)
    assert torch.allclose(deploy_hidden, train_hidden, atol=1e-6)


def test_deploy_step_hidden_matches_training_core_over_fixed_sequence():
    training_root = os.path.join(_repo_root(), "training")
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    qrsac = pytest.importorskip("qrsac")
    TrainActor = qrsac.spinningup.core.SquashedGaussianLidarGRUActor

    deploy = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    train = TrainActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        pool_bins=32,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    deploy.load_state_dict(train.state_dict())

    torch.manual_seed(11)
    obs_seq = torch.randn(5, si.NUM_OBS)
    deploy_hidden = deploy.initial_hidden(1)
    train_hidden = train.initial_hidden(1)
    for t in range(obs_seq.shape[0]):
        step_obs = obs_seq[t : t + 1]
        with torch.no_grad():
            d_action, _, deploy_hidden = deploy.step(
                step_obs, deploy_hidden, deterministic=True, with_logprob=False
            )
            t_action, _, train_hidden = train.step(
                step_obs, train_hidden, deterministic=True, with_logprob=False
            )
        assert torch.allclose(d_action, t_action, atol=1e-6)
        assert torch.allclose(deploy_hidden, train_hidden, atol=1e-6)


def test_real_checkpoint_loader_rejects_pre_break_layout():
    """Pinned pre-break checkpoints must be rejected after the schema break."""
    ckpt = _real_sensor_checkpoint()
    device = torch.device("cpu")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    assert int(payload["actor_obs_dim"]) == 1093
    reject = (
        r"policy_format_version|expected actor dim 1097|actor_layout_version|"
        r"actor_obs_dim"
    )
    with pytest.raises(ValueError, match=reject):
        load_sensor_actor(ckpt, "actor", device)
    with pytest.raises(ValueError, match=reject):
        load_sensor_obs_norm(ckpt, device, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)

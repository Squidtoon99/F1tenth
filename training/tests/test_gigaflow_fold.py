"""Fold Gigaflow ConditionedLidarGRUActor into Sophy's 1097-D lidar_cnn_gru."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from f1tenth_policy.artifact import validate_sensor_policy_artifact
from f1tenth_policy.layout import ACTOR_ARCHITECTURE_NAME, ACTOR_OBS_DIM
from f1tenth_policy.normalizer import load_obs_normalizer
from gigaflow_fold import (
    CONDITION_EMBED_DIM,
    ConditionedTeacher,
    fold_gigaflow_checkpoint,
    fold_gru_input,
    gigaflow_normalize,
    welford_to_obs_norm,
)

GIGAFLOW_CKPT = Path("/home/ubuntu/gigaflow-20h-ckpts/actor_005700.pt")


def test_fold_gru_input_matches_concatenated_condition():
    torch.manual_seed(0)
    weight = torch.randn(12, 7)
    bias = torch.randn(12)
    embed = torch.randn(2)
    trunk = torch.randn(5, 5)
    folded_w, folded_b = fold_gru_input(weight, bias, embed)
    assert tuple(folded_w.shape) == (12, 5)
    full = torch.cat([trunk, embed.expand(trunk.shape[0], -1)], dim=-1)
    expected = full @ weight.T + bias
    got = trunk @ folded_w.T + folded_b
    assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.skipif(not GIGAFLOW_CKPT.is_file(), reason="actor_005700.pt missing")
def test_folded_sophy_actor_matches_conditioned_teacher():
    payload = torch.load(GIGAFLOW_CKPT, map_location="cpu", weights_only=False)
    teacher = ConditionedTeacher(payload).eval()
    folded, artifact = fold_gigaflow_checkpoint(GIGAFLOW_CKPT)
    folded.eval()
    arch = artifact["actor_architecture"]
    fold_meta = artifact["gigaflow_fold"]
    assert arch["name"] == ACTOR_ARCHITECTURE_NAME
    assert arch["obs_dim"] == ACTOR_OBS_DIM
    assert arch["pool_mode"] == "max"
    assert arch["pool_bins"] == 64
    assert arch["projection_dim"] == 512
    assert fold_meta["source_gru_in"] == 592
    assert fold_meta["folded_gru_in"] == 528
    assert int(folded.gru.weight_ih_l0.shape[1]) == 528
    assert fold_meta["condition_embed_dim"] == CONDITION_EMBED_DIM
    validate_sensor_policy_artifact(
        artifact,
        expected_architecture=arch,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
    )

    obs_norm = artifact["obs_norm"]
    sophy_norm = load_obs_normalizer(
        obs_norm["mean"],
        obs_norm["var"],
        device=torch.device("cpu"),
        eps=float(obs_norm["eps"]),
        clip=float(obs_norm["clip"]),
        count=float(obs_norm["count"]),
    )
    assert sophy_norm.clip == pytest.approx(1.0)

    torch.manual_seed(1)
    obs = torch.randn(6, ACTOR_OBS_DIM)
    hidden = teacher.initial_hidden(6)
    sn = payload["sensor_normalizer"]
    giga = gigaflow_normalize(
        obs,
        torch.as_tensor(sn["mean"]).float(),
        torch.as_tensor(sn["m2"]).float(),
        float(sn["count"]),
        clip_sigma=float(sn.get("clip_sigma", 5.0)),
        eps=float(sn.get("eps", 1e-6)),
    )
    converted = welford_to_obs_norm(
        torch.as_tensor(sn["mean"]).float(),
        torch.as_tensor(sn["m2"]).float(),
        float(sn["count"]),
        clip_sigma=float(sn.get("clip_sigma", 5.0)),
        eps=float(sn.get("eps", 1e-6)),
    )
    sophy_z = load_obs_normalizer(
        converted["mean"],
        converted["var"],
        device=torch.device("cpu"),
        eps=float(converted["eps"]),
        clip=float(converted["clip"]),
        count=float(converted["count"]),
    ).normalize(obs)
    assert (giga - sophy_z).abs().max().item() < 1e-6

    with torch.no_grad():
        for _ in range(4):
            h0 = torch.randn_like(hidden) * 0.1
            teacher_act, teacher_h = teacher(obs, h0.clone())
            folded_act, _, folded_h = folded.step(
                sophy_norm.normalize(obs), h0.clone(), deterministic=True
            )
            act_err = (folded_act - teacher_act).abs().max().item()
            hid_err = (folded_h - teacher_h).abs().max().item()
            assert act_err < 1e-6, act_err
            assert hid_err < 1e-6, hid_err
            obs = obs + 0.05 * torch.randn_like(obs)

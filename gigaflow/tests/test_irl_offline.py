"""IRL bake + off-policy teleop fine-tune (never PPO-resume)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.artifacts import (
    bake_actor_artifact,
    build_actor_artifact_payload,
    export_baked_actor,
    validate_actor_artifact,
)
from gigaflow_f1tenth.cli import main
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.irl_identity import (
    IRL_DYNAMICS,
    IRL_MASS_KG,
    IRL_TIRE_MU,
    attach_baked_condition,
    baked_normalized_condition,
    baked_raw_style,
    fold_gru_layer0,
    folded_gru_module,
)
from gigaflow_f1tenth.model import build_actor
from gigaflow_f1tenth.offline_finetune import refuse_ppo, run_bakeoff
from gigaflow_f1tenth.sim.vehicle import F_BRAKE_MAX, F_DRIVE_MAX, VehicleParams
from gigaflow_f1tenth.teleop_dataset import (
    pack_step,
    split_holdout,
    synthetic_trajectories,
    teleop_to_action,
    windows_from_trajectories,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_irl_identity_matches_live_plant():
    params = VehicleParams()
    assert IRL_MASS_KG == pytest.approx(params.mass)
    assert IRL_TIRE_MU == pytest.approx(params.tire_mu)
    assert IRL_DYNAMICS["vmax_scale"] == 1.0
    assert IRL_DYNAMICS["mass_kg"] == pytest.approx(3.444)
    assert F_DRIVE_MAX == pytest.approx(21.46877492171339)
    assert F_BRAKE_MAX == pytest.approx(9.139188067139935)
    style = baked_raw_style()
    assert style.drive_scale == 1.0
    assert style.vmax_scale == 1.0
    assert style.mass_kg == pytest.approx(3.444)
    vec = baked_normalized_condition()
    assert len(vec) == 10
    assert vec[8] == pytest.approx(0.0)
    assert vec[9] == pytest.approx(-0.112, abs=1e-3)


def test_bake_actor_artifact_hardcodes_identity_and_strips_adaptation():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    payload = build_actor_artifact_payload(cfg, actor)
    payload["adaptation"] = {"vmax_scale": 1.7}
    payload["condition_schema"]["conservative_raw"]["vmax_scale"] = 1.7
    baked = bake_actor_artifact(payload, actor=actor)
    validate_actor_artifact(baked)
    assert baked["irl_baked"] is True
    assert "adaptation" not in baked
    raw = baked["condition_schema"]["conservative_raw"]
    assert raw["vmax_scale"] == 1.0
    assert raw["drive_scale"] == 1.0
    assert raw["mass_kg"] == pytest.approx(3.444)
    assert baked["deployment_condition_normalized"][8] == pytest.approx(0.0)


def test_export_baked_actor_roundtrip(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    path = tmp_path / "baked.pt"
    baked = export_baked_actor(cfg, actor, str(path))
    assert path.is_file()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["irl_baked"] is True
    assert loaded["irl_identity"]["vmax_scale"] == 1.0
    validate_actor_artifact(baked)


def test_teleop_pack_and_windows():
    from gigaflow_f1tenth.teleop_dataset import CONTROL_HZ, LONG_HISTORY, STEER_HISTORY

    lidar = np.full(1081, 4.0, dtype=np.float32)
    hist = np.zeros(STEER_HISTORY, dtype=np.float32)
    hist[0] = 0.10
    hist[2] = 0.06
    hist[4] = 0.02
    hist[6] = 0.00
    long_hist = np.zeros(LONG_HISTORY, dtype=np.float32)
    long_hist[0] = 0.3
    long_hist[2] = 0.2
    obs = pack_step(lidar, 1.2, 0.0, 0.1, 1.0, 0.0, long_hist, hist)
    assert CONTROL_HZ == 20.0
    assert obs.shape == (1097,)
    from gigaflow_f1tenth.sim.layout_local import STEER_DELTA0, STEER_T, THROTTLE_PRED

    assert obs[STEER_T] == pytest.approx(0.10)
    assert obs[STEER_DELTA0] == pytest.approx(0.04)
    assert obs[THROTTLE_PRED] == pytest.approx(0.2)
    action, steer = teleop_to_action(0.5, 0.1, 0.0)
    assert action.shape == (2,)
    assert action[0] == pytest.approx(0.5)
    assert steer == pytest.approx(0.1)
    trajs = synthetic_trajectories(n_traj=3, length=20, seed=0)
    train, hold = split_holdout(trajs, seed=0)
    assert train and hold
    windows = windows_from_trajectories(train, window=8, stride=4)
    assert windows["obs"].ndim == 3
    assert windows["actions"].shape[-1] == 2


def test_baked_condition_forward_without_external_c():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    attach_baked_condition(actor)
    actor.eval()
    obs = torch.zeros(2, 1097)
    obs[:, :1081] = 4.0
    hidden = actor.initial_hidden(2, "cpu")
    out = actor.forward(obs, None, hidden, deterministic=True)
    assert out.actions.shape == (2, 2)


def test_gru_layer0_fold_parity():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    attach_baked_condition(actor)
    actor.eval()
    e_c, w_folded, b_folded = fold_gru_layer0(actor)
    split = actor.gru_condition_slice()
    w_ih = actor.gru.weight_ih_l0
    b_ih = actor.gru.bias_ih_l0
    assert w_folded.shape[1] == split
    torch.testing.assert_close(w_folded, w_ih[:, :split], rtol=0.0, atol=0.0)
    torch.testing.assert_close(b_folded, b_ih + w_ih[:, split:] @ e_c, rtol=0.0, atol=0.0)
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.normal(0.0, 0.05, size=(4, 1097)), dtype=torch.float32)
    obs[:, :1081] = torch.clamp(3.0 + obs[:, :1081], 0.06, 30.0)
    hidden = actor.initial_hidden(4, "cpu")
    cond = actor.resolve_condition(None, 4, "cpu")
    with torch.no_grad():
        trunk = actor.encode_trunk(obs, cond)
        out_c, h_c = actor.gru(trunk.unsqueeze(1), hidden.unsqueeze(0).contiguous())
        gru_f, _ = folded_gru_module(actor)
        lidar, proprio = actor._split_obs(obs)
        trunk_f = torch.cat([actor.encoder(lidar), proprio], dim=-1)
        out_f, h_f = gru_f(trunk_f.unsqueeze(1), hidden.unsqueeze(0).contiguous())
    torch.testing.assert_close(out_c, out_f, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(h_c, h_f, rtol=0.0, atol=1e-6)


def test_refuse_ppo_resume():
    with pytest.raises(ValueError, match="never PPO-resume"):
        refuse_ppo("ppo")
    with pytest.raises(ValueError, match="never PPO-resume"):
        refuse_ppo("ppo-resume")
    refuse_ppo("iql")
    refuse_ppo("bc")


def test_bakeoff_synthetic_keep_or_revert(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    ckpt = tmp_path / "actor.pt"
    export_baked_actor(cfg, actor, str(ckpt))
    result = run_bakeoff(
        checkpoint=ckpt,
        output_dir=tmp_path / "out",
        trajectories=synthetic_trajectories(n_traj=3, length=24, seed=1),
        steps=2,
        batch_size=4,
        seed=1,
        device="cpu",
        window=8,
    )
    assert result["scalar"] == "composite_keep_score"
    assert "baseline" in result["arms"]
    assert "kl_clipped_bc" in result["arms"]
    assert result["winner"] in result["arms"]
    assert result["decision"] in {"keep", "revert"}
    assert result["keep"] is (result["winner"] != "baseline")
    for name, row in result["arms"].items():
        assert np.isfinite(row["score"]), name
        assert Path(row["artifact"]).is_file()
    assert Path(result["winner_artifact"]).is_file()
    payload = json.loads((tmp_path / "out" / "bakeoff.json").read_text())
    assert payload["winner"] == result["winner"]


def test_cli_bake_irl_and_finetune_offline(tmp_path, capsys):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    ckpt = tmp_path / "src.pt"
    export_baked_actor(cfg, actor, str(ckpt))
    baked_path = tmp_path / "irl.pt"
    rc = main(["bake-irl", "--checkpoint", str(ckpt), "--output", str(baked_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["irl_baked"] is True
    assert baked_path.is_file()

    out_dir = tmp_path / "ft"
    rc = main(
        [
            "finetune-offline",
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(out_dir),
            "--config",
            str(SMOKE),
            "--synthetic",
            "--method",
            "kl_clipped_bc",
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--window",
            "8",
            "--no-wandb",
        ]
    )
    assert rc == 0
    ft = json.loads(capsys.readouterr().out)
    assert ft["scalar"] == "composite_keep_score"
    assert ft["winner"] in ft["arms"]


def test_cli_finetune_rejects_ppo(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    ckpt = tmp_path / "src.pt"
    export_baked_actor(cfg, actor, str(ckpt))
    with pytest.raises(ValueError, match="never PPO-resume"):
        main(
            [
                "finetune-offline",
                "--checkpoint",
                str(ckpt),
                "--output-dir",
                str(tmp_path / "out"),
                "--synthetic",
                "--method",
                "ppo",
                "--no-wandb",
            ]
        )

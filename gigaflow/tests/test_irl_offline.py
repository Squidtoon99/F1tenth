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
    baked_normalized_condition,
    baked_raw_style,
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
    lidar = np.full(1081, 4.0, dtype=np.float32)
    hist = np.zeros(7, dtype=np.float32)
    obs = pack_step(lidar, 1.2, 0.0, 0.1, 1.0, 0.0, 0.3, 0.2, hist)
    assert obs.shape == (1097,)
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
        arms=(
            {"name": "bc-kl", "method": "bc", "kl_coef": 0.1, "kl_max": 0.05},
            {"name": "iql-kl", "method": "iql", "kl_coef": 0.1, "kl_max": 0.05},
        ),
        include_variations=False,
    )
    assert result["scalar"] == "holdout_action_mse"
    assert set(result["arms"]) == {"baseline", "bc-kl", "iql-kl"}
    assert result["winner"] in result["arms"]
    assert result["decision"] in {"keep", "revert"}
    assert result["keep"] is (result["winner"] != "baseline")
    assert "kl" in result["arms"]["bc-kl"]["train"]
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
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--window",
            "8",
            "--arm",
            "bc-kl",
            "--no-variations",
            "--no-wandb",
        ]
    )
    assert rc == 0
    ft = json.loads(capsys.readouterr().out)
    assert ft["scalar"] == "holdout_action_mse"
    assert ft["winner"] in {"baseline", "bc-kl"}


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

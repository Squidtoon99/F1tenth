from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from config import (
    DEFAULT_CONFIG,
    control_dt_s,
    effective_ppo_horizon,
    effective_qrsac_horizon,
)
from standalone_trainer import build_config, load_config_patch, parse_args

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONTRACT_CONTROL_DT_S = 0.1  # critic contract CONTROL_HZ=10, not sim cadence


def test_control_dt_is_sim_20hz_not_contract_10hz():
    dt = control_dt_s(DEFAULT_CONFIG["env"])
    assert dt == pytest.approx(0.05)
    assert dt != pytest.approx(_CONTRACT_CONTROL_DT_S)


def test_time_horizon_is_identity_at_10hz():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["control_interval"] = 20
    assert control_dt_s(cfg["env"]) == pytest.approx(_CONTRACT_CONTROL_DT_S)
    gamma, gae_lambda = effective_ppo_horizon(cfg)
    assert gamma == pytest.approx(0.9896)
    assert gae_lambda == pytest.approx(0.95)
    q_gamma, n_step = effective_qrsac_horizon(cfg)
    assert q_gamma == pytest.approx(0.9896)
    assert n_step == 7


def test_default_20hz_rescales_gamma_lambda_and_n_step():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    assert control_dt_s(cfg["env"]) == pytest.approx(0.05)
    gamma, gae_lambda = effective_ppo_horizon(cfg)
    assert gamma == pytest.approx(0.9896**0.5, rel=1e-4)
    assert gamma == pytest.approx(0.9948, abs=5e-5)
    assert gae_lambda == pytest.approx(0.95**0.5, rel=1e-4)
    assert gae_lambda == pytest.approx(0.9747, abs=5e-5)
    q_gamma, n_step = effective_qrsac_horizon(cfg)
    assert q_gamma == pytest.approx(0.9896**0.5)
    assert n_step == 14
    # Identity rescale at the critic contract dt would leave 10 Hz values in place.
    assert gamma != pytest.approx(0.9896)
    assert gae_lambda != pytest.approx(0.95)


def test_per_step_override_pins_10hz_values_at_20hz():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["ppo"]["horizon_from_control_dt"] = False
    cfg["model"]["horizon_from_control_dt"] = False
    gamma, gae_lambda = effective_ppo_horizon(cfg)
    assert gamma == pytest.approx(0.9896)
    assert gae_lambda == pytest.approx(0.95)
    q_gamma, n_step = effective_qrsac_horizon(cfg)
    assert q_gamma == pytest.approx(0.9896)
    assert n_step == 7


def test_courtyard_ab_configs_differ_only_on_horizon_flag():
    time_path = _CONFIG_DIR / "courtyard_2_e2e_ppo.json"
    step_path = _CONFIG_DIR / "courtyard_2_e2e_ppo_per_step_horizon.json"
    time_patch = json.loads(time_path.read_text(encoding="utf-8"))
    step_patch = json.loads(step_path.read_text(encoding="utf-8"))
    assert time_patch["algorithm"] == step_patch["algorithm"] == "ppo"
    assert time_patch["ppo"]["horizon_from_control_dt"] is True
    assert step_patch["ppo"]["horizon_from_control_dt"] is False
    time_patch["ppo"].pop("horizon_from_control_dt")
    step_patch["ppo"].pop("horizon_from_control_dt")
    assert time_patch == step_patch


def test_courtyard_ab_configs_resolve_through_build_config():
    time_patch, _ = load_config_patch(str(_CONFIG_DIR / "courtyard_2_e2e_ppo.json"))
    step_patch, _ = load_config_patch(
        str(_CONFIG_DIR / "courtyard_2_e2e_ppo_per_step_horizon.json")
    )
    args, explicit = parse_args([])
    time_cfg = build_config(args, patch=time_patch, explicit=explicit)
    step_cfg = build_config(args, patch=step_patch, explicit=explicit)
    assert control_dt_s(time_cfg["env"]) == pytest.approx(0.05)
    time_gamma, time_lam = effective_ppo_horizon(time_cfg)
    step_gamma, step_lam = effective_ppo_horizon(step_cfg)
    assert time_gamma == pytest.approx(0.9896**0.5)
    assert time_lam == pytest.approx(0.95**0.5)
    assert step_gamma == pytest.approx(0.9896)
    assert step_lam == pytest.approx(0.95)

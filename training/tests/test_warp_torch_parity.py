"""Warp-vs-Torch parity against a committed Torch reference trajectory.

The reference ``.npz`` is generated from the Torch ``TorchVehicleSim`` that still
lives on ``develop`` (see ``analysis/warp_parity/generate_torch_reference.py``).
This test replays the identical schedule, init conditions, and per-env domain
parameters through ``WarpVehicleSim`` and asserts the trajectories match within
documented tolerances.

Tolerances encode the accepted Warp-vs-Torch divergence. They are intentionally
tight on pose/velocity; slip/load are looser because the tyre readback is more
sensitive to per-substep force ordering. A regression that exceeds these bounds is
a Warp physics regression to fix, not a tolerance to widen (see ADR 0007).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from f1tenth_sim.params import VehicleParams
from f1tenth_sim.sim_warp import WarpVehicleSim

_FIXTURE = (
    Path(__file__).resolve().parent / "data" / "torch_reference_trajectory.npz"
)

# Per-channel max-abs tolerances (Warp vs recorded Torch reference), sized ~2x the
# observed divergence over a 200-step aggressive schedule. Pose drift is the loosest
# because tiny per-step float-ordering differences accumulate over the trajectory;
# instantaneous states (vel/yaw/slip/load) stay tight.
TOL = {
    "base_pos": 0.30,
    "yaw": 0.05,
    "base_lin_vel": 0.15,
    "base_ang_vel": 0.10,
    "base_lin_acc": 0.50,
    "tyre_slip": 0.05,
    "tyre_load": 0.03,
    "dof_vel": 3.0,
}


def _load_reference():
    if not _FIXTURE.exists():
        pytest.skip(
            "Torch reference fixture missing; regenerate with "
            "analysis/warp_parity/generate_torch_reference.py"
        )
    return np.load(_FIXTURE, allow_pickle=False)


def _meta(ref) -> dict:
    return json.loads(bytes(ref["meta"]).decode("utf-8"))


def _run_warp(ref, meta):
    cfg = dict(meta["config"])
    # Torch reference trajectory was recorded under absolute steering.
    cfg.setdefault("steering_action_mode", "absolute")
    params = VehicleParams.from_config(cfg)
    num_envs = int(meta["num_envs"])
    sim = WarpVehicleSim(
        params,
        num_envs,
        device="cpu",
        sim_dt=float(meta["sim_dt"]),
        control_dt=float(meta["control_dt"]),
    )
    sim.reset(
        None,
        torch.from_numpy(ref["init_pos"]),
        torch.from_numpy(ref["init_quat"]),
        torch.from_numpy(ref["init_speed"]),
    )
    sim.set_domain(
        torch.from_numpy(ref["domain_mass"]),
        torch.from_numpy(ref["domain_mu"]),
        torch.from_numpy(ref["domain_drive_scale"]),
        torch.from_numpy(ref["domain_steer_bias"]),
    )

    actions = ref["actions"]
    steps = actions.shape[0]
    n_steps = int(meta["n_steps"])
    rec = {name: [] for name in TOL}
    for t in range(steps):
        sim.step(torch.from_numpy(actions[t]), n_steps=n_steps)
        state = sim.read_state()
        wheels = sim.read_wheel_state()
        quat_t = state["base_quat"]
        yaw = 2.0 * torch.atan2(quat_t[:, 3], quat_t[:, 0])
        # read_state/read_wheel_state return live state references for some
        # channels; clone before recording so each step is a distinct snapshot.
        rec["base_pos"].append(state["base_pos"].clone().cpu().numpy())
        rec["yaw"].append(yaw.clone().cpu().numpy())
        rec["base_lin_vel"].append(state["base_lin_vel"].clone().cpu().numpy())
        rec["base_ang_vel"].append(state["base_ang_vel"].clone().cpu().numpy())
        rec["base_lin_acc"].append(state["base_lin_acc"].clone().cpu().numpy())
        rec["tyre_slip"].append(wheels["tyre_slip"].clone().cpu().numpy())
        rec["tyre_load"].append(wheels["tyre_load"].clone().cpu().numpy())
        rec["dof_vel"].append(wheels["dof_vel"].clone().cpu().numpy())
    return {name: np.asarray(values, dtype=np.float32) for name, values in rec.items()}


def test_warp_matches_torch_reference():
    ref = _load_reference()
    meta = _meta(ref)
    warp = _run_warp(ref, meta)

    failures = []
    for channel, tol in TOL.items():
        error = float(np.max(np.abs(warp[channel] - ref[channel])))
        if not np.isfinite(error) or error > tol:
            failures.append(f"{channel}: max_abs_err={error:.4f} > tol={tol}")
    assert not failures, "Warp diverged from Torch reference:\n" + "\n".join(failures)


def test_warp_reference_yaw_endpoint():
    """Anchor final heading to the Torch reference (not a self-referential golden)."""
    ref = _load_reference()
    meta = _meta(ref)
    warp = _run_warp(ref, meta)
    assert np.max(np.abs(warp["yaw"][-1] - ref["yaw"][-1])) < TOL["yaw"]

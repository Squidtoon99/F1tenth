"""Generate the Torch-simulator reference trajectory fixture for Warp parity.

The Warp migration deleted the in-tree Torch vehicle simulator on purpose. To keep
an *independent* parity check, we run the Torch ``TorchVehicleSim`` that still lives
on ``develop`` in a separate checkout, record its trajectories, and commit the
recorded ``.npz`` as the reference. The in-tree test
``training/tests/test_warp_torch_parity.py`` then replays the identical schedule
through ``WarpVehicleSim`` and asserts it matches the recorded Torch reference.

No Torch-simulator code is added to this repository. This script only *imports* it
from ``$F1TENTH_TORCH_ROOT`` (default ``~/projects/F1tenth``) at generation time.

Usage::

    PYTHONPATH=training:libs/f1tenth_contract \\
        .venv/bin/python analysis/warp_parity/generate_torch_reference.py

Regenerate only when intentionally rebaselining against the Torch reference; the
committed fixture is what CI compares against.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Shared physics config used to build VehicleParams on both sides. Kept minimal so
# both the Torch and Warp VehicleParams.from_config produce identical scalars.
REFERENCE_CONFIG = {"tire_friction": 0.9}

# Control cadence matches the Warp production env: 10 sub-steps of 5 ms = 50 ms.
SIM_DT = 0.005
CONTROL_DT = 0.05
N_STEPS = 10

# Per-env domain parameters, chosen to exercise mass / friction / drivetrain /
# steer-bias randomization channels rather than a single nominal car.
DOMAIN = {
    "mass": [3.74, 3.60, 3.90, 3.74],
    "mu": [0.90, 0.65, 0.70, 0.90],
    "drive_scale": [1.0, 0.8, 1.3, 1.0],
    "steer_bias": [0.0, 0.0, 0.0, 0.05],
}
NUM_ENVS = len(DOMAIN["mass"])

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "training"
    / "tests"
    / "data"
    / "torch_reference_trajectory.npz"
)


def build_schedule() -> np.ndarray:
    """Deterministic (throttle, steer) schedule broadcast across all envs.

    Segments exercise straight-line accel, coast, left/right steer sweeps, hard
    braking, and a seeded aggressive-random tail.
    """

    segments: list[np.ndarray] = []

    def hold(throttle: float, steer: float, count: int) -> None:
        segments.append(
            np.tile(np.array([throttle, steer], dtype=np.float32), (count, 1))
        )

    hold(0.5, 0.0, 40)
    hold(0.0, 0.0, 20)
    hold(0.3, 0.5, 40)
    hold(0.3, -0.5, 40)
    hold(-1.0, 0.0, 20)

    rng = np.random.default_rng(0)
    aggressive = rng.uniform(-1.0, 1.0, size=(40, 2)).astype(np.float32)
    segments.append(aggressive)

    schedule = np.concatenate(segments, axis=0)
    return np.repeat(schedule[:, None, :], NUM_ENVS, axis=1)


def _init_conditions():
    pos = torch.zeros(NUM_ENVS, 3)
    quat = torch.zeros(NUM_ENVS, 4)
    quat[:, 0] = 1.0
    speed = torch.zeros(NUM_ENVS)
    return pos, quat, speed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-root",
        default=os.environ.get(
            "F1TENTH_TORCH_ROOT", str(Path.home() / "projects" / "F1tenth")
        ),
        help="Path to the checkout that still contains TorchVehicleSim (develop).",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    torch_training = Path(args.torch_root) / "training"
    if not (torch_training / "f1tenth_sim" / "sim.py").exists():
        raise SystemExit(
            f"TorchVehicleSim not found under {torch_training}. Set --torch-root or "
            "$F1TENTH_TORCH_ROOT to the develop checkout."
        )
    sys.path.insert(0, str(torch_training))

    from f1tenth_sim import TorchVehicleSim, VehicleParams

    params = VehicleParams.from_config(dict(REFERENCE_CONFIG))
    sim = TorchVehicleSim(
        params,
        NUM_ENVS,
        sim_dt=SIM_DT,
        control_dt=CONTROL_DT,
        internal_substeps=1,
    )

    pos, quat, speed = _init_conditions()
    sim.reset(None, pos, quat, speed)
    sim.set_domain(
        None,
        mass=torch.tensor(DOMAIN["mass"]),
        mu=torch.tensor(DOMAIN["mu"]),
        drive_scale=torch.tensor(DOMAIN["drive_scale"]),
        steer_bias=torch.tensor(DOMAIN["steer_bias"]),
    )

    schedule = build_schedule()
    steps = schedule.shape[0]
    rec = {
        "base_pos": [],
        "yaw": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "base_lin_acc": [],
        "tyre_slip": [],
        "tyre_load": [],
        "dof_vel": [],
    }
    for t in range(steps):
        action = torch.from_numpy(schedule[t])
        sim.step(action, n_steps=N_STEPS)
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

    payload = {name: np.asarray(values, dtype=np.float32) for name, values in rec.items()}
    payload["actions"] = schedule.astype(np.float32)
    payload["init_pos"] = pos.cpu().numpy().astype(np.float32)
    payload["init_quat"] = quat.cpu().numpy().astype(np.float32)
    payload["init_speed"] = speed.cpu().numpy().astype(np.float32)
    for key, value in DOMAIN.items():
        payload[f"domain_{key}"] = np.asarray(value, dtype=np.float32)
    payload["meta"] = np.frombuffer(
        json.dumps(
            {
                "sim_dt": SIM_DT,
                "control_dt": CONTROL_DT,
                "n_steps": N_STEPS,
                "num_envs": NUM_ENVS,
                "config": REFERENCE_CONFIG,
                "torch_git": _git_describe(args.torch_root),
            }
        ).encode("utf-8"),
        dtype=np.uint8,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"wrote {output} ({steps} steps x {NUM_ENVS} envs)")


def _git_describe(root: str) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()

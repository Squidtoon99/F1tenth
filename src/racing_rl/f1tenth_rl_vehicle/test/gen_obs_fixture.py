#!/usr/bin/env python3
"""Generate the C++ obs-parity fixture from the Python observation pipeline.

Emits a plain whitespace-separated fixture (test/obs_fixture.txt) consumed by
test_rl_obs_core.cpp: the embedded track, the obs config, and a handful of
(state -> expected observation) cases produced by the *deployed* Python
ObservationBuilder (which the parity test pins to f1tenth_env training math).
The tyre-load block [380:384] uses the builder default (static ratio 1.0), which
the C++ VehicleState mirrors, so the fixture state line is unchanged.

Run from the f1tenth_rl_agent package dir (so its module is importable):

    PYTHONPATH=../f1tenth_rl_agent python gen_obs_fixture.py <centerline_csv> <out.txt>
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from f1tenth_rl_agent.interfaces import default_obs_cfg
from f1tenth_rl_agent.obs_core import ObservationBuilder
from f1tenth_rl_agent.track_io import load_track_csv


def yaw_to_quat_wxyz(yaw: float) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.tensor([[np.cos(half), 0.0, 0.0, np.sin(half)]], dtype=torch.float32)


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def main() -> None:
    csv_path = sys.argv[1]
    out_path = sys.argv[2]

    centerline, wl, wr = load_track_csv(csv_path)
    obs_cfg = default_obs_cfg()
    builder = ObservationBuilder(centerline, wl, wr, obs_cfg, device=torch.device("cpu"))

    rng = np.random.default_rng(7)
    n = centerline.shape[0]
    cases = []
    for _ in range(12):
        idx = int(rng.integers(0, n))
        base = centerline[idx]
        offset = rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
        pos = np.array([base[0] + offset[0], base[1] + offset[1], 0.0], dtype=np.float32)
        yaw = float(rng.uniform(-np.pi, np.pi))
        vx = float(rng.uniform(-0.5, 6.0))
        vy = float(rng.uniform(-1.0, 1.0))
        wz = float(rng.uniform(-1.5, 1.5))
        ax = float(rng.uniform(-3.0, 3.0))
        ay = float(rng.uniform(-2.0, 2.0))
        last_t = float(rng.uniform(-1.0, 1.0))
        last_s = float(rng.uniform(-1.0, 1.0))
        slip = rng.uniform(-0.5, 0.5, size=8).astype(np.float32)

        base_lin_vel = torch.tensor([[vx, vy, 0.0]], dtype=torch.float32)
        base_ang_vel = torch.tensor([[0.0, 0.0, wz]], dtype=torch.float32)
        base_lin_acc = torch.tensor([[ax, ay, 0.0]], dtype=torch.float32)
        last_actions = torch.tensor([[last_t, last_s]], dtype=torch.float32)
        base_pos = torch.tensor([pos], dtype=torch.float32)
        base_quat = yaw_to_quat_wxyz(yaw)
        tyre = torch.tensor([slip], dtype=torch.float32)

        obs = builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            base_lin_acc=base_lin_acc,
            last_actions=last_actions,
            base_pos=base_pos,
            base_quat_wxyz=base_quat,
            tyre_slip=tyre,
        )
        obs_np = obs.squeeze(0).numpy().astype(np.float64)
        cases.append(
            (pos[0], pos[1], yaw, vx, vy, wz, ax, ay, last_t, last_s, slip, obs_np)
        )

    # --- 1v1 opponent-block cases (392-dim) ----------------------------------
    # Built with the deployed build_opponent_block (pinned to training obs_opponent)
    # so the C++ opponent block at [384:392] is verified for parity.
    opp_cfg = default_obs_cfg(enable_opponent_obs=True)
    opp_builder = ObservationBuilder(
        centerline, wl, wr, opp_cfg, device=torch.device("cpu")
    )
    opp_cases = []
    for k in range(10):
        ei = int(rng.integers(0, n))
        eo = rng.uniform(-0.5, 0.5, size=2).astype(np.float64)
        epos = np.array([centerline[ei, 0] + eo[0], centerline[ei, 1] + eo[1], 0.0])
        eyaw = float(rng.uniform(-np.pi, np.pi))
        evx = float(rng.uniform(-0.5, 6.0))
        evy = float(rng.uniform(-1.0, 1.0))
        wz = float(rng.uniform(-1.5, 1.5))
        ax = float(rng.uniform(-3.0, 3.0))
        ay = float(rng.uniform(-2.0, 2.0))
        lt = float(rng.uniform(-1.0, 1.0))
        ls = float(rng.uniform(-1.0, 1.0))
        slip = rng.uniform(-0.5, 0.5, size=8).astype(np.float64)

        oi = int(rng.integers(0, n))
        oo = rng.uniform(-0.5, 0.5, size=2).astype(np.float64)
        opos = np.array([centerline[oi, 0] + oo[0], centerline[oi, 1] + oo[1], 0.0])
        oyaw = float(rng.uniform(-np.pi, np.pi))
        # Some cases stationary (parity for the static-opponent path), some moving.
        if k % 3 == 0:
            ovx, ovy = 0.0, 0.0
            oax, oay = 0.0, 0.0
        else:
            ovx = float(rng.uniform(-4.0, 4.0))
            ovy = float(rng.uniform(-2.0, 2.0))
            oax = float(rng.uniform(-3.0, 3.0))
            oay = float(rng.uniform(-2.0, 2.0))

        ego_vel_world = _rot(eyaw) @ np.array([evx, evy], dtype=np.float64)
        block = opp_builder.build_opponent_block(
            ego_pos=torch.tensor([epos], dtype=torch.float32),
            ego_yaw=torch.tensor([eyaw], dtype=torch.float32),
            ego_vel_world=torch.tensor([ego_vel_world], dtype=torch.float32),
            opp_pos=torch.tensor([opos], dtype=torch.float32),
            opp_vel_world=torch.tensor([[ovx, ovy]], dtype=torch.float32),
            ego_acc_body=torch.tensor([[ax, ay]], dtype=torch.float32),
            opp_yaw=torch.tensor([oyaw], dtype=torch.float32),
            opp_acc_body=torch.tensor([[oax, oay]], dtype=torch.float32),
        )
        obs = opp_builder.build(
            base_lin_vel=torch.tensor([[evx, evy, 0.0]], dtype=torch.float32),
            base_ang_vel=torch.tensor([[0.0, 0.0, wz]], dtype=torch.float32),
            base_lin_acc=torch.tensor([[ax, ay, 0.0]], dtype=torch.float32),
            last_actions=torch.tensor([[lt, ls]], dtype=torch.float32),
            base_pos=torch.tensor([epos], dtype=torch.float32),
            base_quat_wxyz=yaw_to_quat_wxyz(eyaw),
            tyre_slip=torch.tensor([slip], dtype=torch.float32),
            opponent_block=block,
        )
        obs_np = obs.squeeze(0).numpy().astype(np.float64)
        opp_cases.append(
            (
                epos[0], epos[1], eyaw, evx, evy, wz, ax, ay, lt, ls, slip,
                opos[0], opos[1], oyaw, ovx, ovy, oax, oay, obs_np,
            )
        )

    with open(out_path, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(
                f"{centerline[i, 0]:.9g} {centerline[i, 1]:.9g} "
                f"{wl[i]:.9g} {wr[i]:.9g}\n"
            )
        f.write(
            "{} {} {} {} {} {} {} {}\n".format(
                obs_cfg["future_track_num_points"],
                obs_cfg["future_track_horizon_s"],
                obs_cfg["future_track_width"],
                obs_cfg["contact_margin_m"],
                obs_cfg["clip_obs"],
                obs_cfg["obs_scales"]["lin_vel"],
                obs_cfg["obs_scales"]["ang_vel"],
                obs_cfg["obs_scales"]["lin_acc"],
            )
        )
        f.write(f"{len(cases)}\n")
        for (px, py, yaw, vx, vy, wz, ax, ay, lt, ls, slip, obs_np) in cases:
            state = [px, py, yaw, vx, vy, wz, ax, ay, lt, ls] + list(slip)
            f.write(" ".join(f"{v:.9g}" for v in state) + "\n")
            f.write(" ".join(f"{v:.9g}" for v in obs_np) + "\n")

        # Opponent section (read conditionally by the C++ test).
        f.write(f"{len(opp_cases)}\n")
        for case in opp_cases:
            (px, py, yaw, vx, vy, wz, ax, ay, lt, ls, slip,
             opx, opy, oyaw, ovx, ovy, oax, oay, obs_np) = case
            state = [px, py, yaw, vx, vy, wz, ax, ay, lt, ls] + list(slip)
            f.write(" ".join(f"{v:.9g}" for v in state) + "\n")
            f.write(
                f"{opx:.9g} {opy:.9g} {oyaw:.9g} {ovx:.9g} {ovy:.9g} "
                f"{oax:.9g} {oay:.9g}\n"
            )
            f.write(" ".join(f"{v:.9g}" for v in obs_np) + "\n")

    print(
        f"wrote {out_path}: {n} track points, {len(cases)} solo cases, "
        f"{len(opp_cases)} opponent cases"
    )


if __name__ == "__main__":
    main()

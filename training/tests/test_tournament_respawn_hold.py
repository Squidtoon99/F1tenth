"""Real-env integration test for tournament respawn + freeze-hold behavior."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

TRAINING_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = TRAINING_DIR / "analysis"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from f1tenth_env import runtime as rt  # noqa: E402
from f1tenth_env.utils import compute_oob_from_boundary_state  # noqa: E402
from standalone_trainer import DEFAULT_CONFIG  # noqa: E402
from tournament import (  # noqa: E402
    OOB_CONSECUTIVE,
    _crashed_side_respawn,
    _warp_build_step_state,
    advance_hold_race_step,
    apply_shotgun_start,
    build_race_config,
    force_car_oob,
    freeze_steps_for,
    lateral_error_at,
    make_env,
    tangent_yaw_at,
)

POSE_TOL_M = 0.05
YAW_TOL_RAD = 0.15
FREEZE_S = 2.0


def _make_race_env():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = build_race_config(copy.deepcopy(DEFAULT_CONFIG), "Austin")
    return make_env(cfg, 1), cfg


def _init_tracker(env):
    n = env.num_envs
    device = env.device
    sim_ss = _warp_build_step_state(env, "ego")
    opp_ss = _warp_build_step_state(env, "opp")
    z = torch.zeros(n, dtype=rt.tc_float, device=device)
    return {
        "sim_hold": torch.zeros(n, dtype=torch.int32, device=device),
        "opp_hold": torch.zeros(n, dtype=torch.int32, device=device),
        "fx_s": z.clone(), "fy_s": z.clone(), "fyaw_s": z.clone(),
        "fx_o": z.clone(), "fy_o": z.clone(), "fyaw_o": z.clone(),
        "sim_streak": torch.zeros(n, dtype=torch.int32, device=device),
        "opp_streak": torch.zeros(n, dtype=torch.int32, device=device),
        "sim_prog": torch.zeros(n, dtype=rt.tc_float, device=device),
        "opp_prog": torch.zeros(n, dtype=rt.tc_float, device=device),
        "sim_crashes": torch.zeros(n, dtype=torch.int32, device=device),
        "opp_crashes": torch.zeros(n, dtype=torch.int32, device=device),
        "collide_sim": torch.zeros(n, dtype=torch.int32, device=device),
        "collide_opp": torch.zeros(n, dtype=torch.int32, device=device),
        "prev_sim_s": sim_ss["frenet"]["s"].detach().clone(),
        "prev_opp_s": opp_ss["frenet"]["s"].detach().clone(),
        "active": torch.ones(n, dtype=torch.bool, device=device),
    }


def _ego_xy_yaw(env):
    pos = env.base_pos[0, :2].detach().cpu()
    yaw = float(tangent_yaw_at(env, "ego")[0].item())
    return pos, yaw


def test_respawn_hold_positive_side_real_env():
    env, cfg = _make_race_env()
    control_interval = int(cfg["env"]["control_interval"])
    clip = float(cfg["env"]["clip_actions"])
    freeze_steps = freeze_steps_for(env, FREEZE_S)
    throttle = torch.tensor([[1.0, 1.0]], device=env.device)
    zero = torch.zeros_like(throttle)
    try:
        env.reset()
        apply_shotgun_start(env, torch.ones(1, dtype=torch.bool, device=env.device))

        side_sign = force_car_oob(env, "ego", side_positive=True)
        assert side_sign > 0
        sim_ss = _warp_build_step_state(env, "ego")
        oob, _ = compute_oob_from_boundary_state(
            sim_ss["boundary"],
            margin_m=float(env.term_params["term_oob_margin_m"]),
        )
        assert bool(oob[0])

        tracker = _init_tracker(env)
        crash_mask = None
        for _ in range(OOB_CONSECUTIVE):
            tracker = advance_hold_race_step(
                env,
                sim_actions=zero,
                opp_actions=zero,
                control_interval=control_interval,
                clip_actions=clip,
                freeze_steps=freeze_steps,
                **tracker,
            )
            if int(tracker["sim_crashes"][0]) > 0:
                crash_mask = True
                break
        assert crash_mask is not None
        assert int(tracker["sim_crashes"][0]) == 1

        lat_after = float(lateral_error_at(env, "ego")[0].item())
        assert lat_after > 0
        fresh_ss = _warp_build_step_state(env, "ego")
        assert lat_after < float(fresh_ss["boundary"]["w_l_s"][0].item())
        yaw_after = float(tangent_yaw_at(env, "ego")[0].item())
        ego_yaw = float(env._ego.tensor["yaw"][0].item())
        assert abs(ego_yaw - yaw_after) < YAW_TOL_RAD

        pin_xy, pin_yaw = _ego_xy_yaw(env)
        crashes_at_hold_start = int(tracker["sim_crashes"][0])
        collisions_at_hold_start = int(tracker["collide_sim"][0])
        pinned_steps = 0
        for _ in range(freeze_steps + 4):
            if int(tracker["sim_hold"][0]) <= 0:
                break
            tracker = advance_hold_race_step(
                env,
                sim_actions=throttle,
                opp_actions=zero,
                control_interval=control_interval,
                clip_actions=clip,
                freeze_steps=freeze_steps,
                **tracker,
            )
            xy, yaw = _ego_xy_yaw(env)
            assert torch.linalg.norm(xy - pin_xy).item() < POSE_TOL_M
            assert abs(yaw - pin_yaw) < YAW_TOL_RAD
            assert int(tracker["sim_crashes"][0]) == crashes_at_hold_start
            assert int(tracker["collide_sim"][0]) == collisions_at_hold_start
            pinned_steps += 1

        assert pinned_steps >= freeze_steps - 1
        assert int(tracker["sim_hold"][0]) == 0

        pre_move = env.base_pos[0, :2].detach().clone()
        max_delta = 0.0
        for _ in range(8):
            tracker = advance_hold_race_step(
                env,
                sim_actions=throttle,
                opp_actions=zero,
                control_interval=control_interval,
                clip_actions=clip,
                freeze_steps=freeze_steps,
                **tracker,
            )
            post_move = env.base_pos[0, :2].detach().clone()
            max_delta = max(
                max_delta, float(torch.linalg.norm(post_move - pre_move).item()),
            )
        assert max_delta > 0.01
    finally:
        env.close()


def test_respawn_hold_negative_side_real_env():
    env, cfg = _make_race_env()
    control_interval = int(cfg["env"]["control_interval"])
    clip = float(cfg["env"]["clip_actions"])
    freeze_steps = freeze_steps_for(env, FREEZE_S)
    zero = torch.zeros(1, 2, device=env.device)
    try:
        env.reset()
        apply_shotgun_start(env, torch.ones(1, dtype=torch.bool, device=env.device))

        side_sign = force_car_oob(env, "ego", side_positive=False)
        assert side_sign < 0

        mask = torch.tensor([True], device=env.device)
        fx, fy, fyaw = _crashed_side_respawn(env, "ego", mask)
        lat_after = float(lateral_error_at(env, "ego")[0].item())
        assert lat_after < 0
        assert abs(float(fyaw[0].item()) - float(tangent_yaw_at(env, "ego")[0])) < (
            YAW_TOL_RAD
        )

        tracker = _init_tracker(env)
        tracker["sim_hold"] = torch.tensor([freeze_steps], dtype=torch.int32,
                                           device=env.device)
        tracker["fx_s"] = fx
        tracker["fy_s"] = fy
        tracker["fyaw_s"] = fyaw
        pin_xy = env.base_pos[0, :2].detach().clone()
        for _ in range(freeze_steps):
            tracker = advance_hold_race_step(
                env,
                sim_actions=zero,
                opp_actions=zero,
                control_interval=control_interval,
                clip_actions=clip,
                freeze_steps=freeze_steps,
                **tracker,
            )
            assert torch.linalg.norm(
                env.base_pos[0, :2] - pin_xy
            ).item() < POSE_TOL_M
        assert int(tracker["sim_hold"][0]) == 0
    finally:
        env.close()

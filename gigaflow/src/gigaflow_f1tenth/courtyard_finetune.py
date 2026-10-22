"""Online courtyard_2 fine-tune of baked actor_005700.

The student actor sees only 1,097-D LiDAR + NIME proprio with a hardcoded
IRL condition. A privileged centerline teacher (Stanley + slow-in, using
Frenet geometry the actor never receives) labels on-policy states. Training
is DAgger BC + annealed KL to the frozen base + progress/clearance REINFORCE.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gigaflow_f1tenth.artifacts import export_baked_actor
from gigaflow_f1tenth.config import ExperimentConfig, load_config
from gigaflow_f1tenth.evaluation import load_actor_from_checkpoint
from gigaflow_f1tenth.irl_identity import attach_baked_condition, baked_normalized_condition
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.model import LOG_STD_MAX, LOG_STD_MIN
from gigaflow_f1tenth.offline_finetune import gaussian_kl
from gigaflow_f1tenth.rewards import deployment_style
from gigaflow_f1tenth.sim.geometry import project_frenet_numpy
from gigaflow_f1tenth.sim.vehicle import MAX_STEER_RAD, STEERING_DELTA_MAX_RAD
from gigaflow_f1tenth.tracks import load_atlas

COURTYARD_S = 10.877
COURTYARD_EY = 0.0
COURTYARD_YAW = -3.046
COURTYARD_LENGTH_M = 17.619
NORTH_SPAWN = {
    "s": COURTYARD_S,
    "ey": COURTYARD_EY,
    "yaw": COURTYARD_YAW,
    "speed": 0.0,
}


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_on_track(geom, track_id: int, s: float, ey: float) -> dict[str, float]:
    a = int(geom.offsets[track_id])
    b = int(geom.offsets[track_id + 1])
    length = float(geom.track_length[track_id])
    s = float(s) % length
    if s < 0.0:
        s += length
    center = geom.centerline_xy[a:b]
    tangents = geom.tangents_xy[a:b]
    normals = geom.normals_xy[a:b]
    widths = geom.widths_rl[a:b]
    cum = geom.cum_length[a:b]
    seglen = geom.segment_length[a:b]
    segment = int(np.searchsorted(cum, s, side="right") - 1)
    segment = min(max(segment, 0), b - a - 1)
    sl = max(float(seglen[segment]), 1e-6)
    alpha = min(max((s - float(cum[segment])) / sl, 0.0), 1.0)
    following = (segment + 1) % (b - a)
    mid = center[segment] + alpha * (center[following] - center[segment])
    xy = mid + float(ey) * normals[segment]
    yaw = float(np.arctan2(tangents[segment, 1], tangents[segment, 0]))
    fr = project_frenet_numpy(
        np.asarray(xy, dtype=np.float32),
        segment,
        center,
        tangents,
        normals,
        widths,
        cum,
        seglen,
    )
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "yaw": yaw,
        "tx": float(tangents[segment, 0]),
        "ty": float(tangents[segment, 1]),
        **fr,
    }


def write_slot_pose(sim, slot: int, pose: dict[str, float], speed: float) -> None:
    t = sim.buffers.torch_arrays
    t.x[slot] = pose["x"]
    t.y[slot] = pose["y"]
    t.yaw[slot] = pose["yaw"]
    t.prev_x[slot] = pose["x"]
    t.prev_y[slot] = pose["y"]
    t.prev_yaw[slot] = pose["yaw"]
    t.vx[slot] = speed * pose["tx"]
    t.vy[slot] = speed * pose["ty"]
    t.yaw_rate[slot] = 0.0
    t.steer[slot] = 0.0
    t.steer_logical[slot] = 0.0
    t.i_drive[slot] = 0.0
    t.i_brake[slot] = 0.0
    t.i_drive_cmd[slot] = 0.0
    t.i_brake_cmd[slot] = 0.0
    t.applied_effort[slot] = 0.0
    t.frenet_segment[slot] = int(pose["segment"])
    t.frenet_s[slot] = float(pose["s"])
    t.prev_s[slot] = float(pose["s"])
    t.progress_s[slot] = 0.0
    t.frenet_ey[slot] = float(pose["ey"])
    t.boundary_distance[slot] = float(pose["boundary_distance"])
    t.done[slot] = 0
    t.timeout[slot] = 0
    t.active[slot] = 1
    t.trainable[slot] = 1
    t.contact[slot] = 0
    t.wall_contact[slot] = 0
    t.episode_step[slot] = 0
    t.lidar_range_noise_std[slot] = 0.0
    t.lidar_dropout_prob[slot] = 0.0
    t.lidar_far_dropout_prob[slot] = 0.0


def spawn_curriculum(
    sim,
    *,
    rest_frac: float,
    rng: np.random.Generator,
    rest_s: float = COURTYARD_S,
    rest_ey: float = COURTYARD_EY,
    rest_yaw: float = COURTYARD_YAW,
) -> None:
    t = sim.buffers.torch_arrays
    n = int(t.x.shape[0])
    length = float(sim.geom.track_length[0])
    for slot in range(n):
        track_id = int(t.track_id[slot].item())
        u = float(rng.random())
        if u < rest_frac:
            s, ey, speed, yaw = rest_s, rest_ey, 0.0, rest_yaw
        elif u < rest_frac + 0.4:
            s = float(rng.uniform(0.0, length))
            ey = float(rng.uniform(-0.12, 0.12))
            speed = float(rng.uniform(0.0, 1.5))
            yaw = None
        else:
            s = float(rng.uniform(0.0, length))
            ey = float(rng.uniform(-0.18, 0.18))
            speed = float(rng.uniform(1.0, 2.8))
            yaw = None
        pose = pose_on_track(sim.geom, track_id, s, ey)
        if yaw is not None:
            pose["yaw"] = float(yaw)
            pose["tx"] = float(math.cos(yaw))
            pose["ty"] = float(math.sin(yaw))
        write_slot_pose(sim, slot, pose, speed)
    if hasattr(sim, "_invalidate_cuda_graph"):
        sim._invalidate_cuda_graph()
    sim.rebuild_sensors()


def spawn_from_rest(sim) -> dict[str, float]:
    t = sim.buffers.torch_arrays
    n = int(t.x.shape[0])
    meta = None
    for slot in range(n):
        track_id = int(t.track_id[slot].item())
        pose = pose_on_track(sim.geom, track_id, COURTYARD_S, COURTYARD_EY)
        pose["yaw"] = COURTYARD_YAW
        pose["tx"] = float(math.cos(COURTYARD_YAW))
        pose["ty"] = float(math.sin(COURTYARD_YAW))
        write_slot_pose(sim, slot, pose, 0.0)
        if meta is None:
            meta = {
                "s": float(pose["s"]),
                "ey": float(pose["ey"]),
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": float(pose["yaw"]),
                "speed": 0.0,
            }
    if hasattr(sim, "_invalidate_cuda_graph"):
        sim._invalidate_cuda_graph()
    sim.rebuild_sensors()
    return meta or dict(NORTH_SPAWN)


def _teacher_tables(sim):
    cached = getattr(sim, "_courtyard_teacher_tables", None)
    if cached is not None:
        return cached
    device = sim.buffers.torch_arrays.x.device
    g = sim.geom
    tables = {
        "cum": torch.as_tensor(g.cum_length, device=device, dtype=torch.float32),
        "xy": torch.as_tensor(g.centerline_xy, device=device, dtype=torch.float32),
    }
    sim._courtyard_teacher_tables = tables
    return tables


def _centerline_index_at(sim, s_query: torch.Tensor) -> torch.Tensor:
    t = sim.buffers.torch_arrays
    tables = _teacher_tables(sim)
    length = sim._torch_track_length[t.track_id.long()].clamp(min=1e-3)
    s_mod = torch.remainder(s_query, length)
    start = sim._torch_offsets[t.track_id.long()]
    stop = sim._torch_offsets[t.track_id.long() + 1]
    idx = torch.searchsorted(tables["cum"], s_mod.contiguous(), right=True) - 1
    idx = torch.maximum(idx, start)
    idx = torch.minimum(idx, stop - 2)
    return idx.long()


def centerline_teacher(
    sim,
    *,
    target_speed: float = 1.7,
    kp_speed: float = 2.2,
    kp_ey: float = 0.6,
    wheelbase: float = 0.325,
    max_steer: float = MAX_STEER_RAD,
    delta_max: float = STEERING_DELTA_MAX_RAD,
) -> torch.Tensor:
    """Privileged pure-pursuit. Actor never sees these Frenet / centerline features."""
    t = sim.buffers.torch_arrays
    tables = _teacher_tables(sim)
    speed = torch.sqrt(t.vx * t.vx + t.vy * t.vy)
    lookahead = torch.clamp(0.60 + 0.28 * speed, min=0.60, max=1.15)
    tgt = tables["xy"].index_select(0, _centerline_index_at(sim, t.frenet_s + lookahead))
    dx = tgt[:, 0] - t.x
    dy = tgt[:, 1] - t.y
    cosine = torch.cos(t.yaw)
    sine = torch.sin(t.yaw)
    x_body = cosine * dx + sine * dy
    y_body = -sine * dx + cosine * dy
    alpha = torch.atan2(y_body, torch.clamp(x_body, min=0.05))
    steer_pp = torch.atan(2.0 * float(wheelbase) * torch.sin(alpha) / lookahead)
    steer_target = torch.clamp(steer_pp - float(kp_ey) * t.frenet_ey, -max_steer, max_steer)
    steering = (steer_target - t.steer_logical) / max(float(delta_max), 1e-6)
    v_cap = float(target_speed) / (1.0 + 1.6 * alpha.abs() + 1.2 * t.frenet_ey.abs())
    v_cap = torch.clamp(v_cap, min=0.55, max=float(target_speed))
    throttle = float(kp_speed) * (v_cap - speed)
    return torch.stack(
        [throttle.clamp(-0.999, 0.999), steering.clamp(-0.999, 0.999)],
        dim=-1,
    )


def actor_gaussian(actor, obs, hidden, reset_mask):
    hidden = actor._apply_reset_mask(hidden, reset_mask)
    trunk = actor.encode_trunk(obs, None)
    out, h_n = actor.gru(trunk.unsqueeze(1), hidden.unsqueeze(0).contiguous())
    feat = out.squeeze(1)
    net_out = actor.net(feat.to(dtype=torch.float32))
    mu = actor.mu_layer(net_out)
    log_std = torch.clamp(actor.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std, h_n.squeeze(0)


def _fix_styles(sim, cfg: ExperimentConfig) -> None:
    style = deployment_style(cfg.evaluation.conservative_deployment_style)
    sim.apply_styles([style for _ in range(sim.layout.num_slots)])


def build_courtyard_sim(cfg: ExperimentConfig, device: str, *, no_respawn: bool):
    cache = Path(cfg.tracks.manifest_path).expanduser().parent
    atlas = load_atlas(str(cache)).view()
    sim = build_simulator(cfg, atlas, device, sync_no_respawn=no_respawn)
    _fix_styles(sim, cfg)
    return sim


def load_student(cfg: ExperimentConfig, checkpoint: Path, device: str):
    actor = load_actor_from_checkpoint(cfg, checkpoint, device=device)
    attach_baked_condition(actor, baked_normalized_condition())
    actor.to(device)
    actor.train()
    return actor


@torch.no_grad()
def eval_from_rest(
    sim,
    actor,
    *,
    seconds: float = 60.0,
    teacher: bool = False,
    collect_poses: bool = False,
) -> dict[str, Any]:
    actor.eval()
    prev_flag = bool(sim.sync_no_respawn)
    sim.sync_no_respawn = True
    spawn = spawn_from_rest(sim)
    t = sim.buffers.torch_arrays
    n = int(t.x.shape[0])
    hz = float(sim.cfg.agents.control_hz)
    steps = max(2, int(round(seconds * hz)))
    hidden = actor.initial_hidden(n, device=str(t.x.device))
    alive = torch.ones(n, device=t.x.device, dtype=torch.bool)
    dist = torch.zeros(n, device=t.x.device, dtype=torch.float32)
    spd_sum = torch.zeros(n, device=t.x.device, dtype=torch.float32)
    spd_max = torch.zeros(n, device=t.x.device, dtype=torch.float32)
    survival = torch.zeros(n, device=t.x.device, dtype=torch.float32)
    prev_xy = torch.stack([t.x.clone(), t.y.clone()], dim=-1)
    obs = t.sensor_obs
    poses: list[np.ndarray] = []
    for step in range(steps):
        if not bool(alive.any()):
            break
        reset = (~alive).to(dtype=torch.float32)
        if teacher:
            actions = centerline_teacher(sim)
            hidden = actor.initial_hidden(n, device=str(t.x.device))
        else:
            out = actor(
                obs,
                None,
                hidden,
                reset_mask=reset,
                deterministic=True,
            )
            actions = out.actions
            hidden = out.hidden
        result = sim.step(actions)
        obs = result["sensor_obs"]
        if collect_poses and step % 4 == 0 and len(poses) < 250:
            poses.append(
                np.array(
                    [
                        [
                            float(t.x[0].item()),
                            float(t.y[0].item()),
                            float(t.yaw[0].item()),
                            float(t.active[0].item()),
                        ]
                    ],
                    dtype=np.float32,
                )
            )
        xy = torch.stack([t.x, t.y], dim=-1)
        delta = torch.linalg.norm(xy - prev_xy, dim=-1)
        dist = dist + delta * alive.to(dtype=torch.float32)
        speed = torch.sqrt(t.vx * t.vx + t.vy * t.vy)
        spd_sum = spd_sum + speed * alive.to(dtype=torch.float32)
        spd_max = torch.maximum(spd_max, speed * alive.to(dtype=torch.float32))
        survival = survival + alive.to(dtype=torch.float32) / hz
        done = result["done"].to(dtype=torch.bool) | (t.active <= 0)
        alive = alive & ~done
        prev_xy = xy
        if int(alive.sum().item()) == 0:
            break
    sim.sync_no_respawn = prev_flag
    # Slot 0 is the from-rest reference. Max-over-worlds is GPU noise, not a lap.
    ref = 0
    crashed = not bool(alive[ref].item())
    mean_speed = float((spd_sum[ref] / max(float(survival[ref]) * hz, 1.0)).item())
    dist_m = float(dist[ref].item())
    payload = {
        "survival_s": float(survival[ref].item()),
        "dist_m": dist_m,
        "crashed": crashed,
        "mean_speed_mps": mean_speed,
        "max_speed_mps": float(spd_max[ref].item()),
        "n_slots": n,
        "median_dist_m": float(dist.median().item()),
        "max_dist_m": float(dist.max().item()),
        "spawn": spawn,
        "teacher": bool(teacher),
        "full_lap": dist_m >= COURTYARD_LENGTH_M and not crashed,
        "multi_lap": dist_m >= 2.0 * COURTYARD_LENGTH_M and not crashed,
    }
    if collect_poses:
        payload["poses"] = poses
    actor.train()
    return payload


def eval_rank(ev: dict[str, Any]) -> tuple:
    return (
        int(bool(ev.get("multi_lap"))),
        int(bool(ev.get("full_lap"))),
        0 if ev.get("crashed") else 1,
        float(ev.get("dist_m") or 0.0),
        float(ev.get("survival_s") or 0.0),
    )


def train_courtyard(
    cfg: ExperimentConfig,
    *,
    checkpoint: Path,
    run_dir: Path,
    device: str,
    num_updates: int,
    rollout_len: int,
    lr: float,
    eval_every: int,
    eval_seconds: float,
    anneal_updates: int,
    target_speed: float,
    seed: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    sim = build_courtyard_sim(cfg, device, no_respawn=False)
    eval_cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=1,
            max_agents_per_world=1,
            static_opponents_per_world=0,
        ),
        agents=replace(cfg.agents, async_respawn=False),
    )
    eval_sim = build_courtyard_sim(eval_cfg, device, no_respawn=True)
    actor = load_student(cfg, checkpoint, device)
    if hasattr(actor, "gru"):
        actor.gru.flatten_parameters()
    frozen = copy.deepcopy(actor).eval()
    if hasattr(frozen, "gru"):
        frozen.gru.flatten_parameters()
    for param in frozen.parameters():
        param.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    t = sim.buffers.torch_arrays
    n = int(t.x.shape[0])
    hidden = actor.initial_hidden(n, device=device)

    for update in range(int(num_updates)):
        # Teacher-only BC until the 1-world student laps; mix/PG collapse the clone.
        actor.train()
        obs_buf = []
        teacher_buf = []
        reset_buf = []
        spawn_curriculum(sim, rest_frac=0.8, rng=rng)
        hidden = actor.initial_hidden(n, device=device)
        hid0 = hidden.detach()
        obs = t.sensor_obs
        reset = torch.zeros(n, device=device, dtype=torch.float32)
        for _step in range(int(rollout_len)):
            teacher = centerline_teacher(sim, target_speed=target_speed)
            _mu, _log_std, next_hidden = actor_gaussian(actor, obs, hidden, reset)
            result = sim.step(teacher)
            done = result["done"].to(dtype=torch.float32)
            obs_buf.append(obs.detach())
            teacher_buf.append(teacher.detach())
            reset_buf.append(reset.detach())
            hidden = next_hidden.detach()
            reset = done
            obs = result["sensor_obs"]
            if bool((t.active <= 0).all()):
                spawn_curriculum(sim, rest_frac=0.8, rng=rng)
                hidden = actor.initial_hidden(n, device=device)
                reset = torch.ones(n, device=device, dtype=torch.float32)
                obs = t.sensor_obs

        obs_seq = torch.stack(obs_buf, dim=1)
        teacher_seq = torch.stack(teacher_buf, dim=1)
        reset_seq = torch.stack(reset_buf, dim=1)
        logp, _ent, _h = actor.evaluate_actions_sequence(
            obs_seq,
            None,
            hid0,
            teacher_seq,
            reset_mask=reset_seq,
        )[:3]
        bc = -logp.mean()
        mu_s, std_s, _, _ = actor.distribution_params_sequence(
            obs_seq, None, hid0, reset_seq
        )
        with torch.no_grad():
            mu_b, std_b, _, _ = frozen.distribution_params_sequence(
                obs_seq, None, hid0, reset_seq
            )
        kl = gaussian_kl(
            mu_s.reshape(-1, 2),
            torch.log(std_s.clamp(min=1e-6)).reshape(-1, 2),
            mu_b.reshape(-1, 2),
            torch.log(std_b.clamp(min=1e-6)).reshape(-1, 2),
        )
        loss = bc + 0.02 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        opt.step()

        row = {
            "update": update,
            "loss": float(loss.detach().item()),
            "bc": float(bc.detach().item()),
            "kl": float(kl.detach().item()),
            "beta": 1.0,
            "kl_coef": 0.02,
        }
        if update % int(eval_every) == 0 or update + 1 == int(num_updates):
            ev = eval_from_rest(
                eval_sim, actor, seconds=eval_seconds, teacher=False
            )
            row.update(
                {f"eval_{k}": v for k, v in ev.items() if k not in {"spawn", "poses"}}
            )
            row["eval"] = ev
            ckpt_path = run_dir / f"actor_update_{update:05d}.pt"
            export_baked_actor(cfg, actor, path=str(ckpt_path))
            if best is None or eval_rank(ev) > eval_rank(best["eval"]):
                best = {"update": update, "eval": ev, "path": str(ckpt_path)}
                export_baked_actor(cfg, actor, path=str(run_dir / "actor_best.pt"))
            print(json.dumps(row, default=str), flush=True)
            if ev.get("multi_lap"):
                history.append(row)
                break
        else:
            if update % 5 == 0:
                print(json.dumps(row, default=str), flush=True)
        history.append(row)

    winner_src = Path(best["path"]) if best else run_dir / "actor_best.pt"
    winner_dst = run_dir / "winner_actor.pt"
    if winner_src.is_file():
        shutil.copy2(winner_src, winner_dst)
    elif (run_dir / "actor_best.pt").is_file():
        shutil.copy2(run_dir / "actor_best.pt", winner_dst)
    video = None
    if winner_dst.is_file():
        try:
            from gigaflow_f1tenth.visualization import render_side_by_side

            winner_actor = load_student(eval_cfg, winner_dst, device)
            if hasattr(winner_actor, "gru"):
                winner_actor.gru.flatten_parameters()
            left = eval_from_rest(
                eval_sim, frozen, seconds=50.0, collect_poses=True
            )
            right = eval_from_rest(
                eval_sim, winner_actor, seconds=50.0, collect_poses=True
            )
            cache = Path(cfg.tracks.manifest_path).expanduser().parent
            atlas = load_atlas(str(cache)).view()
            video = render_side_by_side(
                atlas,
                left.get("poses") or [],
                right.get("poses") or [],
                run_dir / "side_by_side.mp4",
                car_length=cfg.agents.car_length_m,
                car_width=cfg.agents.car_width_m,
            )
        except Exception as exc:
            print(json.dumps({"render_error": str(exc)}), flush=True)
    report = {
        "best": best,
        "updates": len(history),
        "history_tail": history[-8:],
        "run_dir": str(run_dir),
        "winner": str(winner_dst),
        "video": None if video is None else str(video),
    }
    (run_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=int(args.num_worlds),
            max_agents_per_world=1,
            static_opponents_per_world=0,
            device=args.device,
            solo_world_fraction=1.0,
            density_bins=("sparse",),
            width_scales=(1.0,),
        ),
        agents=replace(cfg.agents, async_respawn=False),
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.teacher_only:
        sim = build_courtyard_sim(cfg, args.device, no_respawn=True)
        actor = load_student(cfg, Path(args.checkpoint), args.device)
        ev = eval_from_rest(
            sim, actor, seconds=float(args.eval_seconds), teacher=True
        )
        (run_dir / "teacher_eval.json").write_text(json.dumps(ev, indent=2) + "\n")
        print(json.dumps(ev, indent=2, default=str), flush=True)
        return ev
    if args.eval_only:
        sim = build_courtyard_sim(cfg, args.device, no_respawn=True)
        actor = load_student(cfg, Path(args.checkpoint), args.device)
        ev = eval_from_rest(
            sim, actor, seconds=float(args.eval_seconds), teacher=False
        )
        (run_dir / "student_eval.json").write_text(json.dumps(ev, indent=2) + "\n")
        print(json.dumps(ev, indent=2, default=str), flush=True)
        return ev
    return train_courtyard(
        cfg,
        checkpoint=Path(args.checkpoint),
        run_dir=run_dir,
        device=args.device,
        num_updates=int(args.num_updates),
        rollout_len=int(args.rollout_len),
        lr=float(args.lr),
        eval_every=int(args.eval_every),
        eval_seconds=float(args.eval_seconds),
        anneal_updates=int(args.anneal_updates),
        target_speed=float(args.target_speed),
        seed=int(args.seed),
    )

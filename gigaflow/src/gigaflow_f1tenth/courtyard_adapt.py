"""Courtyard_2 multi-lap adaptation of a baked Gigaflow actor.

The student ``ConditionedLidarGRUActor`` sees only 1,097-D LiDAR + NIME
proprio and the hardcoded IRL condition. Centerline geometry is used by the
spawn pin and by the DAgger teacher, never as an actor input.

Online PPO here trains a **fresh** critic; it does not PPO-resume actor-only
weights (that path stays rejected in ``offline_finetune``).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp

from gigaflow_f1tenth.config import config_from_dict, load_config
from gigaflow_f1tenth.evaluation import _suite_world_overrides
from gigaflow_f1tenth.irl_identity import attach_baked_condition
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.offline_finetune import (
    _write_actor,
    gaussian_kl,
    load_offline_actor,
)
from gigaflow_f1tenth.rewards import deployment_style
from gigaflow_f1tenth.sim.layout_local import (
    MAX_STEER_RAD,
    STEERING_DELTA_MAX_RAD,
)
from gigaflow_f1tenth.sim.vehicle import VehicleParams
from gigaflow_f1tenth.teleop_dataset import windows_from_trajectories
from gigaflow_f1tenth.tracks import (
    build_track_arrays,
    centerline_table_from_arrays,
    pack_built_tracks,
)
from gigaflow_f1tenth.visualization import FOLLOW_RADIUS_M, render_frame

_GIGA = Path(__file__).resolve().parents[2]
DEFAULT_CFG = _GIGA / "configs" / "production_rtx4080.yaml"
COURTYARD_CSV = (
    _GIGA / "tests" / "fixtures" / "tracks" / "courtyard_2_centerline.csv"
)


def _first_file(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return Path(paths[0])


CKPT_BASE = _first_file(
    (
        Path("/tmp/irl-gigaflow-ft-campaign/actor_005700_irl_baked.pt"),
        Path("/home/ubuntu/projects/F1tenth-gigaflow-ckpts/actor_005700.pt"),
        Path("/home/squid/projects/F1tenth-gigaflow-ckpts/actor_005700.pt"),
        Path("actor_005700.pt"),
    )
)
# Pinned from-rest spawn used by the courtyard follow eval.
SPAWN_S = 10.876999855041504
SPAWN_EY = 0.0
WHEELBASE = 0.325
LAP_TARGET_M = 17.6


def wrap_angle(yaw: np.ndarray | float) -> np.ndarray | float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def wrap_s(s: np.ndarray | float, length: float) -> np.ndarray | float:
    return np.mod(s, float(length))


class CenterlinePath:
    """Closed centerline with curvature; teacher/spawn only, never actor input."""

    def __init__(
        self,
        xy: np.ndarray,
        widths_rl: np.ndarray,
        *,
        name: str = "track",
    ) -> None:
        self.name = str(name)
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.xy.shape[0] < 8:
            raise ValueError("centerline needs at least 8 vertices")
        self.widths_rl = np.asarray(widths_rl, dtype=np.float64).reshape(-1, 2)
        if self.widths_rl.shape[0] != self.xy.shape[0]:
            raise ValueError("widths must match centerline length")
        closed = np.vstack([self.xy, self.xy[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.segment_length = seg.astype(np.float64)
        self.cum = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        self.length = float(np.sum(seg))
        tang = np.diff(closed, axis=0)
        tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        self.tangents = tang
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)
        s_pts = self.cum.copy()
        # Closed-loop gradients for signed curvature (1/m, left positive).
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        dx = np.gradient(np.r_[x, x[0]], np.r_[s_pts, self.length])[:-1]
        dy = np.gradient(np.r_[y, y[0]], np.r_[s_pts, self.length])[:-1]
        ddx = np.gradient(np.r_[dx, dx[0]], np.r_[s_pts, self.length])[:-1]
        ddy = np.gradient(np.r_[dy, dy[0]], np.r_[s_pts, self.length])[:-1]
        speed2 = np.maximum(dx * dx + dy * dy, 1e-9)
        self.kappa = (dx * ddy - dy * ddx) / np.power(speed2, 1.5)

    @classmethod
    def from_csv(cls, path: Path | str, *, name: str = "courtyard_2") -> "CenterlinePath":
        raw = np.genfromtxt(
            path,
            delimiter=",",
            names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"],
            skip_header=1,
            dtype=np.float64,
            comments=None,
        )
        xy = np.stack([raw["x_m"], raw["y_m"]], axis=1)
        widths = np.stack([raw["w_tr_right_m"], raw["w_tr_left_m"]], axis=1)
        return cls(xy, widths, name=name)

    def _index(self, s: float) -> tuple[int, float]:
        wrapped = float(wrap_s(s, self.length))
        if not math.isfinite(wrapped):
            wrapped = 0.0
        i = int(np.searchsorted(self.cum, wrapped, side="right") - 1)
        i = int(np.clip(i, 0, self.xy.shape[0] - 1))
        alpha = 0.0
        denom = float(self.segment_length[i])
        if denom > 1e-9:
            alpha = float(np.clip((wrapped - float(self.cum[i])) / denom, 0.0, 1.0))
        return i, alpha

    def interp(self, values: np.ndarray, s: float) -> np.ndarray:
        i, alpha = self._index(s)
        n = int(values.shape[0])
        i = min(i, n - 1)
        j = (i + 1) % n
        return (1.0 - alpha) * values[i] + alpha * values[j]

    def xy_at(self, s: float) -> np.ndarray:
        return self.interp(self.xy, s)

    def yaw_at(self, s: float) -> float:
        t = self.interp(self.tangents, s)
        return float(math.atan2(t[1], t[0]))

    def kappa_at(self, s: float) -> float:
        return float(self.interp(self.kappa, s))

    def pose_at(self, s: float, ey: float = 0.0) -> dict[str, float]:
        i, alpha = self._index(s)
        j = (i + 1) % self.xy.shape[0]
        center = (1.0 - alpha) * self.xy[i] + alpha * self.xy[j]
        normal = (1.0 - alpha) * self.normals[i] + alpha * self.normals[j]
        nrm = float(np.linalg.norm(normal))
        if nrm > 1e-9:
            normal = normal / nrm
        xy = center + float(ey) * normal
        wr = float((1.0 - alpha) * self.widths_rl[i, 0] + alpha * self.widths_rl[j, 0])
        wl = float((1.0 - alpha) * self.widths_rl[i, 1] + alpha * self.widths_rl[j, 1])
        yaw = self.yaw_at(s)
        return {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "yaw": float(yaw),
            "s": float(wrap_s(s, self.length)),
            "ey": float(ey),
            "segment": int(i),
            "boundary_distance": float(min(wl - ey, wr + ey)),
            "width_left": wl,
            "width_right": wr,
        }

    def required_steer(self, s: float, lookahead_m: float = 0.8) -> float:
        kappa = self.kappa_at(s + lookahead_m)
        return float(np.clip(math.atan(WHEELBASE * kappa), -MAX_STEER_RAD, MAX_STEER_RAD))


def teacher_action(
    path: CenterlinePath,
    *,
    x: float,
    y: float,
    yaw: float,
    s: float,
    ey: float,
    vx: float,
    steer_now: float,
    lookahead_gain: float = 0.7,
    stanley_k: float = 2.6,
    a_lat: float = 1.4,
    v_max: float = 1.55,
) -> np.ndarray:
    """Lookahead PP + Stanley. Centerline is teacher-only."""
    speed = max(float(vx), 0.0)
    ld = float(np.clip(1.0 + lookahead_gain * speed, 1.0, 2.2))
    target = path.xy_at(s + ld)
    dx = float(target[0] - x)
    dy = float(target[1] - y)
    ca = math.cos(yaw)
    sa = math.sin(yaw)
    local_x = ca * dx + sa * dy
    local_y = -sa * dx + ca * dy
    alpha = math.atan2(local_y, max(local_x, 1e-3))
    pp = math.atan(2.0 * WHEELBASE * math.sin(alpha) / max(ld, 0.1))
    heading = float(wrap_angle(path.yaw_at(s) - yaw))
    st = heading + math.atan(stanley_k * (-float(ey)) / (speed + 0.4))
    mix = float(np.clip(0.5 + 0.15 * speed, 0.45, 0.75))
    steer = mix * pp + (1.0 - mix) * st
    kappa = abs(path.kappa_at(s + 0.6 * ld))
    v_corner = math.sqrt(max(a_lat, 1e-3) / max(kappa, 0.12))
    v_des = float(np.clip(v_corner, 0.7, v_max))
    force = float(np.clip((v_des - speed) / 1.6, -1.0, 1.0))
    steer = float(np.clip(steer, -MAX_STEER_RAD, MAX_STEER_RAD))
    delta = float(
        np.clip((steer - float(steer_now)) / STEERING_DELTA_MAX_RAD, -1.0, 1.0)
    )
    return np.asarray([force, delta], dtype=np.float32)


def courtyard_atlas(max_agents: int = 1):
    path = CenterlinePath.from_csv(COURTYARD_CSV)
    table = centerline_table_from_arrays(
        path.xy[:, 0],
        path.xy[:, 1],
        path.widths_rl[:, 0],
        path.widths_rl[:, 1],
    )
    built = build_track_arrays(
        table,
        name="courtyard_2",
        source_sha="local-courtyard-2",
        max_agents=max_agents,
        license_id="local-fixture",
        optional_local=True,
    )
    return pack_built_tracks([built]).view(), path


def suite_cfg(base_cfg, device: str, num_worlds: int = 1):
    raw = _suite_world_overrides("solo", base_cfg)
    worlds = dict(raw["worlds"])
    worlds["num_worlds"] = int(num_worlds)
    worlds["device"] = device
    worlds["static_opponents_per_world"] = 0
    worlds["width_scales"] = [1.0]
    raw["worlds"] = worlds
    agents = dict(raw["agents"])
    agents["async_respawn"] = False
    agents["physics_randomize"] = False
    raw["agents"] = agents
    evaluation = dict(raw["evaluation"])
    evaluation["device"] = device
    evaluation["sync_no_respawn"] = True
    evaluation["viz_enabled"] = False
    raw["evaluation"] = evaluation
    wandb = dict(raw.get("wandb") or {})
    wandb["enabled"] = False
    wandb["mode"] = "disabled"
    raw["wandb"] = wandb
    raw["seed"] = 0
    return config_from_dict(raw, check_training_budget=False)


def _zero_lidar_noise(t, flat: int) -> None:
    t.lidar_range_noise_std[flat] = 0.0
    t.lidar_dropout_prob[flat] = 0.0
    t.lidar_far_dropout_prob[flat] = 0.0
    t.lidar_angle_bias[flat] = 0.0
    t.lidar_extrinsic_x[flat] = 0.0
    t.lidar_extrinsic_y[flat] = 0.0
    t.lidar_extrinsic_yaw[flat] = 0.0
    t.lidar_sector_start[flat] = 0
    t.lidar_sector_width[flat] = 0


def pin_slot(
    sim,
    path: CenterlinePath,
    s: float,
    *,
    ey: float = 0.0,
    vx: float = 0.0,
    flat: int = 0,
) -> dict[str, float]:
    pose = path.pose_at(s, ey)
    t = sim.buffers.torch_arrays
    wheel = float(VehicleParams().wheel_radius)
    t.x[flat] = pose["x"]
    t.y[flat] = pose["y"]
    t.yaw[flat] = pose["yaw"]
    t.prev_x[flat] = pose["x"]
    t.prev_y[flat] = pose["y"]
    t.prev_yaw[flat] = pose["yaw"]
    t.vx[flat] = float(vx)
    t.vy[flat] = 0.0
    t.yaw_rate[flat] = 0.0
    t.steer[flat] = 0.0
    t.steer_logical[flat] = 0.0
    t.i_drive[flat] = 0.0
    t.i_brake[flat] = 0.0
    t.i_drive_cmd[flat] = 0.0
    t.i_brake_cmd[flat] = 0.0
    t.applied_effort[flat] = 0.0
    t.ax[flat] = 0.0
    t.ay[flat] = 0.0
    t.imu_gz[flat] = 0.0
    t.width_scale[flat] = 1.0
    t.frenet_segment[flat] = int(pose["segment"])
    t.frenet_s[flat] = float(pose["s"])
    t.prev_s[flat] = float(pose["s"])
    t.frenet_ey[flat] = float(pose["ey"])
    t.boundary_distance[flat] = float(pose["boundary_distance"])
    t.progress_s[flat] = 0.0
    t.episode_step[flat] = 0
    t.stalled_steps[flat] = 0
    t.done[flat] = 0
    t.timeout[flat] = 0
    t.reset_mask[flat] = 1
    t.active[flat] = 1
    t.trainable[flat] = 1
    t.contact[flat] = 0
    t.wall_contact[flat] = 0
    for name in (
        "executed_long_0",
        "executed_long_1",
        "executed_long_2",
        "executed_steer_0",
        "executed_steer_1",
        "executed_steer_2",
        "executed_steer_3",
        "executed_steer_4",
        "executed_steer_5",
        "executed_steer_6",
    ):
        getattr(t, name)[flat] = 0.0
    _zero_lidar_noise(t, flat)
    om = np.array(sim.buffers.omega.numpy(), copy=True)
    om[flat] = np.full(4, float(vx) / wheel, dtype=np.float32)
    wp.copy(
        sim.buffers.omega,
        wp.array(om, dtype=wp.vec4f, device=sim.buffers.wp_device),
    )
    sim._invalidate_cuda_graph()
    return pose


def spawn_menu(path: CenterlinePath) -> list[tuple[float, float]]:
    """From-rest spawn plus 8 even centerline stations. vx attached by caller."""
    rest = (SPAWN_S, SPAWN_EY)
    even = [(i * path.length / 8.0, 0.0) for i in range(8)]
    return [rest, *even]


def build_courtyard_sim(device: str, num_worlds: int = 1):
    cfg0 = load_config(DEFAULT_CFG)
    cfg = suite_cfg(cfg0, device, num_worlds=num_worlds)
    atlas, path = courtyard_atlas(max_agents=1)
    sim = build_simulator(cfg, atlas, device, sync_no_respawn=True)
    return sim, cfg, atlas, path


@torch.no_grad()
def diagnose_actor(
    actor,
    *,
    device: str,
    seconds: float = 8.0,
) -> dict[str, Any]:
    sim, cfg, _atlas, path = build_courtyard_sim(device, num_worlds=1)
    actor.eval()
    attach_baked_condition(actor)
    actor.gru.flatten_parameters()
    sim.reset_all(0)
    sim.apply_styles([deployment_style("centered_high_collision")])
    pose = pin_slot(sim, path, SPAWN_S, ey=SPAWN_EY, vx=0.0, flat=0)
    sim.rebuild_sensors()
    hidden = actor.initial_hidden(1, device)
    hz = float(cfg.agents.control_hz)
    horizon = int(round(float(seconds) * hz))
    t = sim.buffers.torch_arrays
    rows: list[dict[str, float]] = []
    crashed = False
    for step in range(horizon):
        obs = sim.action_observation().to(device)
        out = actor.forward(obs, None, hidden, deterministic=True)
        hidden = out.hidden
        actions = out.actions.detach()
        force = float(actions[0, 0].item())
        delta = float(actions[0, 1].item())
        s = float(t.frenet_s[0].item())
        ey = float(t.frenet_ey[0].item())
        vx = float(t.vx[0].item())
        yaw = float(t.yaw[0].item())
        steer = float(t.steer[0].item())
        kappa = path.kappa_at(s)
        req = path.required_steer(s, lookahead_m=0.8)
        req_delta = float(
            np.clip((req - steer) / STEERING_DELTA_MAX_RAD, -1.0, 1.0)
        )
        clearance = float(t.boundary_distance[0].item())
        rows.append(
            {
                "t": (step + 1) / hz,
                "s": s,
                "ey": ey,
                "vx": vx,
                "yaw": yaw,
                "yaw_track": path.yaw_at(s),
                "steer": steer,
                "force": force,
                "delta": delta,
                "kappa": kappa,
                "required_steer": req,
                "required_delta": req_delta,
                "clearance": clearance,
                "progress": float(t.progress_s[0].item()),
            }
        )
        sim.step(actions)
        if not bool((t.active[0] > 0).item()):
            crashed = True
            break
    last = rows[-1] if rows else {}
    near = [row for row in rows if row["progress"] >= 4.0] or rows[-8:]
    summary = {
        "spawn": pose,
        "track_length_m": path.length,
        "crashed": crashed,
        "steps": len(rows),
        "survival_s": last.get("t", 0.0),
        "progress_m": last.get("progress", 0.0),
        "crash_s": last.get("s", math.nan),
        "mean_delta_near_corner": float(np.mean([abs(r["delta"]) for r in near])),
        "mean_required_delta_near_corner": float(
            np.mean([abs(r["required_delta"]) for r in near])
        ),
        "max_steer": float(max((abs(r["steer"]) for r in rows), default=0.0)),
        "max_required_steer": float(
            max((abs(r["required_steer"]) for r in rows), default=0.0)
        ),
        "min_clearance": float(min((r["clearance"] for r in rows), default=0.0)),
        "steer_ramp_steps_to_max": int(
            math.ceil(MAX_STEER_RAD / STEERING_DELTA_MAX_RAD)
        ),
        "rows": rows,
    }
    return summary


def _value_net(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    )


def _condition(actor, batch: int, device) -> torch.Tensor:
    return (
        actor.baked_condition.to(device=device, dtype=torch.float32)
        .reshape(1, -1)
        .expand(int(batch), -1)
        .contiguous()
    )


def _gru_features(actor, obs, hidden, resets, cond):
    batch, length, _ = obs.shape
    flat = obs.reshape(batch * length, obs.shape[-1])
    flat_cond = cond.unsqueeze(1).expand(batch, length, -1)
    flat_cond = flat_cond.reshape(batch * length, cond.shape[-1])
    trunk = actor.encode_trunk(flat, flat_cond).view(batch, length, -1)
    gru_out, h = actor._gru_sequence(trunk, hidden, resets)
    return gru_out, h


def collect_on_policy(
    actor,
    sim,
    path: CenterlinePath,
    *,
    device: str,
    n_steps: int,
    spawn_cycle: Sequence[tuple[float, float]],
    hidden: torch.Tensor,
    spawn_i: int,
    deterministic: bool = False,
    teacher: bool = False,
    expert_mix: float = 0.0,
) -> tuple[dict[str, np.ndarray], torch.Tensor, int]:
    t = sim.buffers.torch_arrays
    n = int(t.active.shape[0])
    obs_l: list[np.ndarray] = []
    act_l: list[np.ndarray] = []
    teach_l: list[np.ndarray] = []
    rew_l: list[np.ndarray] = []
    done_l: list[np.ndarray] = []
    reset_l: list[np.ndarray] = []
    logp_l: list[np.ndarray] = []
    actor.train(not deterministic)
    attach_baked_condition(actor)
    actor.gru.flatten_parameters()
    for _ in range(int(n_steps)):
        reset = torch.zeros(n, device=device, dtype=torch.float32)
        for slot in range(n):
            if int(t.active[slot].item()) <= 0:
                s, ey = spawn_cycle[spawn_i % len(spawn_cycle)]
                vx = 0.0 if spawn_i % len(spawn_cycle) == 0 else 0.4
                pin_slot(sim, path, s, ey=ey, vx=vx, flat=slot)
                reset[slot] = 1.0
                spawn_i += 1
        sim.rebuild_sensors()
        obs = sim.action_observation().to(device).clone()
        with torch.set_grad_enabled(False):
            out = actor.forward(
                obs, None, hidden, reset_mask=reset, deterministic=deterministic
            )
        hidden = out.hidden.detach()
        actions = out.actions.detach()
        teacher_np = np.zeros((n, 2), dtype=np.float32)
        if teacher:
            for slot in range(n):
                teacher_np[slot] = teacher_action(
                    path,
                    x=float(t.x[slot].item()),
                    y=float(t.y[slot].item()),
                    yaw=float(t.yaw[slot].item()),
                    s=float(t.frenet_s[slot].item()),
                    ey=float(t.frenet_ey[slot].item()),
                    vx=float(t.vx[slot].item()),
                    steer_now=float(t.steer[slot].item()),
                )
            if float(expert_mix) > 0.0:
                mix = torch.rand(n, device=device) < float(expert_mix)
                teach_t = torch.as_tensor(teacher_np, device=device)
                actions = torch.where(mix.unsqueeze(-1), teach_t, actions)
        prev_progress = t.progress_s.clone()
        info = sim.step(actions)
        ds = (t.progress_s - prev_progress).detach()
        wall = info["wall_contact"].to(dtype=torch.float32)
        near = (t.boundary_distance.detach() < 0.22).to(dtype=torch.float32)
        shaped = (2.5 * ds - 4.0 * wall - 0.8 * near).clamp(-6.0, 6.0)
        obs_l.append(obs.detach().cpu().numpy())
        act_l.append(actions.cpu().numpy())
        teach_l.append(teacher_np)
        rew_l.append(shaped.cpu().numpy())
        done_l.append(info["done"].to(dtype=torch.float32).cpu().numpy())
        reset_l.append(reset.cpu().numpy())
        logp_l.append(out.log_prob.detach().cpu().numpy())
    return (
        {
            "obs": np.stack(obs_l, axis=1),
            "actions": np.stack(act_l, axis=1),
            "teacher": np.stack(teach_l, axis=1),
            "rewards": np.stack(rew_l, axis=1),
            "dones": np.stack(done_l, axis=1),
            "resets": np.stack(reset_l, axis=1),
            "logp": np.stack(logp_l, axis=1),
        },
        hidden,
        spawn_i,
    )


def run_s2(
    actor,
    base,
    payload: Mapping[str, Any],
    *,
    device: str,
    output: Path,
    bc_steps: int = 220,
    ppo_iters: int = 24,
) -> dict[str, Any]:
    sim, _cfg, _atlas, path = build_courtyard_sim(device, num_worlds=2)
    sim.reset_all(1)
    sim.apply_styles(
        [deployment_style("centered_high_collision")] * sim.layout.num_slots
    )
    cycle = spawn_menu(path)
    hidden = actor.initial_hidden(sim.layout.num_slots, device)
    spawn_i = 0
    for slot, (s, ey) in enumerate(cycle[: sim.layout.num_slots]):
        pin_slot(sim, path, s, ey=ey, vx=0.0 if slot == 0 else 0.4, flat=slot)
    sim.rebuild_sensors()
    opt = torch.optim.Adam(actor.parameters(), lr=5.0e-5)
    actor.train()
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)
    last = {"bc": 0.0, "kl": 0.0}
    rng = np.random.default_rng(2)
    trajs: list[dict[str, np.ndarray]] = []
    n = int(sim.layout.num_slots)
    tlen = 0
    for dagger_round in range(3):
        mix = 0.55 * (1.0 - dagger_round / 3.0)
        batch, hidden, spawn_i = collect_on_policy(
            actor,
            sim,
            path,
            device=device,
            n_steps=240,
            spawn_cycle=cycle,
            hidden=hidden,
            spawn_i=spawn_i,
            deterministic=False,
            teacher=True,
            expert_mix=mix,
        )
        n, tlen, _ = batch["obs"].shape
        for i in range(n):
            trajs.append(
                {
                    "obs": batch["obs"][i],
                    "actions": batch["teacher"][i],
                    "rewards": batch["rewards"][i].astype(np.float32),
                    "dones": batch["dones"][i].astype(np.float32),
                    "resets": batch["resets"][i].astype(np.float32),
                }
            )
        windows = windows_from_trajectories(trajs, window=16, stride=4)
        n_win = int(windows["obs"].shape[0])
        print(
            json.dumps(
                {"s2_dagger_round": dagger_round, "mix": mix, "n_win": n_win}
            ),
            flush=True,
        )
        for step in range(max(int(bc_steps) // 3, 40)):
            idx = rng.integers(0, n_win, size=min(16, n_win))
            obs = torch.as_tensor(windows["obs"][idx], device=device)
            teacher = torch.as_tensor(windows["actions"][idx], device=device)
            resets = torch.as_tensor(windows["resets"][idx], device=device)
            bsz = int(obs.shape[0])
            cond = _condition(actor, bsz, device)
            h0 = actor.initial_hidden(bsz, device)
            out = actor.evaluate_actions_sequence(
                obs,
                cond,
                h0,
                teacher,
                reset_mask=resets,
                return_diagnostics=True,
                return_mean=True,
            )
            _logp, _ent, _h, log_std, mean = out
            with torch.no_grad():
                bout = base.evaluate_actions_sequence(
                    obs,
                    cond,
                    h0,
                    teacher,
                    reset_mask=resets,
                    return_diagnostics=True,
                    return_mean=True,
                )
                _bl, _be, _bh, b_log_std, b_mean = bout
            kl = gaussian_kl(mean, log_std, b_mean, b_log_std)
            bc = (mean - teacher).square().mean()
            loss = bc + 0.01 * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            last = {"bc": float(bc.detach().item()), "kl": float(kl.detach().item())}
            if step % 40 == 0:
                print(
                    json.dumps(
                        {"s2_bc": step, "round": dagger_round, **last}
                    ),
                    flush=True,
                )
    ppo_log = ppo_fresh_critic(
        actor,
        base,
        sim,
        path,
        device=device,
        iters=int(ppo_iters),
        steps_per=128,
        kl_coef=0.02,
        hidden=hidden,
        spawn_i=spawn_i,
    )
    artifact = _write_actor(output / "s2_actor.pt", payload, actor)
    return {"artifact": artifact, "bc": last, "ppo": ppo_log, "n_slots": n, "T": tlen}


def ppo_fresh_critic(
    actor,
    base,
    sim,
    path: CenterlinePath,
    *,
    device: str,
    iters: int,
    steps_per: int,
    kl_coef: float,
    hidden: torch.Tensor | None = None,
    spawn_i: int = 0,
) -> list[dict[str, float]]:
    n = int(sim.layout.num_slots)
    if hidden is None:
        hidden = actor.initial_hidden(n, device)
    value = _value_net(int(actor.shapes().gru_hidden_dim)).to(device)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=1.0e-5)
    opt_v = torch.optim.Adam(value.parameters(), lr=3.0e-4)
    cycle = spawn_menu(path)
    logs: list[dict[str, float]] = []
    coef = float(kl_coef)
    for it in range(int(iters)):
        batch, hidden, spawn_i = collect_on_policy(
            actor,
            sim,
            path,
            device=device,
            n_steps=int(steps_per),
            spawn_cycle=cycle,
            hidden=hidden,
            spawn_i=spawn_i,
            deterministic=False,
        )
        obs = torch.as_tensor(batch["obs"], device=device)
        actions = torch.as_tensor(batch["actions"], device=device)
        old_logp = torch.as_tensor(batch["logp"], device=device)
        rewards = torch.as_tensor(batch["rewards"], device=device)
        dones = torch.as_tensor(batch["dones"], device=device)
        resets = torch.as_tensor(batch["resets"], device=device)
        nn_, tt = obs.shape[:2]
        cond = _condition(actor, nn_, device)
        h0 = actor.initial_hidden(nn_, device)
        gru_out, _h = _gru_features(actor, obs, h0, resets, cond)
        values = value(gru_out.detach()).squeeze(-1)
        with torch.no_grad():
            adv = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(nn_, device=device)
            next_v = torch.zeros(nn_, device=device)
            for t in reversed(range(tt)):
                if t == tt - 1:
                    next_nonterminal = 1.0 - dones[:, t]
                    next_values = next_v
                else:
                    next_nonterminal = 1.0 - dones[:, t]
                    next_values = values[:, t + 1]
                delta = rewards[:, t] + 0.99 * next_nonterminal * next_values - values[:, t]
                lastgaelam = delta + 0.99 * 0.95 * next_nonterminal * lastgaelam
                adv[:, t] = lastgaelam
            ret = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        actor.train()
        out = actor.evaluate_actions_sequence(
            obs,
            cond,
            h0,
            actions,
            reset_mask=resets,
            return_diagnostics=True,
            return_mean=True,
        )
        logp, ent, _h, log_std, mean = out
        with torch.no_grad():
            bout = base.evaluate_actions_sequence(
                obs,
                cond,
                h0,
                actions,
                reset_mask=resets,
                return_diagnostics=True,
                return_mean=True,
            )
            _bl, _be, _bh, b_log_std, b_mean = bout
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        pi_loss = -(torch.min(ratio * adv, clipped * adv)).mean()
        v_pred = value(gru_out.detach()).squeeze(-1)
        v_loss = F.mse_loss(v_pred, ret.detach())
        kl = gaussian_kl(mean, log_std, b_mean, b_log_std)
        loss = pi_loss + 0.5 * v_loss - 0.01 * ent.mean() + coef * kl
        opt_pi.zero_grad(set_to_none=True)
        opt_v.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_pi.step()
        opt_v.step()
        row = {
            "iter": float(it),
            "pi": float(pi_loss.detach().item()),
            "v": float(v_loss.detach().item()),
            "kl": float(kl.detach().item()),
            "ent": float(ent.mean().detach().item()),
            "kl_coef": float(coef),
            "ret": float(ret.mean().detach().item()),
        }
        logs.append(row)
        if it % 4 == 0:
            print(json.dumps({"ppo": row}), flush=True)
    return logs


@torch.no_grad()
def evaluate_from_rest(
    actor,
    *,
    device: str,
    seconds: float = 90.0,
    collect_poses: bool = False,
) -> dict[str, Any]:
    sim, cfg, atlas, path = build_courtyard_sim(device, num_worlds=1)
    actor.eval()
    attach_baked_condition(actor)
    actor.gru.flatten_parameters()
    sim.reset_all(0)
    sim.apply_styles([deployment_style("centered_high_collision")])
    pose = pin_slot(sim, path, SPAWN_S, ey=SPAWN_EY, vx=0.0)
    sim.rebuild_sensors()
    hidden = actor.initial_hidden(1, device)
    hz = float(cfg.agents.control_hz)
    horizon = int(round(float(seconds) * hz))
    t = sim.buffers.torch_arrays
    poses: list[np.ndarray] = []
    crashed = False
    steps = 0
    for step in range(horizon):
        obs = sim.action_observation().to(device)
        out = actor.forward(obs, None, hidden, deterministic=True)
        hidden = out.hidden
        sim.step(out.actions.detach())
        steps += 1
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
        if not bool((t.active[0] > 0).item()):
            crashed = True
            break
    progress = float(t.progress_s[0].item())
    survival = steps / hz
    laps = progress / max(path.length, 1e-6)
    return {
        "spawn": pose,
        "track_length_m": path.length,
        "steps": steps,
        "survival_s": survival,
        "progress_m": progress,
        "laps": laps,
        "full_lap": bool(progress >= LAP_TARGET_M or laps >= 1.0),
        "crashed": crashed,
        "final_s": float(t.frenet_s[0].item()),
        "final_ey": float(t.frenet_ey[0].item()),
        "final_vx": float(t.vx[0].item()),
        "min_clearance": float(t.boundary_distance[0].item()),
        "poses": poses,
        "atlas": atlas,
        "path": path,
        "cfg": cfg,
    }


def _banner(frame: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, frame.shape[1], 3), 20, dtype=np.uint8)
    out = np.concatenate([bar, frame], axis=0)
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(out)
        ImageDraw.Draw(img).text((8, 6), text, fill=(255, 255, 255))
        return np.asarray(img)
    except Exception:
        return out


def render_side_by_side(
    atlas,
    left_poses: Sequence[np.ndarray],
    right_poses: Sequence[np.ndarray],
    output: Path,
    *,
    car_length: float,
    car_width: float,
    fps: int = 10,
) -> Path:
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(left_poses), len(right_poses))
    last_l = left_poses[-1] if left_poses else np.zeros((1, 4), dtype=np.float32)
    last_r = right_poses[-1] if right_poses else np.zeros((1, 4), dtype=np.float32)
    with imageio.get_writer(output, fps=fps, codec="libx264", macro_block_size=1) as mp4:
        for i in range(n):
            lp = left_poses[i] if i < len(left_poses) else last_l
            rp = right_poses[i] if i < len(right_poses) else last_r
            left = render_frame(
                atlas,
                0,
                lp,
                car_length=car_length,
                car_width=car_width,
                follow_radius=max(FOLLOW_RADIUS_M, 6.0),
            )
            right = render_frame(
                atlas,
                0,
                rp,
                car_length=car_length,
                car_width=car_width,
                follow_radius=max(FOLLOW_RADIUS_M, 6.0),
            )
            left = _banner(left, "Left: Base actor_005700")
            right = _banner(right, "Right: Courtyard Fine-Tuned")
            h = min(left.shape[0], right.shape[0])
            w = min(left.shape[1], right.shape[1])
            mp4.append_data(np.concatenate([left[:h, :w], right[:h, :w]], axis=1))
    return output


def run_campaign(
    *,
    checkpoint: Path | str = CKPT_BASE,
    output_dir: Path | str | None = None,
    device: str | None = None,
    eval_seconds: float = 90.0,
    skip_train: bool = False,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir) if output_dir else Path("/tmp/courtyard-multilap")
    output.mkdir(parents=True, exist_ok=True)
    actor0, payload = load_offline_actor(checkpoint, device=device)
    attach_baked_condition(actor0)
    actor0.gru.flatten_parameters()
    print(json.dumps({"device": device, "phase": "diagnose"}), flush=True)
    diag = diagnose_actor(copy.deepcopy(actor0), device=device, seconds=8.0)
    diag_rows = diag.pop("rows")
    (output / "diagnose.json").write_text(
        json.dumps({**diag, "n_rows": len(diag_rows)}, indent=2)
    )
    (output / "diagnose_rows.json").write_text(json.dumps(diag_rows))
    print(json.dumps({"diagnose": {k: diag[k] for k in diag if k != "spawn"}}), flush=True)

    evals: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    drop_keys = {"poses", "atlas", "path", "cfg"}
    if not skip_train:
        s2_actor = copy.deepcopy(actor0)
        s2_base = copy.deepcopy(actor0)
        s2 = run_s2(s2_actor, s2_base, payload, device=device, output=output)
        artifacts["s2"] = s2["artifact"]
        ev = evaluate_from_rest(s2_actor, device=device, seconds=eval_seconds)
        evals["s2"] = {k: ev[k] for k in ev if k not in drop_keys}
        print(json.dumps({"eval_s2": evals["s2"]}), flush=True)
        candidates = {"s2": s2_actor}
    else:
        candidates = {"base": actor0}

    base_eval = evaluate_from_rest(
        actor0, device=device, seconds=eval_seconds, collect_poses=True
    )
    evals["base"] = {k: base_eval[k] for k in base_eval if k not in drop_keys}
    winner_name = "base"
    winner_score = -1.0
    for name, stats in evals.items():
        score = float(stats.get("progress_m", 0.0))
        if score > winner_score:
            winner_score = score
            winner_name = name
    if winner_name == "base":
        winner_actor = actor0
    else:
        winner_actor = candidates[winner_name]
    win_eval = evaluate_from_rest(
        winner_actor, device=device, seconds=eval_seconds, collect_poses=True
    )
    evals["winner_replay"] = {k: win_eval[k] for k in win_eval if k not in drop_keys}
    video = output / "side_by_side.mp4"
    cfg = win_eval["cfg"]
    try:
        render_side_by_side(
            win_eval["atlas"],
            base_eval["poses"],
            win_eval["poses"],
            video,
            car_length=cfg.agents.car_length_m,
            car_width=cfg.agents.car_width_m,
        )
    except Exception as exc:
        video = Path(str(video) + ".failed")
        print(json.dumps({"render_error": str(exc)}), flush=True)
    win_path = output / "winner_actor.pt"
    _write_actor(win_path, payload, winner_actor)
    result = {
        "device": device,
        "diagnose": {k: diag[k] for k in diag},
        "evals": evals,
        "artifacts": artifacts,
        "winner": winner_name,
        "winner_checkpoint": str(win_path),
        "video": str(video),
        "laps": evals["winner_replay"]["laps"],
        "progress_m": evals["winner_replay"]["progress_m"],
        "survival_s": evals["winner_replay"]["survival_s"],
    }
    (output / "campaign.json").write_text(json.dumps(result, indent=2, default=str))
    printable = {k: result[k] for k in result if k != "diagnose"}
    print(json.dumps(printable, indent=2, default=str), flush=True)
    return result

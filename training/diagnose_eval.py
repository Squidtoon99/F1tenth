"""Diagnostic eval: roll out a checkpoint over many envs and record *where* and
*why* episodes end, plus how far each episode gets (in laps).

Writes a summary to stdout and a diagnostic PNG (track outline + termination
scatter colored by reason, plus histograms of lap-progress and oob track-location).

    python diagnose_eval.py --checkpoint outputs/runs/<id>/checkpoints/ckpt_390000.pt \
        --num-envs 200 --steps 4000 --out /tmp/diag.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import torch

from f1tenth_env import runtime as rt
from f1tenth_env.env import F1tenthEnv
from standalone_trainer import (
    DEFAULT_CONFIG,
    ObsNormalizer,
    build_models,
    select_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostic eval rollout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", type=str, default=None,
                   help="Run config.json (defaults to <ckpt>/../../config.json).")
    p.add_argument("--opponent", type=str, default="scripted",
                   choices=["scripted", "none"],
                   help="Opponent for eval; scripted keeps obs in-distribution "
                        "without needing an external policy.")
    p.add_argument("--num-envs", type=int, default=200)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--precision", type=str, default=None, choices=["32", "64"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out", type=str, default="/tmp/diag.png")
    p.add_argument("--npz", type=str, default="/tmp/diag.npz")
    return p.parse_args()


def _cum_arclen(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])[:-1], float(d.sum())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = args.config or str(
        Path(args.checkpoint).resolve().parent.parent / "config.json")
    if Path(cfg_path).exists():
        saved = json.loads(Path(cfg_path).read_text())
        cfg = saved["config"]
        precision = args.precision or str(saved.get("args", {}).get("precision", "32"))
        print(f"loaded run config: {cfg_path} (precision={precision})")
    else:
        cfg = DEFAULT_CONFIG
        precision = args.precision or "32"
        print(f"WARNING: no config.json at {cfg_path}; using DEFAULT_CONFIG")

    cfg["env"]["physics_backend"] = "torch"
    # Use a scripted centerline opponent for eval (no external policy needed) so the
    # 390-dim obs stay in-distribution; or drop the opponent entirely.
    if args.opponent == "scripted":
        cfg["env"]["opponent_strategy"] = "scripted"
    track = cfg["env"]["track"]

    device = select_device(args.device)
    rt.configure(
        float_dtype=torch.float64 if precision == "64" else torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )

    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": args.num_envs},
        **cfg["env"],
    }
    control_interval = int(cfg["env"]["control_interval"])
    clip_actions = float(cfg["env"]["clip_actions"])

    env = F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )

    models, _ = build_models(cfg, device)
    normalizer = ObsNormalizer(
        obs_dim=cfg["obs"]["num_obs"], device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    models.actor.load_state_dict(payload["actor"])
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    models.actor.eval()

    from f1tenth_env.utils import compute_track_boundaries
    centerline_full = np.asarray(env.track_state["centerline"], dtype=np.float32)
    centerline = centerline_full[:, :2].astype(np.float64)
    cl_s, track_len = _cum_arclen(centerline)
    left_b, right_b = compute_track_boundaries(
        centerline_full,
        np.asarray(env.track_state["w_tr_left"], dtype=np.float32),
        np.asarray(env.track_state["w_tr_right"], dtype=np.float32),
    )
    left = np.asarray(left_b, dtype=np.float64)[:, :2]
    right = np.asarray(right_b, dtype=np.float64)[:, :2]
    print(f"track='{track}' len={track_len:.1f} m  n_center={len(centerline)}  "
          f"has_opp={env.has_opponent}  num_envs={args.num_envs}  steps={args.steps}")

    N = args.num_envs
    cum_ds = np.zeros(N)          # arc length this episode
    max_frac = np.zeros(N)        # deepest lap-fraction reached this episode
    ep_started = np.zeros(N, dtype=int)

    events = []                   # (reason, x, y, s_frac_loc, laps_done, speed, ey)
    reason_counts = defaultdict(int)
    completed_laps = []           # laps completed at each episode END
    n_episodes = 0

    obs, _ = env.reset()
    obs = obs.to(torch.float32)

    def nearest_sfrac(xy: np.ndarray) -> np.ndarray:
        # xy: (M,2) -> nearest centerline arc-length fraction
        d2 = ((centerline[None, :, :] - xy[:, None, :]) ** 2).sum(-1)
        idx = d2.argmin(1)
        return cl_s[idx] / max(track_len, 1e-6)

    with torch.no_grad():
        for step in range(args.steps):
            st = env.backend.read_state()
            prev_xy = st["base_pos"][:, :2].cpu().numpy().copy()

            actions, _ = models.actor(
                normalizer.normalize(obs),
                deterministic=not args.stochastic, with_logprob=False,
            )
            actions = actions.clamp(-clip_actions, clip_actions)
            obs, _, done, extras = env.step(
                actions.to(rt.tc_float), n_steps=control_interval)
            obs = obs.to(torch.float32)

            m = extras["metrics"]
            ds = m["progress_ds"].detach().cpu().numpy()
            ey = m["lateral_error"].detach().cpu().numpy()
            spd = m["speed_xy"].detach().cpu().numpy()
            cum_ds += ds
            max_frac = np.maximum(max_frac, cum_ds / track_len)

            term = extras["termination"]
            done_np = done.detach().cpu().numpy().astype(bool)
            if done_np.any():
                sfrac = nearest_sfrac(prev_xy)
                # per-reason masks
                reason_masks = {
                    k: term[k].detach().cpu().numpy().astype(bool)
                    for k in term
                }
                for i in np.nonzero(done_np)[0]:
                    # pick the reason(s) that fired for this env
                    fired = [k for k, msk in reason_masks.items() if msk[i]]
                    reason = fired[0] if fired else "unknown"
                    reason_counts[reason] += 1
                    laps = cum_ds[i] / track_len
                    events.append((reason, float(prev_xy[i, 0]), float(prev_xy[i, 1]),
                                   float(sfrac[i]), float(laps), float(spd[i]),
                                   float(ey[i])))
                    completed_laps.append(laps)
                    n_episodes += 1
                cum_ds[done_np] = 0.0
                max_frac[done_np] = 0.0

    env.close()

    # ---- summary ----
    laps_arr = np.array(completed_laps) if completed_laps else np.zeros(1)
    print(f"\n=== {n_episodes} episodes ended ===")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} {v:5d}  ({100*v/max(n_episodes,1):5.1f}%)")
    print(f"\nlaps completed per episode: mean={laps_arr.mean():.3f} "
          f"median={np.median(laps_arr):.3f} max={laps_arr.max():.3f} "
          f"p90={np.percentile(laps_arr,90):.3f}")
    for thr in (0.25, 0.5, 0.9, 1.0):
        print(f"  frac episodes reaching >= {thr:.2f} lap: "
              f"{100*np.mean(laps_arr>=thr):.1f}%")

    # oob hotspots by track fraction
    oob = [e for e in events if e[0] == "out_of_bounds"]
    if oob:
        s_oob = np.array([e[3] for e in oob])
        hist, edges = np.histogram(s_oob, bins=20, range=(0, 1))
        print("\noob location histogram (track fraction 0..1, 20 bins):")
        top = np.argsort(hist)[::-1][:5]
        for b in sorted(top):
            print(f"  s=[{edges[b]:.2f},{edges[b+1]:.2f}) : {hist[b]:4d} oob "
                  f"({100*hist[b]/len(oob):.1f}%)")

    np.savez(args.npz,
             events=np.array([(e[1], e[2], e[3], e[4], e[5], e[6]) for e in events]),
             reasons=np.array([e[0] for e in events]),
             centerline=centerline, left=left, right=right)
    print(f"\nsaved raw -> {args.npz}")

    _plot(args.out, centerline, left, right, events, laps_arr, oob, track_len)
    print(f"saved plot -> {args.out}")


def _plot(out, centerline, left, right, events, laps_arr, oob, track_len):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "out_of_bounds": "#e63c3c", "collision": "#3c8cff",
        "not_moving": "#ffb028", "invalid_state": "#c05ade",
        "time_out": "#3cc86e", "unknown": "#888888",
    }
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(left[:, 0], left[:, 1], color="#5aa0ff", lw=1)
    ax.plot(right[:, 0], right[:, 1], color="#ff8c3c", lw=1)
    ax.plot(centerline[:, 0], centerline[:, 1], color="#888", lw=0.6, ls="--")
    # highlight the dominant oob band s=[0.30,0.40] and the start line s=0
    n = len(centerline)
    hot = centerline[int(0.30 * n):int(0.40 * n) + 1]
    ax.plot(hot[:, 0], hot[:, 1], color="yellow", lw=3, alpha=0.8,
            label="killer corner s=0.30-0.40")
    ax.scatter([centerline[0, 0]], [centerline[0, 1]], marker="*", s=200,
               c="white", edgecolors="k", zorder=5, label="start s=0")
    by_reason = defaultdict(list)
    for e in events:
        by_reason[e[0]].append((e[1], e[2]))
    for reason, pts in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5,
                   c=colors.get(reason, "#888"),
                   label=f"{reason} ({len(pts)})")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Where episodes end (termination locations)")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(laps_arr, bins=40, color="#3cc86e")
    ax2.axvline(1.0, color="r", ls="--", lw=1)
    ax2.set_title("Laps completed per episode")
    ax2.set_xlabel("laps")

    ax3 = fig.add_subplot(2, 2, 4)
    if oob:
        s_oob = np.array([e[3] for e in oob])
        ax3.hist(s_oob, bins=30, range=(0, 1), color="#e63c3c")
    ax3.set_title("Out-of-bounds location (track fraction)")
    ax3.set_xlabel("track fraction 0..1")

    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()

"""Scored attempts for the tree. Lap numbers are the published solo-flying evals."""

from __future__ import annotations

NS_NAME = "garage-seed"

BASE = "/home/ubuntu/projects/F1tenth-baseline-v5/outputs/runs"
NOTAL = (
    "/home/ubuntu/projects/F1tenth-opponents-notal/outputs/runs/"
    "opponents-notal-v6-B/checkpoints/policy_256000000.pt"
)

W = "https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs"

# Train budgets stated for these runs. Fastopp was stopped before a budget call.
BUDGETS = {
    "spatialcaps-v6-B": 512_000_000,
    "spatialcaps-v6-B-s43": 512_000_000,
    "spatialcaps-v2-B": 512_000_000,
    "opponents-notal-v6-B": 256_000_000,
    "opponents-notal-v6-B-s43": 256_000_000,
    "actionchunk-notal-v6-B": 256_000_000,
    "actionchunk-notal-v6-B-s43": 256_000_000,
}

# Published train-life windows. A chart snapshot, not a stored history.
EPISODE_MARKS = {
    "opponents-notal-v6-B": [
        {"step": 32_000_000, "seconds": 1.64},
        {"step": 40_000_000, "seconds": 1.97},
        {"step": 51_000_000, "seconds": 2.00},
        {"step": 56_000_000, "seconds": 2.47},
        {"step": 64_000_000, "seconds": 4.76},
        {"step": 66_000_000, "seconds": 8.30},
        {"step": 72_000_000, "seconds": 37.0},
        {"step": 80_000_000, "seconds": 49.0},
        {"step": 96_000_000, "seconds": 58.0},
        {"step": 112_000_000, "seconds": 77.0},
        {"step": 120_000_000, "seconds": 121.0},
        {"step": 128_000_000, "seconds": 61.0},
        {"step": 256_000_000, "seconds": 160.0},
    ],
    "spatialcaps-v6-B": [
        {"step": 128_000_000, "seconds": 220.0},
    ],
    "opponents-notal-fastopp-v6-B": [
        {"step": 64_000_000, "seconds": 2.10},
    ],
}


def _eval(lap, completion, crash, overtakes, steer, spawn):
    return {
        "scenario": "solo_flying",
        "flying_lap_s": lap,
        "completion": completion,
        "crash": crash,
        "overtakes": overtakes,
        "steer_reversal": steer,
        "spawn_mps": spawn,
    }


def _ckpt(step, disk, best, final, ev, nbytes=None, videos=None):
    return {
        "step": step,
        "disk_path": disk,
        "is_best": best,
        "is_final": final,
        "bytes": nbytes,
        "sha256": None,
        "s3_key": None,
        "evals": [ev] if ev else [],
        "videos": videos or [],
    }


def _video(kind, key, duration, width, height):
    return {
        "kind": kind,
        "s3_key": key,
        "duration_s": duration,
        "width": width,
        "height": height,
        "disk_path": None,
    }


def _run(wid, seed, plant, transitions, stop, ended, started=None):
    return {
        "wandb_run_id": wid,
        "wandb_url": f"{W}/{wid}",
        "seed": seed,
        "plant": plant,
        "transitions": transitions,
        "stop_reason": stop,
        "started_at": started,
        "ended_at": ended,
    }


def experiments():
    sc42 = f"{BASE}/spatialcaps-v6-B/checkpoints"
    sc43 = f"{BASE}/spatialcaps-v6-B-s43/checkpoints"
    ac42 = f"{BASE}/actionchunk-notal-v6-B/checkpoints"
    ac43 = f"{BASE}/actionchunk-notal-v6-B-s43/checkpoints"
    mib = round(18.32 * 1024 * 1024)
    return [
        {
            "slug": "spatialcaps-v6-B",
            "name": "Spatial CAPS s42",
            "parent": None,
            "kind": "ablation",
            "one_change": "caps_spatial_coef=0.1",
            "verdict": "keep",
            "created_at": "2026-10-16T17:45:00Z",
            "run": _run(
                "spatialcaps-v6-B", 42, "Austin v6", 512_229_376,
                "finished", "2026-10-16T17:45:00Z",
            ),
            "checkpoints": [
                _ckpt(
                    168_960_000, f"{sc42}/policy_168960000.pt", True, False,
                    _eval(117.5, 1.0, 0.0, 0.531, 0.093, 3.5),
                ),
                _ckpt(
                    512_229_376, f"{sc42}/policy_512229376.pt", False, True,
                    _eval(117.8, 1.0, 0.0, 0.609, 0.094, 3.5),
                ),
            ],
        },
        {
            "slug": "actionchunk-notal-v6-B",
            "name": "Action chunk no-TAL s42",
            "parent": "opponents-notal-v6-B",
            "kind": "ablation",
            "one_change": "action chunk H=4",
            "verdict": "keep",
            "created_at": "2026-10-17T05:10:00Z",
            "run": _run(
                "actionchunk-notal-v6-B", 42, "Austin v6", 256_008_192,
                "finished", "2026-10-17T05:10:00Z",
            ),
            "checkpoints": [
                _ckpt(
                    256_000_000, f"{ac42}/policy_256000000.pt", True, True,
                    _eval(55.8, 0.984, 0.188, 3.406, 0.200, 2.0),
                    nbytes=17_000_000,
                    videos=[_video(
                        "flying lap",
                        "dashboard/videos/actionchunk-notal-v6-B/flying-opponent.mp4",
                        56.8, 1120, 700,
                    )],
                ),
            ],
        },
        {
            "slug": "actionchunk-notal-v6-B-s43",
            "name": "Action chunk no-TAL s43",
            "parent": "actionchunk-notal-v6-B",
            "kind": "ablation",
            "one_change": "seed 43",
            "verdict": "revert",
            "created_at": "2026-10-17T09:07:00Z",
            "run": _run(
                "actionchunk-notal-v6-B-s43", 43, "Austin v6", 256_114_688,
                "finished", "2026-10-17T09:07:00Z",
            ),
            "checkpoints": [
                _ckpt(
                    256_000_000, f"{ac43}/policy_256000000.pt", True, True,
                    _eval(57.4, 0.656, 0.906, 1.531, 0.207, 2.0),
                    nbytes=17_000_000,
                    videos=[_video(
                        "crash clip",
                        "dashboard/videos/actionchunk-notal-v6-B-s43/flying-opponent.mp4",
                        24.7, 1120, 700,
                    )],
                ),
            ],
        },
        {
            "slug": "opponents-notal-v6-B",
            "name": "Opponents no-TAL s42",
            "parent": None,
            "kind": "ablation",
            "one_change": "TAL off, spawn 2.0 m/s",
            "verdict": "keep",
            "created_at": "2026-10-17T16:06:00Z",
            "run": _run(
                "opponents-notal-v6-B", 42, "Austin v6", 256_114_688,
                "finished", "2026-10-17T16:06:00Z",
            ),
            "checkpoints": [
                _ckpt(
                    256_000_000, NOTAL, True, True,
                    _eval(58.133, 0.984, 0.016, 3.656, 0.164, 2.0),
                ),
            ],
        },
        {
            "slug": "opponents-notal-v6-B-s43",
            "name": "Opponents no-TAL s43",
            "parent": "opponents-notal-v6-B",
            "kind": "ablation",
            "one_change": "seed 43",
            "verdict": "collapsed",
            "created_at": "2026-10-17T20:00:00Z",
            "run": _run(
                "opponents-notal-v6-B-s43", 43, "Austin v6", 126_976_000,
                "collapse", "2026-10-17T20:00:00Z",
            ),
            "checkpoints": [],
        },
        {
            "slug": "spatialcaps-v6-B-s43",
            "name": "Spatial CAPS s43",
            "parent": "spatialcaps-v6-B",
            "kind": "ablation",
            "one_change": "seed 43",
            "verdict": "keep",
            "created_at": "2026-10-17T21:00:00Z",
            "run": _run(
                "spatialcaps-v6-B-s43", 43, "Austin v6", 512_229_376,
                "finished", "2026-10-17T21:00:00Z",
            ),
            "checkpoints": [
                _ckpt(
                    271_360_000, f"{sc43}/policy_271360000.pt", True, False,
                    _eval(113.883, 1.0, 0.0, 0.562, 0.252, 3.5),
                ),
                _ckpt(
                    512_229_376, f"{sc43}/policy_512229376.pt", False, True,
                    _eval(117.498, 0.984, 0.016, 0.531, 0.234, 3.5),
                    nbytes=mib,
                    videos=[_video(
                        "flying lap",
                        "dashboard/videos/spatialcaps-v6-B-s43/flying-opponent.mp4",
                        125.0, None, None,
                    )],
                ),
            ],
        },
        {
            "slug": "opponents-notal-fastopp-v6-B",
            "name": "Fast opponents",
            "parent": "opponents-notal-v6-B",
            "kind": "ablation",
            "one_change": "opponents 6.5–7.6 m/s",
            "verdict": "stopped",
            "created_at": "2026-10-18T01:00:00Z",
            "run": _run(
                "opponents-notal-fastopp-v6-B", 42, "Austin v6", 91_000_000,
                "stopped", "2026-10-18T01:00:00Z",
            ),
            "checkpoints": [],
        },
        {
            "slug": "spatialcaps-v2-B",
            "name": "Spatial CAPS v2",
            "parent": "spatialcaps-v6-B",
            "kind": "baseline",
            "one_change": "SIM_READY_V2 plant",
            "verdict": "running",
            "created_at": "2026-10-18T02:00:00Z",
            "run": _run(
                "spatialcaps-v2-B", 42, "SIM_READY_V2", None, None, None,
            ),
            "checkpoints": [],
        },
    ]

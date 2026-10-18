"""Research-attempt catalog from the store TSV, later scored arms, and live ids.

Display names never use the old fixed-speed-reward acronym or a no- prefix.
Wandb run ids are left unchanged.
"""

from __future__ import annotations

import csv
import os
import re
import uuid
from pathlib import Path

NS = uuid.UUID("a3f1c2d4-5b6e-4789-8abc-def012345678")
NS_NAME = "garage-seed"
W = "https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs"

DEFAULT_STORE = Path(
    "/run/user/1000/cursor_agent_stores/"
    "bc-6a7cf7f6-0f92-4e89-b654-cc9668017dde/files"
)
HERE = Path(__file__).resolve().parent

PASS = {"worked", "keep", "pass"}
FAIL = {"didn't", "didnt", "revert", "reject", "collapsed", "fail"}
HOLD = {"on-hold", "inconclusive", "hold"}

PARENTS = {
    "entropy-ab-austin-A": None,
    "entropy-ab-austin-B": "entropy-ab-austin-A",
    "entropy-ab-austin-C": "entropy-ab-austin-A",
    "entropy-ab-austin-D": "entropy-ab-austin-A",
    "baseline-v5-s42": "entropy-ab-austin-C",
    "baseline-v5-s43": "baseline-v5-s42",
    "critic-v5-B": "baseline-v5-s42",
    "critic-v5-B-s43": "critic-v5-B",
    "dr-v5-s42": "baseline-v5-s42",
    "dr-v5-s43": "dr-v5-s42",
    "critic-v5-C": "baseline-v5-s42",
    "critic-v5-C-s43": "critic-v5-C",
    "critic-v5-D": "baseline-v5-s42",
    "critic-v5-D-s43": "critic-v5-D",
    "tal-v5-B": "baseline-v5-s42",
    "tal-v5-B-s43": "tal-v5-B",
    "stdmax-v6-B": "tal-v5-B",
    "stdmax-v6-B-s43": "stdmax-v6-B",
    "lr-linear-v6-B": "tal-v5-B",
    "lr-linear-v6-B-s43": "lr-linear-v6-B",
    "reversal-v6-B": "tal-v5-B",
    "reversal-v6-B-s43": "reversal-v6-B",
    "caps-v6-B": "tal-v5-B",
    "alpha-rpo-v5-B": "baseline-v5-s42",
    "tal-dr-v6-B": "tal-v5-B",
    "spawncap-v6-B": "tal-v5-B",
    "spawncap-v6-B-s43": "spawncap-v6-B",
    "wallcost-v6-B": "tal-v5-B",
    "opponents-v6-B": "tal-v5-B",
}

CHANGES = {
    "entropy-ab-austin-A": "entropy_coef=0.01",
    "entropy-ab-austin-B": "entropy_coef=0",
    "entropy-ab-austin-C": "anneal entropy 0.01 → 0 over 32M",
    "entropy-ab-austin-D": "entropy_from_squashed=true",
    "baseline-v5-s42": "obs v5 + anneal C",
    "baseline-v5-s43": "seed 43",
    "critic-v5-B": "value_clip=0",
    "critic-v5-B-s43": "seed 43",
    "dr-v5-s42": "dynamics DR",
    "dr-v5-s43": "seed 43",
    "critic-v5-C": "PopArt value normalization",
    "critic-v5-C-s43": "seed 43",
    "critic-v5-D": "value_max_grad_norm=10",
    "critic-v5-D-s43": "seed 43",
    "tal-v5-B": "fixed-speed reward scale 0.2",
    "tal-v5-B-s43": "seed 43",
    "stdmax-v6-B": "log_std_max=0",
    "stdmax-v6-B-s43": "seed 43",
    "lr-linear-v6-B": "lr_schedule=linear",
    "lr-linear-v6-B-s43": "seed 43",
    "reversal-v6-B": "reversal_coef_throttle=0.5",
    "reversal-v6-B-s43": "seed 43",
    "caps-v6-B": "caps_coef=0.1",
    "alpha-rpo-v5-B": "residual_base_enabled vs v5",
    "tal-dr-v6-B": "fixed-speed reward + dynamics DR",
    "spawncap-v6-B": "reset_speed_max 7→4",
    "spawncap-v6-B-s43": "seed 43",
    "wallcost-v6-B": "continuous_quadratic wall cost",
    "opponents-v6-B": "scripted opponents",
}

TITLES = {
    "entropy-ab-austin-A": "Entropy A",
    "entropy-ab-austin-B": "Entropy B",
    "entropy-ab-austin-C": "Entropy C",
    "entropy-ab-austin-D": "Entropy D",
    "baseline-v5-s42": "Baseline v5",
    "baseline-v5-s43": "Baseline v5",
    "critic-v5-B": "Value clip off",
    "critic-v5-B-s43": "Value clip off",
    "dr-v5-s42": "Dynamics DR",
    "dr-v5-s43": "Dynamics DR",
    "critic-v5-C": "PopArt",
    "critic-v5-C-s43": "PopArt",
    "critic-v5-D": "Value grad clip",
    "critic-v5-D-s43": "Value grad clip",
    "tal-v5-B": "Fixed-speed reward",
    "tal-v5-B-s43": "Fixed-speed reward",
    "stdmax-v6-B": "Std max cap",
    "stdmax-v6-B-s43": "Std max cap",
    "lr-linear-v6-B": "Linear LR",
    "lr-linear-v6-B-s43": "Linear LR",
    "reversal-v6-B": "Throttle reversal",
    "reversal-v6-B-s43": "Throttle reversal",
    "caps-v6-B": "CAPS",
    "alpha-rpo-v5-B": "α-RPO",
    "tal-dr-v6-B": "Fixed-speed + DR",
    "spawncap-v6-B": "Spawn cap",
    "spawncap-v6-B-s43": "Spawn cap",
    "wallcost-v6-B": "Wall cost",
    "opponents-v6-B": "Scripted opponents",
}

SENSOR_IO = {
    "kind": "sensor PPO",
    "inputs": [
        {"name": "LiDAR ranges", "size": 1081, "note": "270° FOV, 0.06–30 m"},
        {"name": "IMU accel", "size": 3, "note": "ax, ay, az"},
        {"name": "IMU gyro", "size": 3, "note": "gx, gy, yaw rate"},
        {"name": "VESC speed", "size": 1, "note": "body-forward m/s"},
        {"name": "VESC current echo", "size": 1, "note": "slewed command"},
        {"name": "Throttle history", "size": 2, "note": "current and previous"},
        {"name": "Steer history", "size": 3, "note": "t, t-1, t-2"},
        {"name": "Steer deltas", "size": 3, "note": "three consecutive diffs"},
    ],
    "outputs": [
        {"name": "Steer", "size": 1, "note": "delta command"},
        {"name": "Throttle", "size": 1, "note": "longitudinal current"},
    ],
    "actor_dim": 1097,
    "action_dim": 2,
}

CHUNK_IO = {
    **SENSOR_IO,
    "kind": "sensor PPO, action chunk H=4",
    "outputs": [
        {"name": "Steer chunk", "size": 4, "note": "4-step steer plan"},
        {"name": "Throttle chunk", "size": 4, "note": "4-step throttle plan"},
    ],
    "action_dim": 8,
}

GIGA_IO = {
    "kind": "Gigaflow actor",
    "inputs": [
        {"name": "Scan / occupancy", "size": None, "note": "Gigaflow gym obs"},
        {"name": "Ego state", "size": None, "note": "pose, speed, command echo"},
    ],
    "outputs": [
        {"name": "Steer", "size": 1, "note": "delta command"},
        {"name": "Throttle", "size": 1, "note": "signed current"},
    ],
    "actor_dim": None,
    "action_dim": 2,
}


def store_root() -> Path:
    raw = os.environ.get("GARAGE_STORE")
    if raw:
        return Path(raw)
    if DEFAULT_STORE.is_dir():
        return DEFAULT_STORE
    return HERE / "testdata"


def media_root() -> Path:
    return store_root() / "media"


def results_path() -> Path:
    candidate = store_root() / "docs" / "results.tsv"
    if candidate.is_file():
        return candidate
    return HERE / "testdata" / "results.tsv"


def experiment_id(slug: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:experiment:{slug}")


def run_id(wandb: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:run:{wandb}")


def video_id(key: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:video:{key}")


def eval_id(slug: str, scenario: str, step: int) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:catalog-eval:{slug}:{scenario}:{step}")


def public_text(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"\bno-TAL\b", "progress-reward", text, flags=re.I)
    out = re.sub(r"\bnotal\b", "progress-reward", out, flags=re.I)
    out = re.sub(r"\bTAL\b", "fixed-speed reward", out)
    out = re.sub(r"(?<![A-Za-z])tal(?![A-Za-z])", "fixed-speed reward", out)
    out = re.sub(r"\bno-(?=[A-Za-z])", "", out)
    return out


def result_of(verdict: str) -> str:
    key = (verdict or "").strip().lower()
    if key in PASS:
        return "pass"
    if key in FAIL:
        return "fail"
    if key in HOLD:
        return "hold"
    if key in {"running", "stopped"}:
        return key
    return key or "hold"


def _num(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"n/a", "nan", "none"}:
        return None
    return float(text)


def _int(value):
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def display_name(slug: str, seed=None) -> str:
    if "notal" in slug:
        if slug.startswith("actionchunk"):
            title = "Action chunk progress-reward"
        elif "fastopp" in slug:
            title = "Fast opponents progress-reward"
        else:
            title = "Opponents progress-reward"
    elif slug in TITLES:
        title = TITLES[slug]
    else:
        title = slug.replace("-", " ")
        title = public_text(title)
    if seed is not None:
        return f"{title} s{seed}"
    return title


def _clip(slug: str, kind: str, rel: str, duration=None, width=None, height=None):
    path = media_root() / rel
    if not path.is_file():
        return None
    key = f"catalog/{slug}/{path.name}"
    return {
        "id": video_id(key),
        "kind": kind,
        "s3_key": None,
        "disk_path": str(path),
        "duration_s": duration,
        "width": width,
        "height": height,
        "run_name": display_name(slug),
        "stream": f"/api/videos/{video_id(key)}/stream",
    }


def videos_for(slug: str) -> list[dict]:
    clips = {
        "actionchunk-notal-v6-B": [
            ("flying lap", "actionchunk-notal/flying-opponent.mp4", 56.8, 1120, 700),
        ],
        "actionchunk-notal-v6-B-s43": [
            ("eval", "actionchunk-notal-s43/flying-opponent.mp4", 24.7, 1120, 700),
        ],
        "actionchunk-v6-B": [
            ("flying lap", "actionchunk-v6-b/flying-opponent.mp4", None, None, None),
        ],
        "opponents-notal-v6-B": [
            ("flying lap", "opponents-notal/flying-opponent.mp4", None, None, None),
        ],
        "spatialcaps-v6-B": [
            ("flying lap", "spatialcaps-v6-b/flying-opponent-rand.mp4", None, None, None),
        ],
        "spatialcaps-v6-B-s43": [
            ("flying lap", "spatialcaps-v6-b-s43/flying-opponent.mp4", 125.0, None, None),
        ],
        "tinylidarnet-v6-B": [
            ("crash clip", "tinylidarnet-v6-b/flying-crash.mp4", None, None, None),
        ],
    }
    found = []
    for item in clips.get(slug, []):
        video = _clip(slug, *item)
        if video:
            found.append(video)
    return found


def model_io(slug: str) -> dict:
    if "gigaflow" in slug:
        return GIGA_IO
    if "actionchunk" in slug:
        return CHUNK_IO
    return SENSOR_IO


def later_arms() -> list[dict]:
    return [
        _arm(
            "wallprox-v6-B", "opponents-v6-B", "wall_proximity_m=0.05",
            "fail", 42, "Austin v6", 128_000_000, None, 0.0, 1.0, 0.0,
            "collapsed", "Scripted opponents",
        ),
        _arm(
            "gradcaps-v6-B", "opponents-v6-B", "grad_caps_coef=0.1",
            "fail", 42, "Austin v6", 128_500_000, None, 0.019, 1.0, 0.0,
            "collapsed", "Scripted opponents",
        ),
        _arm(
            "spatialcaps-v6-B", "opponents-v6-B", "caps_spatial_coef=0.1",
            "pass", 42, "Austin v6", 512_229_376, 117.5, 1.0, 0.0, 0.531,
            "finished", "Spatial CAPS",
        ),
        _arm(
            "spatialcaps-v6-B-s43", "spatialcaps-v6-B", "seed 43",
            "pass", 43, "Austin v6", 512_229_376, 113.883, 1.0, 0.0, 0.562,
            "finished", "Spatial CAPS",
        ),
        _arm(
            "actionchunk-v6-B", "opponents-v6-B", "action chunk H=4",
            "pass", 42, "Austin v6", 256_000_000, 115.9, 0.984, 0.016, 0.625,
            "finished", "Action chunk",
        ),
        _arm(
            "tinylidarnet-v6-B", "opponents-v6-B", "TinyLidarNet GRU",
            "fail", 42, "Austin v6", 256_000_000, None, 0.0, 1.0, 0.375,
            "finished", "TinyLidarNet",
        ),
        _arm(
            "opponents-notal-v6-B", "opponents-v6-B",
            "progress reward, spawn 2.0 m/s",
            "pass", 42, "Austin v6", 256_114_688, 58.133, 0.984, 0.016, 3.656,
            "finished", "Opponents progress-reward",
        ),
        _arm(
            "opponents-notal-v6-B-s43", "opponents-notal-v6-B", "seed 43",
            "fail", 43, "Austin v6", 126_976_000, None, 0.019, 1.0, 0.0,
            "collapse", "Opponents progress-reward",
        ),
        _arm(
            "actionchunk-notal-v6-B", "opponents-notal-v6-B",
            "action chunk H=4",
            "pass", 42, "Austin v6", 256_008_192, 55.8, 0.984, 0.188, 3.406,
            "finished", "Action chunk progress-reward",
        ),
        _arm(
            "actionchunk-notal-v6-B-s43", "actionchunk-notal-v6-B", "seed 43",
            "fail", 43, "Austin v6", 256_114_688, 57.4, 0.656, 0.906, 1.531,
            "finished", "Action chunk progress-reward",
        ),
        _arm(
            "opponents-notal-fastopp-v6-B", "opponents-notal-v6-B",
            "opponents 6.5–7.6 m/s",
            "stopped", 42, "Austin v6", 91_000_000, None, None, None, None,
            "stopped", "Fast opponents progress-reward",
        ),
        _arm(
            "spatialcaps-v2-B", "spatialcaps-v6-B", "SIM_READY_V2 plant",
            "running", 42, "SIM_READY_V2", None, None, None, None, None,
            None, "Spatial CAPS v2",
        ),
        _arm(
            "simv2-progress-s42", "spatialcaps-v6-B",
            "SIM_READY_V2 + progress reward",
            "running", 42, "SIM_READY_V2", None, None, None, None, None,
            None, "Sim v2 progress-reward",
        ),
        _arm(
            "gigaflow-v2", None, "Gigaflow on SIM_READY_V2",
            "running", None, "SIM_READY_V2", None, None, None, None, None,
            None, "Gigaflow v2",
        ),
        _arm(
            "sophy-multi-v2-B", "spatialcaps-v6-B",
            "Sophy multi-scenario on v2",
            "running", 42, "SIM_READY_V2", None, None, None, None, None,
            None, "Sophy multi v2",
        ),
    ]


def _arm(
    slug, parent, change, verdict, seed, plant, transitions, lap,
    completion, crash, overtakes, stop, title,
):
    name = f"{title} s{seed}" if seed is not None else title
    return {
        "slug": slug,
        "wandb_run_id": slug,
        "parent_slug": parent,
        "one_change": public_text(change),
        "verdict": verdict,
        "seed": seed,
        "plant": plant,
        "transitions": transitions,
        "lap": lap,
        "completion": completion,
        "crash": crash,
        "overtakes": overtakes,
        "steer_reversal": None,
        "spawn_mps": 2.0 if "notal" in slug else 3.5,
        "stop_reason": stop,
        "name": name,
        "kind": "ablation",
        "budget": transitions if verdict != "running" else 512_000_000,
    }


def _from_tsv_row(row: dict) -> dict:
    slug = row["run_id"]
    seed = _int(row.get("seed"))
    lap = _num(row.get("flying_mean_lap_s"))
    completion = _num(row.get("flying_full_lap"))
    crash = _num(row.get("flying_crash"))
    outcome = (row.get("outcome") or "").strip()
    return {
        "slug": slug,
        "wandb_run_id": slug,
        "parent_slug": PARENTS.get(slug),
        "one_change": CHANGES.get(slug, public_text(row.get("notes") or "")),
        "verdict": outcome or "hold",
        "seed": seed,
        "plant": f"{row.get('track') or 'Austin'} {row.get('obs_version') or ''}".strip(),
        "transitions": _int(row.get("transitions")),
        "lap": lap,
        "completion": completion,
        "crash": crash,
        "overtakes": None,
        "steer_reversal": None,
        "spawn_mps": 3.5,
        "stop_reason": "finished",
        "name": display_name(slug, seed),
        "kind": row.get("round") or "ablation",
        "budget": _int(row.get("transitions")),
        "best_ckpt": row.get("best_ckpt"),
        "final_ckpt": row.get("final_ckpt"),
        "final_completion": _num(row.get("final_flying_full_lap")),
        "last3_mean": _num(row.get("last3_mean")),
        "notes": public_text(row.get("notes") or ""),
    }


def load_tsv() -> list[dict]:
    path = results_path()
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [_from_tsv_row(row) for row in csv.DictReader(handle, delimiter="\t")]


def attempts() -> list[dict]:
    by_slug = {row["slug"]: row for row in load_tsv()}
    for row in later_arms():
        by_slug[row["slug"]] = row
    rows = []
    for row in by_slug.values():
        rows.append(_finalize(row))
    return _ordered(rows)


def _finalize(row: dict) -> dict:
    slug = row["slug"]
    parent = row.get("parent_slug")
    videos = videos_for(slug)
    item = {
        **row,
        "id": experiment_id(slug),
        "run_id": run_id(row["wandb_run_id"]),
        "parent_id": experiment_id(parent) if parent else None,
        "name": public_text(row["name"]),
        "one_change": public_text(row.get("one_change") or ""),
        "result": result_of(row.get("verdict") or ""),
        "wandb_url": f"{W}/{row['wandb_run_id']}",
        "video": videos[0] if videos else None,
        "videos": videos,
        "model_io": model_io(slug),
    }
    return item


def _ordered(rows: list[dict]) -> list[dict]:
    by_slug = {row["slug"]: row for row in rows}
    ordered = []
    seen = set()

    def visit(row):
        if row["slug"] in seen:
            return
        parent = row.get("parent_slug")
        if parent and parent in by_slug:
            visit(by_slug[parent])
        seen.add(row["slug"])
        ordered.append(row)

    for row in rows:
        visit(row)
    return ordered


def attempt_map() -> dict[str, dict]:
    return {row["slug"]: row for row in attempts()}


def find_attempt(token: str) -> dict | None:
    for row in attempts():
        if token in {row["slug"], str(row["id"]), str(row["run_id"]), row["wandb_run_id"]}:
            return row
    return None


def ancestor_path(slug: str) -> list[dict]:
    by_slug = attempt_map()
    chain = []
    current = by_slug.get(slug)
    while current:
        chain.append({
            "slug": current["slug"],
            "name": current["name"],
            "one_change": current["one_change"],
            "result": current["result"],
            "verdict": current["verdict"],
            "lap": current["lap"],
            "run_id": current["run_id"],
        })
        parent = current.get("parent_slug")
        current = by_slug.get(parent) if parent else None
    chain.reverse()
    return chain


def evals_for(row: dict) -> list[dict]:
    if row.get("lap") is None and row.get("completion") is None:
        return []
    step = row.get("transitions") or 0
    return [{
        "id": eval_id(row["slug"], "solo_flying", step),
        "scenario": "solo_flying",
        "flying_lap_s": row.get("lap"),
        "completion": row.get("completion"),
        "crash": row.get("crash"),
        "overtakes": row.get("overtakes"),
        "steer_reversal": row.get("steer_reversal"),
        "spawn_mps": row.get("spawn_mps"),
        "step": step,
        "is_best": True,
        "is_final": True,
        "run_name": row["name"],
        "run_id": row["run_id"],
    }]


def all_evals() -> list[dict]:
    rows = []
    for attempt in attempts():
        rows.extend(evals_for(attempt))
    rows.sort(key=lambda item: (item["flying_lap_s"] is None, item["flying_lap_s"] or 0))
    return rows

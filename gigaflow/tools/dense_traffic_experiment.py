#!/usr/bin/env python3
"""Queue + CPU-validate dense-traffic max-agents experiment (8 vs 10 vs 12).

Never touches production defaults, live CUDA viewers, or remote H100 trainers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.dense_traffic import (
    VARIANT_AGENTS,
    h100_queue_plan,
    rebuild_capacity_variants,
    run_cpu_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def _cuda_viewer_busy() -> bool:
    """Best-effort: refuse local CUDA if a gigaflow viewer already holds a GPU."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["ps", "-eo", "pid,cmd"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        if "gigaflow view" in line and "--device" in line and "cuda" in line:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "dense_traffic_experiment",
    )
    p.add_argument(
        "--src-atlas-cache",
        type=Path,
        default=Path("~/.cache/gigaflow/tracks").expanduser(),
        help="Production atlas cache to copy (never modified in place)",
    )
    p.add_argument(
        "--rebuild-atlases",
        action="store_true",
        help="Rebuild capacity-aware atlas variants under output-dir/atlases",
    )
    p.add_argument(
        "--skip-validate",
        action="store_true",
        help="Only rebuild atlases / emit H100 queue (no CPU sim/PPO)",
    )
    p.add_argument("--sim-steps", type=int, default=40)
    p.add_argument("--learner-updates", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow-cuda",
        action="store_true",
        help="Unused; validation is always CPU. Kept to document the refusal.",
    )
    args = p.parse_args()

    if args.allow_cuda:
        print(
            json.dumps(
                {
                    "warning": "dense_traffic validation always uses CPU; "
                    "--allow-cuda is ignored to protect live viewers"
                }
            )
        )
    if _cuda_viewer_busy():
        print(
            json.dumps(
                {
                    "ok": True,
                    "cuda_viewer_busy": True,
                    "action": "cpu_only_validation",
                }
            )
        )

    # Force no CUDA device visibility for the validation process itself.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    configs = {
        n: load_config(ROOT / f"configs/experiments/dense_traffic_cpu_max{n}.yaml")
        for n in VARIANT_AGENTS
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    atlas_variants: dict[int, str] = {}
    home_cache = Path("~/.cache/gigaflow").expanduser()
    for n in VARIANT_AGENTS:
        candidate = home_cache / f"tracks_dense_max{n}"
        if (candidate / "manifest.json").is_file():
            atlas_variants[n] = str(candidate)

    if args.rebuild_atlases:
        src_cache = args.src_atlas_cache.expanduser()
        if not src_cache.exists():
            raise SystemExit(
                f"src atlas cache missing: {src_cache} "
                "(run gigaflow prepare-tracks once for production cache)"
            )
        base = configs[8]
        from gigaflow_f1tenth.tracks import rebuild_atlas_capacity

        # Experiment-local copies under output-dir (for the report).
        rebuilt = rebuild_capacity_variants(
            src_cache,
            out / "atlases",
            max_agents_list=VARIANT_AGENTS,
            car_width_m=base.agents.car_width_m,
            car_length_m=base.agents.car_length_m,
        )
        # Canonical paths referenced by dense_traffic_h100_max*.yaml.
        for n in VARIANT_AGENTS:
            dst = home_cache / f"tracks_dense_max{n}"
            rebuild_atlas_capacity(
                src_cache,
                dst,
                max_agents=n,
                car_width_m=base.agents.car_width_m,
                car_length_m=base.agents.car_length_m,
            )
            atlas_variants[n] = str(dst)
            # Keep output-dir copies as well.
            atlas_variants[n] = str(rebuilt[n])
        # Prefer canonical cache paths in the report.
        atlas_variants = {
            n: str(home_cache / f"tracks_dense_max{n}") for n in VARIANT_AGENTS
        }

    queue = h100_queue_plan(
        repo_gigaflow=ROOT,
        atlas_root=out / "atlases",
        run_root=out / "h100_queue",
    )
    (out / "h100_queue.json").write_text(
        json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.skip_validate:
        payload = {
            "ok": True,
            "validated": False,
            "atlas_variants": atlas_variants,
            "h100_queue": str(out / "h100_queue.json"),
            "recommendation": {
                "try_first": 10,
                "reason": (
                    "Try max_agents=10 first before 12 (smaller step, "
                    "safer VRAM/occlusion headroom)."
                ),
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = run_cpu_experiment(
        configs,
        output_dir=out,
        src_atlas_cache=args.src_atlas_cache if args.rebuild_atlases else None,
        rebuild_atlases=False,  # already rebuilt above when requested
        sim_steps=int(args.sim_steps),
        learner_updates=int(args.learner_updates),
        seed=int(args.seed),
    )
    # Attach atlas paths if rebuilt separately.
    if atlas_variants:
        summary_path = Path(result["report"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["atlas_variants"] = {str(k): v for k, v in atlas_variants.items()}
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    payload = {
        "ok": True,
        "validated": True,
        "report": result["report"],
        "recommendation": result["recommendation"],
        "gates": {str(k): v for k, v in result["gates"].items()},
        "h100_queue": str(out / "h100_queue.json"),
        "atlas_variants": atlas_variants,
        "metrics_brief": {
            str(k): {
                "cars_dense": v.realized_cars_per_dense_world_mean,
                "close_pair_rate": v.close_pair_rate,
                "spawn_reject_rate": v.spawn_reject_rate,
                "occlusion": v.lidar_occlusion_proxy,
                "visibility": v.track_visibility_proxy,
                "collision_per_km": v.collision_per_km,
                "oob_per_km": v.oob_per_km,
                "sim_tps": v.sim_world_ticks_per_s,
                "vram_est_gib": v.vram_est_gib,
                "ppo_finite": v.ppo_finite,
                "solo_frac": v.solo_world_frac,
                "h2h_cars": v.head_to_head_cars,
            }
            for k, v in result["results"].items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

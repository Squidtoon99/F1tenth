"""CLI entry points for config, tracks, train, eval, soak, view, and benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gigaflow_f1tenth.config import (
    PINNED_UPSTREAM_TRACK_COUNT,
    config_to_dict,
    estimate_memory_bytes,
    load_config,
    replace_wandb_config,
)
from gigaflow_f1tenth.evaluation import (
    build_evaluator,
    load_actor_from_checkpoint,
    run_evaluation,
    run_soak,
)
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import (
    DEFAULT_EDT_RESOLUTION_M,
    DEFAULT_LUT_RESOLUTION_M,
    prepare_tracks,
)
from gigaflow_f1tenth.trainer import build_trainer, run_training
from gigaflow_f1tenth.wandb_log import build_wandb_session


def _apply_wandb_cli_overrides(cfg, args: argparse.Namespace):
    tags = None
    if getattr(args, "wandb_tags", None):
        tags = tuple(args.wandb_tags)
    return replace_wandb_config(
        cfg,
        enabled=getattr(args, "wandb", None),
        mode=getattr(args, "wandb_mode", None),
        entity=getattr(args, "wandb_entity", None),
        project=getattr(args, "wandb_project", None),
        group=getattr(args, "wandb_group", None),
        name=getattr(args, "wandb_name", None),
        tags=tags,
        notes=getattr(args, "wandb_notes", None),
        run_id=getattr(args, "wandb_run_id", None),
        resume=getattr(args, "wandb_resume", None),
    )


def _add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable W&B (overrides config.wandb.enabled; default: config).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="W&B mode override (online requires local credentials; never stores keys).",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tag",
        dest="wandb_tags",
        action="append",
        default=None,
        help="W&B tag (repeatable)",
    )
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        choices=("allow", "must", "never", "auto"),
        default=None,
        help="W&B resume policy for an existing run id",
    )


def _cmd_validate_config(args: argparse.Namespace) -> int:
    cfg = _apply_wandb_cli_overrides(load_config(args.config), args)
    payload = {
        "ok": True,
        "config_version": cfg.config_version,
        "estimate_memory_bytes": estimate_memory_bytes(cfg),
        "budget_bytes": cfg.profiling.estimate_bytes_budget,
        "num_slots": cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world,
        "pinned_upstream_track_count": PINNED_UPSTREAM_TRACK_COUNT,
        "config_num_tracks": cfg.tracks.num_tracks,
        "wandb_enabled": cfg.wandb.enabled,
        "wandb_mode": cfg.wandb.mode,
    }
    if args.dump:
        payload["config"] = config_to_dict(cfg)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_prepare_tracks(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    atlas = prepare_tracks(
        cfg,
        str(args.cache_dir),
        pin_path=args.pin,
        lut_resolution=args.lut_resolution,
        edt_resolution=args.edt_resolution,
        skip_download=args.skip_download,
    )
    man = atlas.manifest()
    payload = {
        "ok": True,
        "cache_dir": str(args.cache_dir),
        "num_tracks": len(man),
        "config_num_tracks": cfg.tracks.num_tracks,
        "pinned_upstream_track_count": PINNED_UPSTREAM_TRACK_COUNT,
        "track_names": [e.name for e in man],
        "manifest_path": str(Path(args.cache_dir) / "manifest.json"),
        "total_length_m": sum(e.length_m for e in man),
        "capacities": [e.capacity for e in man],
    }
    if len(man) != cfg.tracks.num_tracks and not args.skip_download:
        payload["warning"] = (
            f"atlas has {len(man)} tracks but config.tracks.num_tracks="
            f"{cfg.tracks.num_tracks}"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = _apply_wandb_cli_overrides(load_config(args.config), args)
    num_updates = int(args.num_updates or cfg.ppo.total_updates)
    if args.smoke:
        num_updates = min(num_updates, 2)
    run_dir = args.run_dir
    progress = run_training(
        cfg,
        num_updates,
        device=args.device,
        run_dir=run_dir,
        checkpoint_interval=int(args.checkpoint_interval),
        resume_from=args.resume_from,
    )
    payload = {
        "ok": True,
        "update_index": progress.update_index,
        "transitions": progress.transitions,
        "metrics": progress.metrics,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "wandb_enabled": cfg.wandb.enabled and cfg.wandb.mode != "disabled",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _apply_wandb_cli_overrides(load_config(args.config), args)
    actor = None
    if args.checkpoint is not None:
        actor = load_actor_from_checkpoint(
            cfg, args.checkpoint, device=args.device or "cpu"
        )
    suites = tuple(args.suite) if args.suite else None
    reports = run_evaluation(
        cfg,
        suites=suites,
        actor=actor,
        device=args.device,
        output_dir=args.output_dir,
    )
    evaluator = build_evaluator(cfg, actor=actor, device=args.device)
    gates = evaluator.promotion_gates(reports)
    session = build_wandb_session(cfg, run_dir=args.output_dir)
    try:
        session.start()
        media = []
        if args.output_dir is not None:
            out = Path(args.output_dir)
            media = sorted(
                list(out.glob("*.png"))
                + list(out.glob("*.gif"))
                + list(out.glob("*.mp4"))
                + list(out.glob("*.webm"))
            )
        session.log_evaluation(
            reports,
            source_step=0,
            global_step=0,
            media_paths=media,
        )
    finally:
        session.finish()
    payload = {
        "ok": all(gates.values()),
        "num_reports": len(reports),
        "promotion_gates": gates,
        "wandb_run_id": session.run_id,
        "reports": [
            {
                "suite": r.suite,
                "seed": r.seed,
                "metrics": r.metrics.__dict__,
                "extras": r.extras,
            }
            for r in reports
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def _cmd_soak(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = run_soak(
        cfg,
        steps=args.steps,
        device=args.device,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _cmd_view(args: argparse.Namespace) -> int:
    from gigaflow_f1tenth.viewer.replay import ViewerError, ViewerLaunchArgs
    from gigaflow_f1tenth.viewer.server import run_viewer

    launch = ViewerLaunchArgs(
        checkpoint=Path(args.checkpoint),
        config=Path(args.config),
        cache_dir=Path(args.cache_dir),
        suite=str(args.suite),
        seed=int(args.seed),
        device=str(args.device),
        track=args.track,
        track_id=args.track_id,
        host=str(args.host),
        ws_port=int(args.ws_port),
        http_port=int(args.http_port),
        open_browser=not bool(args.no_browser),
        policy_update_file=args.policy_update_file,
        checkpoint_roots=tuple(args.checkpoint_root or ()),
    )
    try:
        return run_viewer(launch)
    except ViewerError as exc:
        print(f"error: {exc}", flush=True)
        return 2


def _cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    device = args.device or (
        "cpu"
        if not torch.cuda.is_available()
        else cfg.worlds.device
    )
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    if args.mode == "sim":
        sim = build_simulator(cfg, atlas, device)
        n = world_slot_layout(cfg).num_slots
        actions = torch.zeros((n, 2), dtype=torch.float32)
        actions[:, 0] = 0.15
        sim.step(actions)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            sim.step(actions)
        elapsed = time.perf_counter() - t0
        active = int(sim.state().active.sum().item())
        payload = {
            "mode": "sim",
            "device": device,
            "slots": n,
            "active": active,
            "steps": args.steps,
            "elapsed_s": elapsed,
            "world_ticks_per_s": args.steps / max(elapsed, 1e-9),
            "agent_transitions_per_s": (active * args.steps) / max(elapsed, 1e-9),
        }
    else:
        trainer = build_trainer(cfg, atlas=atlas, device=device)
        trainer.setup()
        # Warmup one update (includes compile / first kernels).
        trainer.train_update()
        t0 = time.perf_counter()
        for _ in range(max(1, args.steps)):
            progress = trainer.train_update()
        elapsed = time.perf_counter() - t0
        payload = {
            "mode": "learner",
            "device": device,
            "updates": max(1, args.steps),
            "elapsed_s": elapsed,
            "updates_per_s": max(1, args.steps) / max(elapsed, 1e-9),
            "transitions_per_s": progress.metrics.get("transitions_per_s", 0.0),
            "profile": trainer.profile,
            "last_metrics": progress.metrics,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gigaflow",
        description="Isolated Gigaflow F1TENTH self-play tooling",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser(
        "validate-config",
        help="Load and validate an experiment YAML",
    )
    p_val.add_argument("--config", type=Path, required=True)
    p_val.add_argument(
        "--dump",
        action="store_true",
        help="Include resolved config JSON",
    )
    _add_wandb_args(p_val)
    p_val.set_defaults(func=_cmd_validate_config)

    p_tracks = sub.add_parser(
        "prepare-tracks",
        help="Download, checksum, validate, and pack the track atlas",
    )
    p_tracks.add_argument("--cache-dir", type=Path, required=True)
    p_tracks.add_argument("--config", type=Path, required=True)
    p_tracks.add_argument(
        "--pin",
        type=Path,
        default=None,
        help="Optional track_pin.json override",
    )
    p_tracks.add_argument(
        "--lut-resolution",
        type=float,
        default=DEFAULT_LUT_RESOLUTION_M,
    )
    p_tracks.add_argument(
        "--edt-resolution",
        type=float,
        default=DEFAULT_EDT_RESOLUTION_M,
    )
    p_tracks.add_argument(
        "--skip-download",
        action="store_true",
        help="Use checksummed cache / local fixtures only",
    )
    p_tracks.set_defaults(func=_cmd_prepare_tracks)

    p_train = sub.add_parser("train", help="Launch self-play training updates")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--num-updates", type=int, default=None)
    p_train.add_argument("--run-dir", type=Path, default=None)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--checkpoint-interval", type=int, default=0)
    p_train.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Exact-resume checkpoint path (restores PPO/sim and W&B run id)",
    )
    p_train.add_argument(
        "--smoke",
        action="store_true",
        help="Cap updates for a short integration smoke (not production)",
    )
    _add_wandb_args(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Run fixed-seed evaluation suites")
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--device", type=str, default=None)
    p_eval.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite name (repeatable); defaults to config.evaluation.suite",
    )
    _add_wandb_args(p_eval)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_soak = sub.add_parser("soak", help="Random/policy-action soak / anomaly check")
    p_soak.add_argument("--config", type=Path, required=True)
    p_soak.add_argument("--steps", type=int, default=None)
    p_soak.add_argument("--device", type=str, default=None)
    p_soak.add_argument("--output-dir", type=Path, default=None)
    p_soak.add_argument(
        "--mode",
        choices=("random", "policy"),
        default="random",
        help="Action source for the soak loop",
    )
    p_soak.set_defaults(func=_cmd_soak)

    p_bench = sub.add_parser("benchmark", help="Throughput profiling")
    p_bench.add_argument("--config", type=Path, required=True)
    p_bench.add_argument(
        "--mode",
        choices=("sim", "learner"),
        default="sim",
    )
    p_bench.add_argument("--steps", type=int, default=20)
    p_bench.add_argument("--device", type=str, default=None)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_view = sub.add_parser(
        "view",
        help="Live local WebGL checkpoint replay (does not touch training)",
    )
    p_view.add_argument("--checkpoint", type=Path, required=True)
    p_view.add_argument("--config", type=Path, required=True)
    p_view.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Prepared track atlas directory (atlas.npz + manifest.json)",
    )
    p_view.add_argument(
        "--suite",
        choices=("solo", "head_to_head", "dense"),
        default="solo",
    )
    p_view.add_argument("--seed", type=int, default=0)
    p_view.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or local cuda / cuda:N",
    )
    p_view.add_argument(
        "--track",
        type=str,
        default=None,
        help="Track name from the atlas (e.g. Austin)",
    )
    p_view.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Track index in the atlas (mutually exclusive with --track)",
    )
    p_view.add_argument("--host", type=str, default="127.0.0.1")
    p_view.add_argument("--ws-port", type=int, default=8765)
    p_view.add_argument("--http-port", type=int, default=8766)
    p_view.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory searched recursively for actor_*.pt offered in the UI "
            "checkpoint selector (repeatable; defaults to gigaflow/outputs)"
        ),
    )
    p_view.add_argument(
        "--policy-update-file",
        type=Path,
        default=None,
        help="Atomic local checkpoint hot-reload request file",
    )
    p_view.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab for the local UI",
    )
    p_view.set_defaults(func=_cmd_view)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Fold Gigaflow actor_005700 and fine-tune it in Sophy PPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from gigaflow_fold import export_folded_checkpoint

DEFAULT_GIGAFLOW = Path("/home/ubuntu/gigaflow-20h-ckpts/actor_005700.pt")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "gigaflow_sophy_sim_finetune.json"


def _worktree_env() -> dict[str, str]:
    training = Path(__file__).resolve().parent
    repo = training.parent
    policy = repo / "libs" / "f1tenth_policy"
    env = os.environ.copy()
    prefix = os.pathsep.join((str(training), str(policy)))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = prefix if not existing else prefix + os.pathsep + existing
    return env


def _smoke_patch(base: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["ppo"]["rollout_steps"] = 8
    cfg["ppo"]["epochs"] = 1
    cfg["ppo"]["env_minibatches"] = 8
    cfg["ppo"]["critic_warmup_updates"] = 1
    cfg["schedule"]["total_transitions"] = 128
    cfg["schedule"]["log_interval_transitions"] = 64
    cfg["schedule"]["export_interval_transitions"] = 0
    cfg["schedule"]["eval_interval_transitions"] = 0
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gigaflow-ckpt", type=Path, default=DEFAULT_GIGAFLOW)
    parser.add_argument(
        "--folded-output",
        type=Path,
        default=Path("outputs/gigaflow_folded_005700.pt"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-transitions", type=int, default=None)
    parser.add_argument("--run-id", type=str, default="gigaflow-fold-sophy")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args(argv)

    report = export_folded_checkpoint(args.gigaflow_ckpt, args.folded_output)
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    if args.fold_only:
        return 0

    training = Path(__file__).resolve().parent
    config_path = args.config
    if args.smoke:
        base = json.loads(args.config.read_text(encoding="utf-8"))
        smoke_path = args.folded_output.with_name("gigaflow_sophy_sim_finetune_smoke.json")
        smoke_path.write_text(
            json.dumps(_smoke_patch(base), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_path = smoke_path

    cmd = [
        sys.executable,
        str(training / "standalone_trainer.py"),
        "--algorithm",
        "ppo",
        "--config",
        str(config_path),
        "--init-ckpt",
        str(args.folded_output),
        "--device",
        "cpu" if args.smoke and args.device == "auto" else args.device,
        "--run-id",
        args.run_id if not args.smoke else f"{args.run_id}-smoke",
        "--no-compile",
        "--no-wandb" if args.smoke else "--wandb",
    ]
    if args.smoke:
        cmd.extend(["--num-envs", "8", "--total-transitions", "128"])
    else:
        if args.num_envs is not None:
            cmd.extend(["--num-envs", str(args.num_envs)])
        if args.total_transitions is not None:
            cmd.extend(["--total-transitions", str(args.total_transitions)])
    if args.run_dir:
        cmd.extend(["--run-dir", args.run_dir])
    print("exec", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(training), env=_worktree_env())


if __name__ == "__main__":
    raise SystemExit(main())

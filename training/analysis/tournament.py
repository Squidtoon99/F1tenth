"""Solo / fixed-opponent ranking benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config import DEFAULT_CONFIG
from evaluation import deterministic_rollout
from f1tenth_env import F1tenthEnv
from f1tenth_policy import (
    actor_from_architecture,
    load_sensor_artifact,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.normalizer import load_obs_normalizer
from standalone_trainer import build_env_cfg


def load_champion(path: str, device: torch.device = torch.device("cpu")):
    payload = load_sensor_artifact(path, map_location=device)
    architecture = payload["actor_architecture"]
    validate_sensor_policy_artifact(
        payload,
        expected_architecture=architecture,
    )
    actor = actor_from_architecture(architecture).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    normalizer = load_obs_normalizer(
        payload["obs_norm"]["mean"],
        payload["obs_norm"]["var"],
        device=device,
        count=float(payload["obs_norm"].get("count", 1e-8)),
    )
    return actor, normalizer, payload


def make_eval_env(
    *,
    opponent_strategy: str | None,
    num_envs: int = 1,
    cfg: dict | None = None,
) -> F1tenthEnv:
    cfg = cfg or DEFAULT_CONFIG
    env_cfg = build_env_cfg(
        cfg,
        opponent_strategy=opponent_strategy,
        domain_randomization={
            **cfg["env"]["domain_randomization"],
            "enabled": False,
        },
    )
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def race_metrics_from_rollout(extras_list: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate pace / lifespan / course-limit / collision from rollout extras."""
    progress = 0.0
    oob_steps = 0
    collisions = 0
    lifespan = 0
    for extras in extras_list:
        metrics = extras.get("metrics") or {}
        terms = (extras.get("rewards") or {}).get("terms") or {}
        term = extras.get("termination") or {}
        if "progress_ds" in metrics:
            progress += float(torch.as_tensor(metrics["progress_ds"]).sum())
        elif "progress" in terms:
            progress += float(torch.as_tensor(terms["progress"]).sum())
        if "oob_mask" in metrics:
            oob_steps += int(torch.as_tensor(metrics["oob_mask"]).sum().item())
        if "car_collision" in metrics:
            collisions += int(torch.as_tensor(metrics["car_collision"]).sum().item())
        if any(bool(torch.as_tensor(v).any()) for v in term.values() if hasattr(v, "any")):
            pass
        lifespan += 1
    return {
        "progress_m": progress,
        "oob_steps": float(oob_steps),
        "collisions": float(collisions),
        "lifespan_steps": float(lifespan),
    }


def run_solo_benchmark(
    checkpoint: str,
    *,
    num_steps: int = 200,
    device: str = "cpu",
) -> dict[str, float]:
    actor, normalizer, _ = load_champion(checkpoint, device=torch.device(device))
    env = make_eval_env(opponent_strategy=None)
    try:
        extras_list: list[dict] = []

        def callback(step_i, rollout_env, state_before, reward, done, extras):
            del step_i, rollout_env, state_before, reward, done
            extras_list.append(extras)

        deterministic_rollout(
            env,
            actor,
            normalizer.normalize,
            num_steps=num_steps,
            control_interval=int(DEFAULT_CONFIG["env"]["control_interval"]),
            clip_actions=float(DEFAULT_CONFIG["env"]["clip_actions"]),
            seed=0,
            callback=callback,
            with_sensors=True,
        )
        return race_metrics_from_rollout(extras_list)
    finally:
        env.close()


def rank_checkpoints(
    checkpoints: list[str],
    *,
    num_steps: int = 200,
) -> list[dict[str, Any]]:
    rows = []
    for path in checkpoints:
        metrics = run_solo_benchmark(path, num_steps=num_steps)
        rows.append({"checkpoint": path, **metrics})
    rows.sort(key=lambda r: (-r["progress_m"], r["oob_steps"], r["collisions"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="Sensor policy artifacts")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = rank_checkpoints(args.checkpoints, num_steps=args.num_steps)
    text = json.dumps(rows, indent=2)
    if args.out is not None:
        args.out.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

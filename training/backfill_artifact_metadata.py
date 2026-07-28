"""Backfill missing sensor-policy artifact metadata fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from f1tenth_policy import STEERING_ACTION_MODE, STEERING_DELTA_MAX_RAD


def backfill_steering_metadata(
    path: Path,
    *,
    steering_action_mode: str = STEERING_ACTION_MODE,
    steering_delta_max_rad: float = STEERING_DELTA_MAX_RAD,
    dry_run: bool = False,
) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: artifact payload must be a mapping")
    changed = False
    if payload.get("steering_action_mode") != steering_action_mode:
        payload["steering_action_mode"] = steering_action_mode
        changed = True
    if steering_action_mode == "delta":
        current = payload.get("steering_delta_max_rad")
        if current is None or abs(float(current) - steering_delta_max_rad) > 1.0e-9:
            payload["steering_delta_max_rad"] = float(steering_delta_max_rad)
            changed = True
    if not changed:
        return False
    if dry_run:
        return True
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill steering_action_mode on sensor policy artifacts"
    )
    parser.add_argument("paths", nargs="+", help="Artifact paths or directories")
    parser.add_argument(
        "--steering-action-mode",
        default=STEERING_ACTION_MODE,
        choices=("delta", "absolute"),
    )
    parser.add_argument(
        "--steering-delta-max-rad",
        type=float,
        default=STEERING_DELTA_MAX_RAD,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            targets.extend(sorted(path.glob("policy_*.pt")))
        else:
            targets.append(path)
    if not targets:
        raise SystemExit("No artifacts matched")

    updated = 0
    for path in targets:
        if backfill_steering_metadata(
            path,
            steering_action_mode=args.steering_action_mode,
            steering_delta_max_rad=args.steering_delta_max_rad,
            dry_run=args.dry_run,
        ):
            updated += 1
            print(f"{'would update' if args.dry_run else 'updated'}: {path}")
    print(f"done: {updated}/{len(targets)} artifacts changed")


if __name__ == "__main__":
    main()

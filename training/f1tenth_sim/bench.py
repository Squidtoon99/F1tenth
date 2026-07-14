"""Throughput benchmark for the pure-Torch vehicle simulator.

Run as ``python -m f1tenth_sim.bench`` (from ``training/``). Reports batched
substep/control-step throughput across env counts and available devices.
"""

from __future__ import annotations

import argparse
import time

import torch

from .params import VehicleParams
from .sim import TorchVehicleSim


def _devices(requested: str | None) -> list[str]:
    if requested:
        return [requested]
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _bench(device: str, n_envs: int, control_steps: int,
           substeps: int, warmup: int) -> dict:
    dev = torch.device(device)
    params = VehicleParams.from_config({"tire_friction": 0.9})
    sim = TorchVehicleSim(params, n_envs, device=dev, sim_dt=0.005, control_dt=0.05)
    quat = torch.zeros(n_envs, 4, device=dev)
    quat[:, 0] = 1.0
    sim.reset(None, torch.zeros(n_envs, 3, device=dev), quat,
              torch.zeros(n_envs, device=dev))
    a = torch.zeros(n_envs, 2, device=dev)
    a[:, 0] = 0.5
    a[:, 1] = 0.1

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        sim.step(a, n_steps=substeps)
    sync()

    t0 = time.perf_counter()
    for _ in range(control_steps):
        sim.step(a, n_steps=substeps)
    sync()
    dt = time.perf_counter() - t0

    finite = bool(torch.isfinite(sim.read_state()["base_pos"]).all().item())
    ctrl_per_s = n_envs * control_steps / dt
    return {
        "control_per_s": ctrl_per_s,
        "substep_per_s": ctrl_per_s * substeps,
        "finite": finite,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--envs", type=int, nargs="+",
                    default=[256, 1024, 4096, 16384])
    ap.add_argument("--control-steps", type=int, default=100)
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  control_steps={args.control_steps} "
          f"substeps={args.substeps}")
    header = f"{'device':>6} {'n_envs':>8} " \
             f"{'ctrl/s':>14} {'substep/s':>16} {'finite':>7}"
    print(header)
    print("-" * len(header))
    for device in _devices(args.device):
        for n in args.envs:
            r = _bench(device, n, args.control_steps, args.substeps, args.warmup)
            print(f"{device:>6} {n:>8} "
                  f"{r['control_per_s']:>14,.0f} {r['substep_per_s']:>16,.0f} "
                  f"{str(r['finite']):>7}")


if __name__ == "__main__":
    main()

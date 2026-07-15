from __future__ import annotations

import pytest
import torch
import warp as wp

from f1tenth_sim.params import VehicleParams
from f1tenth_sim.sim_warp import WarpVehicleSim


def _reset(sim, num_envs, device, speed=None):
    pos = torch.zeros(num_envs, 3, device=device)
    quat = torch.zeros(num_envs, 4, device=device)
    quat[:, 0] = 1.0
    if speed is None:
        speed = torch.zeros(num_envs, device=device)
    sim.reset(None, pos, quat, speed)


def test_warp_cpu_random_trajectory_is_deterministic():
    params = VehicleParams.from_config({"tire_friction": 0.9})
    num_envs = 32
    first = WarpVehicleSim(
        params, num_envs, device="cpu", sim_dt=0.005, control_dt=0.05
    )
    second = WarpVehicleSim(
        params, num_envs, device="cpu", sim_dt=0.005, control_dt=0.05
    )
    speed = torch.linspace(0.0, 4.0, num_envs)
    _reset(first, num_envs, torch.device("cpu"), speed)
    _reset(second, num_envs, torch.device("cpu"), speed)
    actions = torch.rand(
        20, num_envs, 2, generator=torch.Generator().manual_seed(0)
    ) * 2.0 - 1.0
    for action in actions:
        first.step(action, n_steps=10)
        second.step(action, n_steps=10)

    first_state = first.read_state()
    second_state = second.read_state()
    for key in ("base_pos", "base_lin_vel", "base_ang_vel", "base_lin_acc"):
        assert torch.equal(first_state[key], second_state[key])
        assert torch.isfinite(first_state[key]).all()


def test_warp_vehicle_direction_and_batch_independence():
    params = VehicleParams.from_config({"tire_friction": 0.9})
    sim = WarpVehicleSim(params, 3, device="cpu")
    _reset(sim, 3, torch.device("cpu"))
    actions = torch.tensor([[0.5, 0.0], [0.5, 0.3], [0.5, -0.3]])
    for _ in range(30):
        sim.step(actions)
    state = sim.read_state()
    yaw_rate = state["base_ang_vel"][:, 2]
    assert abs(float(yaw_rate[0])) < 1.0e-3
    assert float(yaw_rate[1]) > 0.0
    assert float(yaw_rate[2]) < 0.0
    assert float(state["base_lin_vel"][0, 0]) > 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_warp_cuda_is_finite_and_deterministic():
    params = VehicleParams.from_config({"tire_friction": 0.9})
    num_envs = 256
    first = WarpVehicleSim(params, num_envs, device="cuda")
    second = WarpVehicleSim(params, num_envs, device="cuda")
    _reset(first, num_envs, torch.device("cuda"))
    _reset(second, num_envs, torch.device("cuda"))
    actions = torch.rand(
        40,
        num_envs,
        2,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(7),
    ) * 2.0 - 1.0
    for action in actions:
        first.step(action)
        second.step(action)
    wp.synchronize_device("cuda:0")

    first_state = first.read_state()
    second_state = second.read_state()
    for key in ("base_pos", "base_lin_vel", "base_ang_vel", "base_lin_acc"):
        assert torch.equal(first_state[key], second_state[key])
        assert torch.isfinite(first_state[key]).all()

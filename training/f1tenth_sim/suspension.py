"""Load transfer / suspension models producing per-wheel normal load ``Fz``.

Two tiers, both fully vectorized over the env batch and wheels ``[LR, RR, LF, RF]``:

- ``quasi_static_loads`` (Tier 1): instantaneous longitudinal + lateral weight
  transfer from body accelerations ``ax, ay`` about a static axle split. Sums to
  the total vehicle weight (no vertical dynamics).
- ``SuspensionFilter`` (Tier 2): a stable, critically-damped first-order relaxation
  of the quasi-static target, approximating a spring-damper's dynamic ``Fz`` lag
  and exposing steady roll/pitch angle estimates. This keeps the vertical response
  smooth without introducing a stiff heave/roll/pitch ODE.
"""

from __future__ import annotations

import torch


def static_wheel_loads(params, n_envs: int, device, dtype) -> torch.Tensor:
    front, rear = params.static_axle_loads()
    fz = torch.empty((n_envs, 4), device=device, dtype=dtype)
    fz[:, 0] = rear / 2.0
    fz[:, 1] = rear / 2.0
    fz[:, 2] = front / 2.0
    fz[:, 3] = front / 2.0
    return fz


def quasi_static_loads(
    params,
    ax: torch.Tensor,
    ay: torch.Tensor,
    mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-wheel normal load (N,4) under body-frame specific forces ax, ay (m/s^2).

    +ax (forward accel) shifts load rearward; +ay (leftward accel, e.g. a left
    turn) shifts load onto the right (outer) wheels. Wheel order [LR, RR, LF, RF].
    """
    g = params.gravity
    m = mass if mass is not None else torch.as_tensor(
        params.mass, device=ax.device, dtype=ax.dtype
    )
    w = m * g
    wf = w * params.lr / params.wheelbase
    wr = w * params.lf / params.wheelbase

    d_long = m * ax * params.h_cg / params.wheelbase
    fz_front = wf - d_long
    fz_rear = wr + d_long

    d_lat_total = m * ay * params.h_cg / max(params.track_width, 1e-6)
    kf = params.roll_stiffness_front
    d_lat_front = kf * d_lat_total
    d_lat_rear = (1.0 - kf) * d_lat_total

    fz = torch.stack(
        [
            fz_rear / 2.0 - d_lat_rear,
            fz_rear / 2.0 + d_lat_rear,
            fz_front / 2.0 - d_lat_front,
            fz_front / 2.0 + d_lat_front,
        ],
        dim=-1,
    )
    return fz.clamp_min(0.0)


class SuspensionFilter:
    """Critically-damped first-order lag of the quasi-static load transfer."""

    def __init__(self, params, n_envs: int, device, dtype):
        self.params = params
        self.fz = static_wheel_loads(params, n_envs, device, dtype)
        wn = (max(params.susp_stiffness, 1e-3) / max(params.mass / 4.0, 1e-3)) ** 0.5
        self.tau = 1.0 / max(wn, 1e-3)

    def reset(self, mask: torch.Tensor, params) -> None:
        static = static_wheel_loads(
            params, self.fz.shape[0], self.fz.device, self.fz.dtype
        )
        self.fz = torch.where(mask.unsqueeze(1), static, self.fz)

    def step(
        self,
        ax: torch.Tensor,
        ay: torch.Tensor,
        dt: float,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = quasi_static_loads(self.params, ax, ay, mass)
        alpha = dt / (self.tau + dt)
        self.fz = self.fz + alpha * (target - self.fz)
        return self.fz

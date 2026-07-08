"""Vectorized Pacejka Magic-Formula tire with a combined-slip friction ellipse.

Given per-wheel slip ratio ``kappa``, slip angle ``alpha`` (rad), normal load
``Fz`` (N) and surface friction ``mu``, returns wheel-frame longitudinal and
lateral forces ``Fx, Fy`` (N). All inputs are broadcastable tensors of shape
``(N, W)`` (typically ``W = 4`` wheels); scalars/params are Python floats.

The peak force is load-sensitive (``mu`` decreases as ``Fz`` rises above the
static per-wheel load) and the pure-slip longitudinal/lateral forces are coupled
through a friction ellipse so the combined force magnitude never exceeds
``mu * Fz`` -- the drift/saturation behaviour that makes the GT-Sophy tyre-slip
penalty physically meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TireModel:
    B_long: float = 11.0
    C_long: float = 1.55
    E_long: float = 0.55
    B_lat: float = 11.0
    C_lat: float = 1.45
    E_lat: float = 0.6
    load_sens: float = 0.15
    Fz0: float = 9.17

    def _magic(self, slip: torch.Tensor, B: float, C: float, E: float) -> torch.Tensor:
        bx = B * slip
        return torch.sin(C * torch.atan(bx - E * (bx - torch.atan(bx))))

    def load_scaled_mu(self, Fz: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        rel = Fz / max(self.Fz0, 1e-6) - 1.0
        return mu * (1.0 - self.load_sens * rel)

    def forces(
        self,
        kappa: torch.Tensor,
        alpha: torch.Tensor,
        Fz: torch.Tensor,
        mu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Fz = Fz.clamp_min(0.0)
        peak = self.load_scaled_mu(Fz, mu).clamp_min(1e-4) * Fz

        fx0 = peak * self._magic(kappa, self.B_long, self.C_long, self.E_long)
        fy0 = peak * self._magic(alpha, self.B_lat, self.C_lat, self.E_lat)

        # Friction ellipse: scale the pure-slip forces so the combined magnitude
        # stays within the peak (mu * Fz) friction circle.
        peak_safe = peak.clamp_min(1e-6)
        combined = torch.sqrt((fx0 / peak_safe) ** 2 + (fy0 / peak_safe) ** 2)
        scale = 1.0 / combined.clamp_min(1.0)
        return fx0 * scale, fy0 * scale


def make_tire_from_params(params) -> TireModel:
    return TireModel(
        B_long=params.tire_B_long,
        C_long=params.tire_C_long,
        E_long=params.tire_E_long,
        B_lat=params.tire_B_lat,
        C_lat=params.tire_C_lat,
        E_lat=params.tire_E_lat,
        load_sens=params.tire_load_sens,
        Fz0=params.static_wheel_load(),
    )

"""Running sensor observation normalizer (Welford mean/std), owned by the actor.

The reference paper normalizes every observation to ``[-1, 1]``; the raw 1097-D
sensor vector mixes LiDAR ranges (metres, up to ~30) with proprioceptive rates
(rad/s) and normalized action-history channels with no shared scale. This
module tracks per-dimension running mean/variance and maps to a clipped
z-score so the CNN sees comparable magnitudes across channels.

Registered as buffers on an ``nn.Module`` (not parameters) so the statistics
ride along for free with ``actor.state_dict()`` / ``load_state_dict()``:
checkpoints, resume, and deployable artifacts all serialize and restore them
through the actor's own state, with no separate wiring.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

# Maps +/- this many running standard deviations to the [-1, 1] edge.
DEFAULT_CLIP_SIGMA = 5.0
DEFAULT_VAR_EPS = 1e-6


class SensorNormalizer(nn.Module):
    """Per-dimension running mean/std via Welford's parallel-batch update.

    A never-updated normalizer (``count == 0``) is the identity transform, so
    a freshly built actor behaves exactly as before any statistics exist.
    """

    def __init__(
        self,
        dim: int,
        *,
        clip_sigma: float = DEFAULT_CLIP_SIGMA,
        eps: float = DEFAULT_VAR_EPS,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.clip_sigma = float(clip_sigma)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.dim, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(self.dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        """Fold a batch of raw observations ``[..., dim]`` into running stats."""
        if obs.shape[-1] != self.dim:
            raise ValueError(
                f"obs last dim {obs.shape[-1]} != normalizer dim {self.dim}"
            )
        flat = obs.detach().reshape(-1, self.dim).to(dtype=torch.float64)
        batch_count = float(flat.shape[0])
        if batch_count == 0.0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        count = float(self.count.item())
        total = count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        new_m2 = (
            self.m2
            + batch_var * batch_count
            + delta.square() * (count * batch_count / total)
        )
        self.mean.copy_(new_mean)
        self.m2.copy_(new_m2)
        self.count.fill_(total)

    def std(self) -> torch.Tensor:
        var = self.m2 / self.count.clamp(min=1.0)
        return torch.sqrt(torch.clamp(var, min=self.eps))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Clipped z-score; identity until at least one batch has been folded in."""
        mean = self.mean.to(device=obs.device, dtype=obs.dtype)
        std = self.std().to(device=obs.device, dtype=obs.dtype)
        z = torch.clamp((obs - mean) / (std * self.clip_sigma), -1.0, 1.0)
        uninitialized = (self.count < 1.0).to(dtype=obs.dtype, device=obs.device)
        return uninitialized * obs + (1.0 - uninitialized) * z

    def to_dict(self) -> dict[str, Any]:
        """Deploy-artifact view: plain tensors, independent of the actor module."""
        return {
            "dim": self.dim,
            "clip_sigma": self.clip_sigma,
            "eps": self.eps,
            "mean": self.mean.detach().cpu().clone(),
            "m2": self.m2.detach().cpu().clone(),
            "count": float(self.count.item()),
        }

    def load_dict(self, state: Mapping[str, Any]) -> None:
        dim = int(state["dim"])
        if dim != self.dim:
            raise ValueError(f"normalizer dim mismatch: {dim} != {self.dim}")
        with torch.no_grad():
            self.mean.copy_(torch.as_tensor(state["mean"], dtype=torch.float64))
            self.m2.copy_(torch.as_tensor(state["m2"], dtype=torch.float64))
            self.count.fill_(float(state["count"]))

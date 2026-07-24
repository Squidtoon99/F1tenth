from __future__ import annotations

import torch

from f1tenth_policy.layout import OBS_NORM_CLIP, OBS_NORM_EPS


class ObsNormalizer:
    """Welford parallel mean/variance normalizer (float32 stats)."""

    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        eps: float = OBS_NORM_EPS,
        clip: float = OBS_NORM_CLIP,
    ):
        self.device = device
        self.eps = float(eps)
        self.clip = float(clip)
        self.mean = torch.zeros(obs_dim, device=device, dtype=torch.float32)
        self.var = torch.ones(obs_dim, device=device, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.to(torch.float32)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        self.var = m2 / tot_count
        self.count = tot_count

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x.to(torch.float32) - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normed, -self.clip, self.clip)

    @torch.no_grad()
    def normalize_into(self, x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x.to(torch.float32))
        out.sub_(self.mean)
        out.div_(torch.sqrt(self.var + self.eps))
        out.clamp_(-self.clip, self.clip)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().clone(),
            "var": self.var.detach().cpu().clone(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = torch.as_tensor(
            state["mean"], device=self.device, dtype=torch.float32
        )
        self.var = torch.as_tensor(
            state["var"], device=self.device, dtype=torch.float32
        )
        self.count = float(state.get("count", self.eps))


def load_obs_normalizer(
    mean,
    var,
    *,
    device: torch.device,
    eps: float = OBS_NORM_EPS,
    clip: float = OBS_NORM_CLIP,
    count: float | None = None,
) -> ObsNormalizer:
    normalizer = ObsNormalizer(
        obs_dim=int(torch.as_tensor(mean).numel()),
        device=device,
        eps=eps,
        clip=clip,
    )
    normalizer.mean = torch.as_tensor(mean, device=device, dtype=torch.float32)
    normalizer.var = torch.as_tensor(var, device=device, dtype=torch.float32)
    if count is not None:
        normalizer.count = float(count)
    return normalizer

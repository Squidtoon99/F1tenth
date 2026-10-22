"""Fold a constant Gigaflow condition embedding into a Sophy 1097-D GRU actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from f1tenth_policy.actor import (
    LidarCNNEncoder,
    SquashedGaussianLidarGRUActor,
    actor_from_architecture,
    mlp,
)
from f1tenth_policy.artifact import build_sensor_artifact_payload
from f1tenth_policy.layout import (
    ACTOR_ARCHITECTURE_NAME,
    CRITIC_OBS_DIM,
)

CONDITION_EMBED_DIM = 64
GIGAFLOW_CLIP_SIGMA = 5.0
GIGAFLOW_VAR_EPS = 1e-6
GIGAFLOW_LOG_STD_MIN = -5.0
GIGAFLOW_LOG_STD_MAX = 0.5
SIM_READY_V2_PLANT = {
    "f_drive_max": 21.4688,
    "f_brake_max": 9.1392,
    "c_coast": 4.8048,
    "steer_gain": 1.0512,
    "t_delta": 0.09,
    "lat_stiffness_scale": 5.3293,
}
IRL_RAW_DEFAULTS = {
    "alpha_collision": 3.0,
    "alpha_boundary": 3.0,
    "alpha_l_center": 0.038,
    "alpha_center_bias": 0.0,
    "alpha_passing": 0.4,
    "drive_scale": 1.0,
    "steer_scale": 1.0,
    "accel_scale": 1.0,
    "vmax_scale": 1.0,
    "mass_kg": 3.444,
}


def gigaflow_normalize(
    obs: torch.Tensor,
    mean: torch.Tensor,
    m2: torch.Tensor,
    count: float,
    *,
    clip_sigma: float = GIGAFLOW_CLIP_SIGMA,
    eps: float = GIGAFLOW_VAR_EPS,
) -> torch.Tensor:
    """Match Gigaflow SensorNormalizer: z = clip((x-mean)/(std*clip_sigma), -1, 1)."""
    if float(count) < 1.0:
        return obs
    var = m2 / max(float(count), 1.0)
    std = torch.sqrt(torch.clamp(var, min=float(eps)))
    z = (obs - mean.to(dtype=obs.dtype, device=obs.device)) / (
        std.to(dtype=obs.dtype, device=obs.device) * float(clip_sigma)
    )
    return torch.clamp(z, -1.0, 1.0)


def welford_to_obs_norm(
    mean: torch.Tensor,
    m2: torch.Tensor,
    count: float,
    *,
    clip_sigma: float = GIGAFLOW_CLIP_SIGMA,
    eps: float = GIGAFLOW_VAR_EPS,
) -> dict[str, Any]:
    """Sophy obs_norm whose z-score matches Gigaflow when clip=1."""
    count_f = float(count)
    var = m2.to(dtype=torch.float32) / max(count_f, 1.0)
    var = torch.clamp(var, min=float(eps))
    return {
        "mean": mean.detach().cpu().to(torch.float32).clone(),
        "var": (var * (float(clip_sigma) ** 2)).contiguous(),
        "count": count_f,
        "clip": 1.0,
        "clip_sigma": float(clip_sigma),
        "eps": 0.0,
    }


def condition_embedding(state: dict[str, torch.Tensor], c_irl: torch.Tensor) -> torch.Tensor:
    hidden = F.relu(
        F.linear(
            c_irl,
            state["condition_encoder.net.0.weight"],
            state["condition_encoder.net.0.bias"],
        )
    )
    return F.relu(
        F.linear(
            hidden,
            state["condition_encoder.net.2.weight"],
            state["condition_encoder.net.2.bias"],
        )
    )


def fold_gru_input(
    weight_ih: torch.Tensor,
    bias_ih: torch.Tensor,
    embed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embed = embed.reshape(-1).to(dtype=weight_ih.dtype, device=weight_ih.device)
    cond_dim = int(embed.numel())
    if weight_ih.shape[1] <= cond_dim:
        raise ValueError(
            f"GRU input {weight_ih.shape[1]} is too small to fold {cond_dim}-D embed"
        )
    trunk = int(weight_ih.shape[1] - cond_dim)
    folded_w = weight_ih[:, :trunk].contiguous()
    folded_b = bias_ih + weight_ih[:, trunk:] @ embed
    return folded_w, folded_b


def irl_condition_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    vec = payload.get("deployment_condition_normalized")
    if vec is None:
        raise KeyError("checkpoint missing deployment_condition_normalized")
    return torch.as_tensor(vec, dtype=torch.float32)


def sophy_architecture_from_gigaflow(arch: dict[str, Any], gru_in: int) -> dict[str, Any]:
    projection = int(arch["cnn_projection_dim"])
    proprio = int(arch["proprio_dim"])
    if gru_in != projection + proprio:
        raise ValueError(
            f"folded GRU in={gru_in} != projection+proprio={projection + proprio}"
        )
    pool_mode = str(arch.get("pool_mode", "avg"))
    return {
        "name": ACTOR_ARCHITECTURE_NAME,
        "version": 1,
        "obs_dim": int(arch["sensor_obs_dim"]),
        "lidar_dim": int(arch["lidar_dim"]),
        "proprio_dim": proprio,
        "conv_channels": list(arch["cnn_conv_channels"]),
        "kernels": list(arch["cnn_kernels"]),
        "strides": list(arch["cnn_strides"]),
        "padding": list(arch["cnn_padding"]),
        "pool": "adaptive_max_1d" if pool_mode == "max" else "adaptive_avg_1d",
        "pool_mode": pool_mode,
        "pool_bins": int(arch["pool_bins"]),
        "projection_dim": projection,
        "gru_hidden_dim": int(arch["gru_hidden_dim"]),
        "hidden_layers": list(arch["mlp_sizes"]),
        "activation": "relu",
        "action_dim": int(arch["action_dim"]),
    }


def folded_sophy_state_dict(
    gigaflow_sd: dict[str, torch.Tensor],
    embed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    w_ih, b_ih = fold_gru_input(
        gigaflow_sd["gru.weight_ih_l0"],
        gigaflow_sd["gru.bias_ih_l0"],
        embed,
    )
    out = {}
    for key, value in gigaflow_sd.items():
        if key.startswith("sensor_normalizer.") or key.startswith("condition_encoder."):
            continue
        if key == "gru.weight_ih_l0":
            out[key] = w_ih
        elif key == "gru.bias_ih_l0":
            out[key] = b_ih
        else:
            out[key] = value
    return out


class ConditionedTeacher(nn.Module):
    """Gigaflow CNN-GRU with an explicit condition channel (parity only)."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__()
        arch = payload["actor_architecture"]
        sd = payload["actor_state_dict"]
        self.lidar_dim = int(arch["lidar_dim"])
        self.encoder = LidarCNNEncoder(
            lidar_dim=self.lidar_dim,
            pool_bins=int(arch["pool_bins"]),
            projection_dim=int(arch["cnn_projection_dim"]),
            pool_mode=str(arch.get("pool_mode", "avg")),
        )
        gru_in = int(sd["gru.weight_ih_l0"].shape[1])
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=int(arch["gru_hidden_dim"]),
            batch_first=True,
        )
        hidden_sizes = list(arch["mlp_sizes"])
        self.net = mlp(
            [int(arch["gru_hidden_dim"])] + hidden_sizes,
            nn.ReLU,
            nn.ReLU,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], int(arch["action_dim"]))
        self.log_std_layer = nn.Linear(hidden_sizes[-1], int(arch["action_dim"]))
        self.encoder.load_state_dict(
            {
                key[len("encoder.") :]: value
                for key, value in sd.items()
                if key.startswith("encoder.")
            }
        )
        self.gru.load_state_dict(
            {
                key[len("gru.") :]: value
                for key, value in sd.items()
                if key.startswith("gru.")
            }
        )
        self.net.load_state_dict(
            {
                key[len("net.") :]: value
                for key, value in sd.items()
                if key.startswith("net.")
            }
        )
        self.mu_layer.load_state_dict(
            {
                "weight": sd["mu_layer.weight"],
                "bias": sd["mu_layer.bias"],
            }
        )
        self.log_std_layer.load_state_dict(
            {
                "weight": sd["log_std_layer.weight"],
                "bias": sd["log_std_layer.bias"],
            }
        )
        sn = payload["sensor_normalizer"]
        self.register_buffer("norm_mean", torch.as_tensor(sn["mean"]).float())
        self.register_buffer("norm_m2", torch.as_tensor(sn["m2"]).float())
        self.norm_count = float(sn["count"])
        self.clip_sigma = float(sn.get("clip_sigma", GIGAFLOW_CLIP_SIGMA))
        self.embed = condition_embedding(sd, irl_condition_from_payload(payload))
        self.gru_hidden_dim = int(arch["gru_hidden_dim"])

    def initial_hidden(self, batch_size, *, device=None, dtype=None):
        return torch.zeros(
            int(batch_size),
            self.gru_hidden_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, obs, hidden):
        normed = gigaflow_normalize(
            obs, self.norm_mean, self.norm_m2, self.norm_count, clip_sigma=self.clip_sigma
        )
        lidar = normed[..., : self.lidar_dim]
        proprio = normed[..., self.lidar_dim :]
        features = self.encoder(lidar)
        cond = self.embed.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
        trunk = torch.cat([features, proprio, cond], dim=-1)
        out, h_n = self.gru(trunk.unsqueeze(1), hidden.unsqueeze(0).contiguous())
        net_out = self.net(out.squeeze(1))
        mu = self.mu_layer(net_out)
        actions = torch.tanh(mu)
        return actions, h_n.squeeze(0)


def fold_gigaflow_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SquashedGaussianLidarGRUActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Gigaflow checkpoint must be a mapping")
    arch = payload["actor_architecture"]
    sd = payload["actor_state_dict"]
    c_irl = irl_condition_from_payload(payload)
    embed = condition_embedding(sd, c_irl)
    folded_w, folded_b = fold_gru_input(sd["gru.weight_ih_l0"], sd["gru.bias_ih_l0"], embed)
    sophy_arch = sophy_architecture_from_gigaflow(arch, int(folded_w.shape[1]))
    actor = actor_from_architecture(sophy_arch)
    actor.log_std_min = GIGAFLOW_LOG_STD_MIN
    actor.log_std_max = GIGAFLOW_LOG_STD_MAX
    actor.load_state_dict(folded_sophy_state_dict(sd, embed), strict=True)
    actor.gru.flatten_parameters()
    sn = payload["sensor_normalizer"]
    obs_norm = welford_to_obs_norm(
        sn["mean"],
        sn["m2"],
        float(sn["count"]),
        clip_sigma=float(sn.get("clip_sigma", GIGAFLOW_CLIP_SIGMA)),
        eps=float(sn.get("eps", GIGAFLOW_VAR_EPS)),
    )
    fold_meta = {
        "source_architecture": str(arch.get("name")),
        "source_gru_in": int(sd["gru.weight_ih_l0"].shape[1]),
        "folded_gru_in": int(folded_w.shape[1]),
        "condition_embed_dim": CONDITION_EMBED_DIM,
        "c_irl": c_irl.detach().cpu().tolist(),
        "irl_raw_defaults": dict(IRL_RAW_DEFAULTS),
        "deployment_style": payload.get("deployment_style"),
        "log_std_min": GIGAFLOW_LOG_STD_MIN,
        "log_std_max": GIGAFLOW_LOG_STD_MAX,
        "sim_ready_v2_plant": dict(SIM_READY_V2_PLANT),
    }
    artifact = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=obs_norm,
        actor_architecture=sophy_arch,
        env_transitions=0,
        actor_obs_dim=int(sophy_arch["obs_dim"]),
        critic_obs_dim=CRITIC_OBS_DIM,
        action_dim=int(sophy_arch["action_dim"]),
        f_drive_max=float(SIM_READY_V2_PLANT["f_drive_max"]),
        f_brake_max=float(SIM_READY_V2_PLANT["f_brake_max"]),
        training_protocol={"gigaflow_fold": fold_meta},
    )
    artifact["gigaflow_fold"] = fold_meta
    return actor.to(device), artifact


def export_folded_checkpoint(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    actor, artifact = fold_gigaflow_checkpoint(checkpoint)
    del actor
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    report = {
        "ok": True,
        "output": str(path),
        "actor_architecture": artifact["actor_architecture"],
        "gigaflow_fold": {
            key: value
            for key, value in artifact["gigaflow_fold"].items()
            if key != "c_irl"
        },
        "c_irl": artifact["gigaflow_fold"]["c_irl"],
        "obs_norm_count": artifact["obs_norm"]["count"],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_folded_checkpoint(args.checkpoint, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

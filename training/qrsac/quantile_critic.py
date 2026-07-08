import torch
import torch.nn as nn


class QuantileCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), num_quantiles=32):
        super().__init__()
        layers = []
        last_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, num_quantiles)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.head(self.backbone(x))  # (B, 32)

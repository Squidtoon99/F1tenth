import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

LIDAR_DIM = 1081
PROPRIO_DIM = 12
CNN_CONV_CHANNELS = (32, 64, 64)
CNN_KERNELS = (7, 5, 3)
CNN_STRIDES = (3, 2, 2)
CNN_PADDING = (3, 2, 1)
CNN_PROJECTION_DIM = 256
ALLOWED_LIDAR_POOL_BINS = frozenset((16, 32))


def _squashed_gaussian_forward(net, mu_layer, log_std_layer, act_limit, obs,
                               deterministic, with_logprob):
    net_out = net(obs)
    mu = mu_layer(net_out)
    log_std = log_std_layer(net_out)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

    pi_distribution = Normal(mu, std)
    if deterministic:
        pi_action = mu
    else:
        pi_action = pi_distribution.rsample()

    if with_logprob:
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # type: ignore
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
            axis=1
        )
    else:
        logp_pi = None

    pi_action = torch.tanh(pi_action)
    pi_action = act_limit * pi_action
    return pi_action, logp_pi


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(int(s) for s in hidden_sizes)
        self.net = mlp(
            [obs_dim] + list(hidden_sizes), activation, activation  # type: ignore
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.actor_architecture = {
            "name": "flat_mlp",
            "version": 1,
            "obs_dim": self.obs_dim,
            "hidden_layers": list(self.hidden_sizes),
            "activation": "relu",
            "action_dim": self.act_dim,
        }

    def forward(self, obs, deterministic=False, with_logprob=True):
        return _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            obs,
            deterministic,
            with_logprob,
        )


class LidarCNNEncoder(nn.Module):

    def __init__(
        self,
        lidar_dim=LIDAR_DIM,
        pool_bins=32,
        projection_dim=CNN_PROJECTION_DIM,
    ):
        super().__init__()
        pool_bins = int(pool_bins)
        if pool_bins not in ALLOWED_LIDAR_POOL_BINS:
            raise ValueError(
                f"lidar_pool_bins must be one of {sorted(ALLOWED_LIDAR_POOL_BINS)}, "
                f"got {pool_bins}"
            )
        self.lidar_dim = int(lidar_dim)
        self.pool_bins = pool_bins
        self.projection_dim = int(projection_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(
                1,
                CNN_CONV_CHANNELS[0],
                kernel_size=CNN_KERNELS[0],
                stride=CNN_STRIDES[0],
                padding=CNN_PADDING[0],
            ),
            nn.ReLU(),
            nn.Conv1d(
                CNN_CONV_CHANNELS[0],
                CNN_CONV_CHANNELS[1],
                kernel_size=CNN_KERNELS[1],
                stride=CNN_STRIDES[1],
                padding=CNN_PADDING[1],
            ),
            nn.ReLU(),
            nn.Conv1d(
                CNN_CONV_CHANNELS[1],
                CNN_CONV_CHANNELS[2],
                kernel_size=CNN_KERNELS[2],
                stride=CNN_STRIDES[2],
                padding=CNN_PADDING[2],
            ),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.projection = nn.Linear(
            CNN_CONV_CHANNELS[2] * self.pool_bins, self.projection_dim
        )

    def forward_features(self, lidar):
        x = lidar.unsqueeze(1)
        x = self.conv(x)
        pooled = self.pool(x)
        flat = pooled.flatten(1)
        return F.relu(self.projection(flat)), pooled

    def forward(self, lidar):
        features, _ = self.forward_features(lidar)
        return features


class SquashedGaussianLidarActor(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        act_limit,
        lidar_dim=LIDAR_DIM,
        proprio_dim=PROPRIO_DIM,
        pool_bins=32,
        projection_dim=CNN_PROJECTION_DIM,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.lidar_dim = int(lidar_dim)
        self.proprio_dim = int(proprio_dim)
        self.pool_bins = int(pool_bins)
        self.projection_dim = int(projection_dim)
        self.hidden_sizes = tuple(int(s) for s in hidden_sizes)
        if self.obs_dim != self.lidar_dim + self.proprio_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} must equal lidar_dim+proprio_dim="
                f"{self.lidar_dim + self.proprio_dim}"
            )
        self.encoder = LidarCNNEncoder(
            lidar_dim=self.lidar_dim,
            pool_bins=self.pool_bins,
            projection_dim=self.projection_dim,
        )
        trunk_in = self.projection_dim + self.proprio_dim
        self.net = mlp(
            [trunk_in] + list(hidden_sizes), activation, activation  # type: ignore
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.actor_architecture = {
            "name": "lidar_cnn",
            "version": 1,
            "obs_dim": self.obs_dim,
            "lidar_dim": self.lidar_dim,
            "proprio_dim": self.proprio_dim,
            "conv_channels": list(CNN_CONV_CHANNELS),
            "kernels": list(CNN_KERNELS),
            "strides": list(CNN_STRIDES),
            "padding": list(CNN_PADDING),
            "pool": "adaptive_avg_1d",
            "pool_bins": self.pool_bins,
            "projection_dim": self.projection_dim,
            "hidden_layers": list(self.hidden_sizes),
            "activation": "relu",
            "action_dim": self.act_dim,
        }

    def encode(self, obs):
        lidar = obs[..., : self.lidar_dim]
        proprio = obs[..., self.lidar_dim :]
        features, pooled = self.encoder.forward_features(lidar)
        trunk_in = torch.cat([features, proprio], dim=-1)
        return trunk_in, pooled

    def forward(self, obs, deterministic=False, with_logprob=True):
        trunk_in, _ = self.encode(obs)
        return _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            trunk_in,
            deterministic,
            with_logprob,
        )


def make_actor(
    actor_type,
    obs_dim,
    act_dim,
    hidden_sizes,
    activation=nn.ReLU,
    act_limit=1.0,
    lidar_pool_bins=32,
    lidar_dim=LIDAR_DIM,
    proprio_dim=PROPRIO_DIM,
):
    if actor_type == "flat_mlp":
        return SquashedGaussianMLPActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            act_limit=act_limit,
        )
    if actor_type == "lidar_cnn":
        return SquashedGaussianLidarActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            act_limit=act_limit,
            lidar_dim=lidar_dim,
            proprio_dim=proprio_dim,
            pool_bins=lidar_pool_bins,
        )
    raise ValueError(
        f"Unsupported actor_type={actor_type!r}; expected 'flat_mlp' or 'lidar_cnn'"
    )

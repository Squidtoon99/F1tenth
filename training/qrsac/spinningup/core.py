"""QR-SAC actor building blocks — adapter over shared f1tenth_policy."""

from f1tenth_policy.actor import (
    ALLOWED_LIDAR_POOL_BINS,
    CNN_CONV_CHANNELS,
    CNN_KERNELS,
    CNN_PADDING,
    CNN_PROJECTION_DIM,
    CNN_STRIDES,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    LOG_STD_MAX,
    LOG_STD_MIN,
    LidarCNNEncoder,
    PROPRIO_DIM,
    SquashedGaussianLidarGRUActor,
    _partition_invariant_standard_normal,
    _squashed_gaussian_forward,
    make_actor as _make_actor,
    mlp,
)
from f1tenth_policy.layout import ACTOR_ARCHITECTURE_NAME

# Retained name for import compatibility with older test imports that only need
# the GRU path. flat_mlp / feed-forward lidar_cnn are intentionally gone.
SquashedGaussianLidarActor = SquashedGaussianLidarGRUActor


def make_actor(
    actor_type=ACTOR_ARCHITECTURE_NAME,
    obs_dim=None,
    act_dim=None,
    hidden_sizes=None,
    activation=None,
    act_limit=1.0,
    lidar_pool_bins=32,
    lidar_projection_dim=CNN_PROJECTION_DIM,
    lidar_dim=LIDAR_DIM,
    proprio_dim=PROPRIO_DIM,
    gru_hidden_dim=GRU_HIDDEN_DIM,
):
    import torch.nn as nn

    if activation is None:
        activation = nn.ReLU
    if obs_dim is None or act_dim is None or hidden_sizes is None:
        raise ValueError("make_actor requires obs_dim, act_dim, and hidden_sizes")
    return _make_actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        act_limit=act_limit,
        lidar_pool_bins=lidar_pool_bins,
        lidar_projection_dim=lidar_projection_dim,
        lidar_dim=lidar_dim,
        proprio_dim=proprio_dim,
        gru_hidden_dim=gru_hidden_dim,
        actor_type=actor_type,
    )


__all__ = [
    "ALLOWED_LIDAR_POOL_BINS",
    "ACTOR_ARCHITECTURE_NAME",
    "CNN_CONV_CHANNELS",
    "CNN_KERNELS",
    "CNN_PADDING",
    "CNN_PROJECTION_DIM",
    "CNN_STRIDES",
    "GRU_HIDDEN_DIM",
    "LIDAR_DIM",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "LidarCNNEncoder",
    "PROPRIO_DIM",
    "SquashedGaussianLidarActor",
    "SquashedGaussianLidarGRUActor",
    "make_actor",
    "mlp",
    "_partition_invariant_standard_normal",
    "_squashed_gaussian_forward",
]

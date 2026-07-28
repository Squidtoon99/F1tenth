from .qrsac import Models, QRSACTrainer, QuantileCritic
from .spinningup.core import (
    LidarCNNEncoder,
    SquashedGaussianLidarGRUActor,
    make_actor,
)

# Alias retained for call sites that still import the older name.
SquashedGaussianLidarActor = SquashedGaussianLidarGRUActor

__all__ = [
    "QRSACTrainer",
    "QuantileCritic",
    "SquashedGaussianLidarActor",
    "SquashedGaussianLidarGRUActor",
    "LidarCNNEncoder",
    "make_actor",
    "Models",
]

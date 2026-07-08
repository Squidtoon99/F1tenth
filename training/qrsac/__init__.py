from .qrsac import QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor, Models

# NOTE: the replay buffers live in ``qrsac.replay`` and are imported explicitly
# (``from qrsac.replay import ReplayBuffer, TabledReplayBuffer``). They are part of
# the legacy distributed/Redis training path and pull in heavier deps (config,
# redis); the single-process QR-SAC trainer uses its own NStepReplayBuffer. Keeping
# them out of this package __init__ means ``import qrsac`` stays light.

__all__ = [
    "QRSACTrainer",
    "QuantileCritic",
    "SquashedGaussianMLPActor",
    "Models",
]

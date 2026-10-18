"""Isolated Gigaflow-inspired F1TENTH self-play package."""

from gigaflow_f1tenth.config import (
    ACTION_DIM,
    CNN_PROJECTION_DIM,
    CONTROL_HZ,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    PROPRIO_DIM,
    SENSOR_OBS_DIM,
    ExperimentConfig,
    load_config,
    validate_config,
)

__all__ = [
    "ACTION_DIM",
    "CNN_PROJECTION_DIM",
    "CONTROL_HZ",
    "ExperimentConfig",
    "GRU_HIDDEN_DIM",
    "LIDAR_DIM",
    "PROPRIO_DIM",
    "SENSOR_OBS_DIM",
    "load_config",
    "validate_config",
]

__version__ = "0.1.0"

"""
Experiments package aggregating reusable KAN components (bases, models,
inference algorithms, visualisation and training entry points).
"""

from . import config, bases, models, inference, targets, visualization, training

__all__ = [
    "config",
    "bases",
    "models",
    "inference",
    "targets",
    "visualization",
    "training",
]

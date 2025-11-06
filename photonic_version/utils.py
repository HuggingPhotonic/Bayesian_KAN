"""
Utility helpers shared by photonic training/inference scripts.
"""

from __future__ import annotations

import os
import torch


def get_device(preferred: str | None = None) -> torch.device:
    """
    Resolve a torch.device for computation.

    Order of precedence:
        1. Explicit ``preferred`` argument.
        2. ``PHOTONIC_DEVICE`` environment variable.
        3. CUDA if available, else CPU.
    """

    if preferred is not None:
        return torch.device(preferred)
    env_choice = os.environ.get("PHOTONIC_DEVICE")
    if env_choice:
        return torch.device(env_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


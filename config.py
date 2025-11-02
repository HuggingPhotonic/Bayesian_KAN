"""
Configuration helpers for experiments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import numpy as np


@dataclass
class ExperimentConfig:
    device: torch.device
    results_dir: Path
    seed: int = 42


def default_config(
    results_subdir: str = "default",
    *,
    device: Optional[str] = None,
    seed: int = 42,
) -> ExperimentConfig:
    if device in (None, "auto"):
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
    base = Path(__file__).resolve().parent / "results" / results_subdir
    base.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(device=device_obj, results_dir=base, seed=seed)
    set_seed(seed)
    return cfg


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

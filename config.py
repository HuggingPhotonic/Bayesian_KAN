"""
Configuration helpers for experiments.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np


@dataclass
class ExperimentConfig:
    device: torch.device
    results_dir: Path
    seed: int = 42


def default_config(results_subdir: str = "default") -> ExperimentConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(__file__).resolve().parent / "results" / results_subdir
    base.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(device=device, results_dir=base)
    set_seed(cfg.seed)
    return cfg


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

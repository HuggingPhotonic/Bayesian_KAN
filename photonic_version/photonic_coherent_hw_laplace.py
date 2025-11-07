"""
Laplace approximation inference for the coherent hardware photonic KAN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import LaplaceConfig, run_laplace
from photonic_version.photonic_coherent_hw_det import HardwareCoherentPhotonicKAN
from photonic_version.photonic_kan import target_function
from photonic_version.utils import get_device


def summarise_parameters(model: nn.Module) -> None:
    print("\n=== Trainable parameters ===")
    for name, param in model.named_parameters():
        print(f"{name:45s} {tuple(param.shape)}")


def run_laplace_inference() -> None:
    torch.manual_seed(42)
    device = get_device()

    n_train = 512
    X_train = torch.rand(n_train, 1, device=device) * 2 - 1
    y_train = target_function(X_train)

    layer_sizes: Sequence[int] = (1, 16, 16, 1)
    basis_kwargs = {
        "num_rings": 12,
        "wl_nm_range": (1546.0, 1554.0),
        "R_um": 30.0,
        "neff": 2.34,
        "ng": 4.2,
        "loss_dB_cm": 3.0,
        "kappa": 0.1,
    }

    model = HardwareCoherentPhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)

    config = LaplaceConfig(
        map_epochs=600,
        map_lr=5e-4,
        weight_decay=5e-5,
        noise_var=0.05,
        prior_var=1.0,
        damping=5e-3,
        n_samples=200,
    )

    X_eval = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    result = run_laplace(model, X_train, y_train, X_eval, config)

    output_dir = Path(__file__).resolve().parent / "results" / "hardware_coherent_laplace"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(result.map_losses.numpy(), label="MAP loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log posterior")
    plt.title("Laplace MAP optimisation (HW coherent)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_loss.png", dpi=150)
    plt.close()

    X_eval_np = X_eval.squeeze().cpu().numpy()
    mean = result.mean.squeeze().numpy()
    std = result.std.squeeze().numpy()
    truth = target_function(X_eval).squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(X_eval_np, truth, label="Target", linewidth=1.5)
    plt.plot(X_eval_np, mean, label="Posterior mean", linewidth=1.2)
    plt.fill_between(
        X_eval_np,
        mean - 2 * std,
        mean + 2 * std,
        color="tab:orange",
        alpha=0.25,
        label="±2σ",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laplace posterior (HW coherent photonic KAN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(f"\nLaplace approximation finished. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    run_laplace_inference()

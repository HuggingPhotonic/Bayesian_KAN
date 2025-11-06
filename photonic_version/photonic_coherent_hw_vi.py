"""
Variational (Bayesian) training for the coherent hardware photonic KAN.
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

from inference import VIConfig, train_vi
from photonic_version.hardware_bayes import BayesianHardwarePhotonicCoherentBasis
from photonic_version.photonic_kan import target_function
from photonic_version.utils import get_device


class BayesianHardwareCoherentPhotonicKAN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], *, basis_kwargs: dict):
        super().__init__()
        layers = []
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(BayesianHardwarePhotonicCoherentBasis(in_f, out_f, **basis_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        kl_total = torch.zeros(1, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, kl = layer(x, sample=sample, n_samples=n_samples)
            kl_total = kl_total + kl.to(x.device)
        return x, kl_total


def summarise_parameters(model: nn.Module) -> None:
    print("\n=== Trainable parameters ===")
    for name, param in model.named_parameters():
        print(f"{name:45s} {tuple(param.shape)}")


def run_variational_training() -> None:
    torch.manual_seed(42)
    device = get_device()

    n_train = 512
    X_train = torch.rand(n_train, 1, device=device) * 2 - 1
    y_train = target_function(X_train)

    layer_sizes = (1, 32, 32, 1)
    basis_kwargs = {
        "num_rings": 12,
        "wl_nm_range": (1546.0, 1554.0),
        "R_um": 30.0,
        "neff": 2.34,
        "ng": 4.2,
        "loss_dB_cm": 3.0,
        "kappa": 0.1,
    }

    model = BayesianHardwareCoherentPhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)

    config = VIConfig(
        epochs=2000,
        lr=5e-4,
        weight_decay=5e-5,
        kl_max=1e-3,
        kl_warmup_epochs=500,
        n_samples=4,
        grad_clip=1.0,
        eval_samples=200,
    )

    X_eval = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    result = train_vi(model, X_train, y_train, X_eval, config)

    output_dir = Path(__file__).resolve().parent / "results" / "hardware_coherent_bayes"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(result.losses.numpy(), label="ELBO")
    plt.plot(result.recon_losses.numpy(), label="Reconstruction")
    plt.plot(result.kl_terms.numpy(), label="KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Bayesian HW Photonic KAN (coherent) VI losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_losses.png", dpi=150)
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
    plt.title("Bayesian HW Photonic KAN (coherent) prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(f"\nFinished VI training. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    run_variational_training()

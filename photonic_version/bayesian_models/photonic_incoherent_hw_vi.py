"""
Variational (Bayesian) training for the incoherent hardware photonic KAN.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import VIConfig, train_vi
from photonic_version.bases.hardware import BayesianHardwarePhotonicIncoherentBasis
from photonic_version.photonic_kan import target_function
from photonic_version.utils import get_device


class BayesianHardwareIncoherentPhotonicKAN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], *, basis_kwargs: dict):
        super().__init__()
        layers = []
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(BayesianHardwarePhotonicIncoherentBasis(in_f, out_f, **basis_kwargs))
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

    layer_sizes = (1, 48, 48, 1)
    basis_kwargs = {
        "num_rings": 32,
        "wl_nm_range": (1546.0, 1554.0),
        "R_um": 30.0,
        "neff": 2.34,
        "ng": 4.2,
        "loss_dB_cm": 3.0,
        "kappa": 0.15,
    }

    model = BayesianHardwareIncoherentPhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)

    # Calmer VI settings: reduce LR/weight decay, slow down KL ramp, and use
    # more MC samples so that the variational VOA weights have time to
    # discover useful non-linear structure before the prior dominates.
    config = VIConfig(
        epochs=1800,
        lr=2e-4,
        weight_decay=5e-5,
        kl_max=0.25,
        kl_warmup_epochs=1400,
        n_samples=8,
        grad_clip=0.5,
        eval_samples=400,
    )

    X_eval = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    result = train_vi(model, X_train, y_train, X_eval, config)

    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(__file__).resolve().parent / "results" / script_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result.losses.numpy(), color="tab:blue")
    axes[0].set_ylabel("ELBO")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Bayesian HW Photonic KAN (incoherent) VI losses")

    axes[1].plot(result.recon_losses.numpy(), color="tab:green")
    axes[1].set_ylabel("Reconstruction")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.kl_terms.numpy(), color="tab:orange")
    axes[2].set_ylabel("KL")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_losses.png", dpi=150)
    plt.close(fig)

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
    plt.title("Bayesian HW Photonic KAN (incoherent) prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(f"\nFinished VI training. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    run_variational_training()

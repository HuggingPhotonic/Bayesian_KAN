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
        print(f"{name:40s} shape={tuple(param.shape)} requires_grad={param.requires_grad}")
    print()


def run_variational_training() -> None:
    torch.manual_seed(42)
    device = get_device()

    n_train = 2048
    X_train = torch.rand(n_train, 1, device=device) * 2 - 1
    y_train = target_function(X_train)

    layer_sizes = (1, 16, 16, 1)
    basis_kwargs = {
        "num_rings": 8,
        "wl_nm_range": (1546.0, 1554.0),
        "R_um": 30.0,
        "neff": 2.34,
        "ng": 4.2,
        "loss_dB_cm": 3.0,
        "kappa": 0.2,
    }

    model = BayesianHardwareCoherentPhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)

    def log_parameter_statistics(epoch: int) -> None:
        with torch.no_grad():
            theta_means = []
            phase_means = []
            base_means = []
            ring_phase_means = []
            ring_coupling_means = []
            for layer in model.layers:
                mixer = getattr(layer, "mzi_mixer", None)
                if mixer is None:
                    continue
                if hasattr(mixer, "theta_mean"):
                    theta_means.append(mixer.theta_mean.abs().mean().item())
                if hasattr(mixer, "phase"):
                    phase_means.append(mixer.phase.mean.abs().mean().item())
                base_means.append(layer.base_weight.abs().mean().item())
                if hasattr(layer, "wg_phase_mean"):
                    ring_phase_means.append(layer.wg_phase_mean.abs().mean().item())
                if hasattr(layer, "dc_coupling_mean"):
                    ring_coupling_means.append(layer.dc_coupling_mean.abs().mean().item())
            theta_avg = sum(theta_means) / len(theta_means) if theta_means else 0.0
            phase_avg = sum(phase_means) / len(phase_means) if phase_means else 0.0
            base_avg = sum(base_means) / len(base_means) if base_means else 0.0
            ring_phase_avg = sum(ring_phase_means) / len(ring_phase_means) if ring_phase_means else 0.0
            ring_coupling_avg = sum(ring_coupling_means) / len(ring_coupling_means) if ring_coupling_means else 0.0
            print(
                f"[Epoch {epoch}] |theta|={theta_avg:.4f} |phase|={phase_avg:.4f} "
                f"|base|={base_avg:.4f} |ring_phase|={ring_phase_avg:.4f} "
                f"|ring_coupling_raw|={ring_coupling_avg:.4f}"
            )

    config = VIConfig(
        epochs=2000,           # Increased from 1000
        lr=5e-4,               # Increased from 1.5e-4
        weight_decay=5e-5,
        kl_max=0.1,            # Increased from 1e-2 to 0.1
        kl_warmup_epochs=400,  # Increased from 200
        n_samples=50,
        grad_clip=1.0,
        eval_samples=200,
    )

    X_eval = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    result = train_vi(model, X_train, y_train, X_eval, config, callback=log_parameter_statistics)

    output_dir = Path(__file__).resolve().parent / "results" / "hardware_coherent_bayes"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result.losses.numpy(), color="tab:blue")
    axes[0].set_ylabel("ELBO")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Bayesian HW Photonic KAN (coherent) VI losses")

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
    plt.title("Bayesian HW Photonic KAN (coherent) prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(f"\nFinished VI training. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    run_variational_training()

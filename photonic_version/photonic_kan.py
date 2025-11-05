"""
Demo training script for a photonic KAN using Photontorch microring bases.

Trainable parameters:
    * Per-layer linear mixing coefficients (`coeffs` or `coeff_mean/log_var`)
    * Per-layer residual weights (`base_weight`)

Fixed device parameters (buffers):
    * Ring radius, effective/group indices, propagation loss, coupling ratio
    * Phase offsets (translated into effective-index perturbations)
    * Wavelength normalisation range
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from tqdm.auto import trange
import torch
import torch.nn as nn

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photonic_version.basis import PhotonicTorchBasis


def target_function(x: torch.Tensor) -> torch.Tensor:
    """1D benchmark: sin(3x) + 0.3*cos(10x)."""
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)


class PhotonicKAN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], *, basis_kwargs: dict):
        super().__init__()
        layers = []
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(PhotonicTorchBasis(in_f, out_f, **basis_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kl_total = torch.zeros(1, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, kl = layer(x, sample=False)
            kl_total = kl_total + kl.to(x.device)
        self._last_kl = kl_total
        return x

    @property
    def last_kl(self) -> torch.Tensor:
        return getattr(self, "_last_kl", torch.zeros(1))


def summarise_parameters(model: nn.Module) -> None:
    print("\n=== Trainable parameters ===")
    for name, param in model.named_parameters():
        print(f"{name:40s} shape={tuple(param.shape)} requires_grad={param.requires_grad}")

    print("\n=== Fixed buffers (device parameters) ===")
    for name, buffer in model.named_buffers():
        print(f"{name:40s} shape={tuple(buffer.shape)} requires_grad=False")
    print()


def train_photonic_kan() -> None:
    torch.manual_seed(42)

    device = torch.device("cpu")  # Photontorch components operate on CPU

    n_train = 512
    X_train = torch.rand(n_train, 1, device=device) * 2 - 1  # [-1, 1]
    y_train = target_function(X_train)

    layer_sizes = (1, 32, 32, 1)
    basis_kwargs = {
        "num_rings": 32,
        "wl_nm_range": (1546.0, 1554.0),
        "mixing": "coherent",
        "residual": False,
    }

    model = PhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=5e-5)
    mse_loss = nn.MSELoss()

    epochs = 500
    losses = []

    progress = trange(1, epochs + 1, desc="Training Photonic KAN", leave=True)
    grad_reported = False
    for epoch in progress:
        optimizer.zero_grad()
        preds = model(X_train)
        loss = mse_loss(preds, y_train)
        loss.backward()

        if not grad_reported:
            missing = []
            small = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    missing.append(name)
                else:
                    norm = param.grad.data.norm().item()
                    if norm < 1e-8:
                        small.append((name, norm))
            print("\nFirst backward pass gradient diagnostics:")
            if missing:
                print("  No grad:", ", ".join(missing))
            if small:
                print("  Very small grad:", ", ".join(f"{n}({norm:.2e})" for n, norm in small))
            if not missing and not small:
                print("  All parameters receive meaningful gradients.")
            grad_reported = True

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    # Evaluation grid
    grid = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    with torch.no_grad():
        preds_grid = model(grid)
        truth_grid = target_function(grid)

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Photonic KAN training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(grid.squeeze().cpu().numpy(), truth_grid.squeeze().cpu().numpy(), label="Target", linewidth=1.5)
    plt.plot(grid.squeeze().cpu().numpy(), preds_grid.squeeze().cpu().numpy(), label="Photonic KAN", linewidth=1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Photonic KAN fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(f"\nFinished training. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    train_photonic_kan()

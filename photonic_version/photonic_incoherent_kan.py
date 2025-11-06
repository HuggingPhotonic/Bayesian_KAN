"""
Training entry-point for an incoherent photonic KAN (power-domain basis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys

import matplotlib.pyplot as plt
from tqdm.auto import trange
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photonic_version.basic_function2 import PhotonicIncoherentBasis
from photonic_version.photonic_kan import target_function
from photonic_version.utils import get_device


class IncoherentPhotonicKAN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], *, basis_kwargs: dict):
        super().__init__()
        layers = []
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Reuse the intensity-only basis for each hidden transformation.
            layers.append(PhotonicIncoherentBasis(in_f, out_f, **basis_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        kl_total = torch.zeros(1, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            # Incoherent basis shares the same (output, KL) contract.
            x, kl = layer(x, sample=sample)
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


def train_incoherent_kan() -> None:
    torch.manual_seed(42)

    device = get_device()

    n_train = 512
    # Random samples on [-1, 1]; viewed as wavelength control inputs.
    X_train = torch.rand(n_train, 1, device=device) * 2 - 1
    y_train = target_function(X_train)

    # Mirror the coherent experiment but with power-domain basis layers.
    layer_sizes = (1, 32, 32, 1)
    basis_kwargs = {
        "num_rings": 24,
        "wl_nm_range": (1546.0, 1554.0),
        "R_um": 30.0,
        "neff": 2.34,
        "ng": 4.2,
        "loss_dB_cm": 3.0,
        "kappa": 0.2,
        "variational": False,
    }

    model = IncoherentPhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=5e-5)
    mse_loss = nn.MSELoss()

    epochs = 2000
    losses = []

    progress = trange(1, epochs + 1, desc="Training Incoherent Photonic KAN", leave=True)
    for epoch in progress:
        optimizer.zero_grad()
        # Forward pass through power-domain photonic basis layers.
        preds = model(X_train)
        loss = mse_loss(preds, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    # Evaluate on a dense grid for visualisation.
    grid = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    with torch.no_grad():
        preds_grid = model(grid)
        truth_grid = target_function(grid)

    output_dir = Path(__file__).resolve().parent / "results" / "incoherent"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot training loss over time.
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Photonic KAN (incoherent) training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    # Plot fitted function versus the analytic target.
    plt.figure(figsize=(10, 4))
    x_plot = grid.squeeze().cpu().numpy()
    truth_plot = truth_grid.squeeze().cpu().numpy()
    pred_plot = preds_grid.squeeze().cpu().numpy()
    plt.plot(x_plot, truth_plot, label="Target", linewidth=1.5)
    plt.plot(x_plot, pred_plot, label="Incoherent Photonic KAN", linewidth=1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Photonic KAN fit (incoherent)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(f"\nFinished training. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    train_incoherent_kan()

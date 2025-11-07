"""
Hamiltonian Monte Carlo inference for the coherent hardware photonic KAN.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence
import sys

import matplotlib.pyplot as plt
from tqdm.auto import trange
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import HMCConfig, run_hmc
from photonic_version.deterministic_models.photonic_coherent_hw_det import (
    HardwareCoherentPhotonicKAN,
)
from photonic_version.photonic_kan import target_function
from photonic_version.utils import get_device


def summarise_parameters(model: nn.Module) -> None:
    print("\n=== Trainable parameters ===")
    for name, param in model.named_parameters():
        print(f"{name:45s} {tuple(param.shape)}")


def run_hmc_inference() -> None:
    torch.manual_seed(42)
    device = get_device()

    n_train = 2048
    X_train = torch.rand(n_train, 1, device=device) * 2 - 1
    y_train = target_function(X_train)

    layer_sizes: Sequence[int] = (1, 16, 16, 1)
    basis_kwargs = {
        "num_rings": 8,
        "wl_nm_range": (1546.0, 1554.0),
        "R_um": 30.0,
        "neff": 2.34,
        "ng": 4.2,
        "loss_dB_cm": 3.0,
        "kappa": 0.1,
    }

    model = HardwareCoherentPhotonicKAN(layer_sizes, basis_kwargs=basis_kwargs).to(device)
    summarise_parameters(model)

    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(__file__).resolve().parent / "results" / script_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic MAP training stage
    map_epochs = 1000
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=map_epochs, eta_min=5e-5
    )
    mse_loss = nn.MSELoss()
    losses = []
    progress = trange(1, map_epochs + 1, desc="HMC MAP training", leave=True)
    for epoch in progress:
        optimizer.zero_grad()
        preds = model(X_train)
        loss = mse_loss(preds, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    with torch.no_grad():
        X_eval_temp = torch.linspace(-1, 1, 100, device=device).unsqueeze(-1)
        y_eval_temp = model(X_eval_temp)
        y_true_temp = target_function(X_eval_temp)
        map_mse = nn.MSELoss()(y_eval_temp, y_true_temp).item()
        print(f"MAP test MSE: {map_mse:.6f}")

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("HMC MAP training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_training_loss.png", dpi=150)
    plt.close()

    with torch.no_grad():
        grid_map = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
        preds_map = model(grid_map)
        truth_map = target_function(grid_map)
    plt.figure(figsize=(10, 4))
    plt.plot(grid_map.squeeze().cpu().numpy(), truth_map.squeeze().cpu().numpy(), label="Target", linewidth=1.5)
    plt.plot(grid_map.squeeze().cpu().numpy(), preds_map.squeeze().cpu().numpy(), label="MAP fit", linewidth=1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("HMC MAP fit (coherent)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_prediction.png", dpi=150)
    plt.close()

    config = HMCConfig(
        n_samples=400,
        burn_in=300,
        step_size=5e-3,
        n_leapfrog=25,
        noise_var=0.01,
        prior_var=5.0,
    )

    X_eval = torch.linspace(-1, 1, 800, device=device).unsqueeze(-1)
    result = run_hmc(model, X_train, y_train, X_eval, config)

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
    plt.title("HMC posterior (HW coherent photonic KAN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction.png", dpi=150)
    plt.close()

    print(
        f"\nHMC finished. Acceptance rate={result.acceptance_rate:.3f}. "
        f"Artefacts saved to {output_dir}"
    )


if __name__ == "__main__":
    run_hmc_inference()

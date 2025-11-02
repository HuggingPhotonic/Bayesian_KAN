"""
Refactored 2D KAN MCMC experimentation playground.

This module offers a modular scaffold for comparing multiple MCMC samplers
on a deterministic KAN regression model. It currently includes Random-Walk
Metropolis (RWM) and Hamiltonian Monte Carlo (HMC) implementations and can
be extended with additional samplers (e.g., Gibbs, NUTS).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import trange

from kan2d import KAN as DeterministicKAN2D
from kan2d import target_function

OUTPUT_DIR = Path(__file__).parent / "results_mcmc_2d_refactored"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SamplerResult:
    name: str
    samples: torch.Tensor
    acceptance_rate: float


def negative_log_posterior(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_var: float = 0.05,
    prior_var: float = 1.0,
) -> torch.Tensor:
    preds = model(x)
    sse = torch.sum((preds - y) ** 2)
    theta = parameters_to_vector(model.parameters())
    prior = torch.sum(theta ** 2) / prior_var
    return 0.5 * sse / noise_var + 0.5 * prior


def train_map(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 600,
    lr: float = 1e-3,
    noise_var: float = 0.05,
    prior_var: float = 1.0,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
    losses: List[float] = []
    progress = trange(epochs, desc="MAP Training 2D", leave=True)
    for epoch in progress:
        optimizer.zero_grad()
        loss = negative_log_posterior(model, x, y, noise_var, prior_var)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    return losses


def log_posterior_and_grad(
    model: nn.Module,
    theta_vec: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_var: float,
    prior_var: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vector_to_parameters(theta_vec, model.parameters())
    model.zero_grad()
    nlp = negative_log_posterior(model, x, y, noise_var, prior_var)
    grads = torch.autograd.grad(nlp, model.parameters())
    grad_vec = -parameters_to_vector(grads)
    return (-nlp).detach(), grad_vec.detach()


def sampler_random_walk_metropolis(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_samples: int = 400,
    burn_in: int = 200,
    step_size: float = 0.0015,
    noise_var: float = 0.05,
    prior_var: float = 1.0,
) -> SamplerResult:
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    current_lp, _ = log_posterior_and_grad(model, current, x, y, noise_var, prior_var)
    samples: List[torch.Tensor] = []
    accept = 0
    total = burn_in + n_samples
    for step in trange(total, desc="RWM 2D", leave=True):
        proposal = current + step_size * torch.randn_like(current)
        proposal_lp, _ = log_posterior_and_grad(model, proposal, x, y, noise_var, prior_var)
        log_alpha = proposal_lp - current_lp
        if torch.log(torch.rand(1)) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())
        if step >= burn_in:
            samples.append(current.clone())
    acceptance = accept / total
    return SamplerResult("RandomWalkMetropolis",
                         torch.stack(samples) if samples else torch.empty(0),
                         acceptance)


def sampler_hmc(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_samples: int = 400,
    burn_in: int = 200,
    step_size: float = 0.0015,
    n_leapfrog: int = 15,
    noise_var: float = 0.05,
    prior_var: float = 1.0,
) -> SamplerResult:
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    current_logp, current_grad = log_posterior_and_grad(model, current, x, y, noise_var, prior_var)
    samples: List[torch.Tensor] = []
    accept = 0
    total = burn_in + n_samples
    for step in trange(total, desc="HMC 2D", leave=True):
        theta = current.clone()
        grad = current_grad.clone()
        momentum = torch.randn_like(theta)
        current_H = -current_logp + 0.5 * torch.sum(momentum ** 2)

        theta_new = theta.clone()
        momentum_new = momentum.clone()
        momentum_new = momentum_new + 0.5 * step_size * grad
        for l in range(n_leapfrog):
            theta_new = theta_new + step_size * momentum_new
            logp_new, grad_new = log_posterior_and_grad(model, theta_new, x, y, noise_var, prior_var)
            if l != n_leapfrog - 1:
                momentum_new = momentum_new + step_size * grad_new
        momentum_new = momentum_new + 0.5 * step_size * grad_new
        momentum_new = -momentum_new

        new_H = -logp_new + 0.5 * torch.sum(momentum_new ** 2)
        log_alpha = -(new_H - current_H)
        if torch.log(torch.rand(1)) < log_alpha:
            current = theta_new.detach()
            current_logp = logp_new.detach()
            current_grad = grad_new.detach()
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())
        if step >= burn_in:
            samples.append(current.clone())
    acceptance = accept / total
    return SamplerResult("HamiltonianMonteCarlo",
                         torch.stack(samples) if samples else torch.empty(0),
                         acceptance)


SAMPLER_REGISTRY: Dict[str, Callable[..., SamplerResult]] = {
    "rwm": sampler_random_walk_metropolis,
    "hmc": sampler_hmc,
}


def posterior_stats(
    model: nn.Module,
    sample_set: torch.Tensor,
    x_eval: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_set.numel() == 0:
        raise RuntimeError("Sampler returned no samples.")
    theta_ref = parameters_to_vector(model.parameters()).detach()
    preds = []
    with torch.no_grad():
        for sample in sample_set:
            vector_to_parameters(sample, model.parameters())
            preds.append(model(x_eval))
        vector_to_parameters(theta_ref, model.parameters())
    preds = torch.stack(preds)
    return preds.mean(0).cpu(), preds.std(0).cpu()


def visualise_surfaces(
    X_grid: torch.Tensor,
    Y_grid: torch.Tensor,
    Z_true: torch.Tensor,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    xg = X_grid.cpu()
    yg = Y_grid.cpu()
    fig = plt.figure(figsize=(18, 5 * len(stats)))
    ax = fig.add_subplot(len(stats) + 1, 1, 1, projection="3d")
    ax.plot_surface(xg, yg, Z_true, cmap="viridis", alpha=0.8)
    ax.set_title("Ground Truth")

    for idx, (name, (mean, std)) in enumerate(stats.items(), start=2):
        mean_np = mean.reshape(xg.shape)
        ax_mean = fig.add_subplot(len(stats) + 1, 2, 2 * idx - 2, projection="3d")
        ax_std = fig.add_subplot(len(stats) + 1, 2, 2 * idx - 1, projection="3d")
        ax_mean.plot_surface(xg, yg, mean_np, cmap="viridis", alpha=0.8)
        ax_mean.set_title(f"{name} Mean")
        ax_std.plot_surface(xg, yg, std.reshape(xg.shape), cmap="hot", alpha=0.8)
        ax_std.set_title(f"{name} Std Dev")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_2d_mcmc_surfaces_comparison.png", dpi=150, bbox_inches="tight")
    print("Surface comparison saved!")


def visualise_contours(
    X_grid: torch.Tensor,
    Y_grid: torch.Tensor,
    Z_true: torch.Tensor,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    xg = X_grid.cpu()
    yg = Y_grid.cpu()
    fig, axes = plt.subplots(len(stats) + 1, 2, figsize=(16, 5 * (len(stats) + 1)))
    c1 = axes[0, 0].contourf(xg, yg, Z_true, levels=20, cmap="viridis")
    axes[0, 0].set_title("Ground Truth")
    plt.colorbar(c1, ax=axes[0, 0])
    axes[0, 1].axis("off")

    for idx, (name, (mean, std)) in enumerate(stats.items(), start=1):
        mean_np = mean.reshape(xg.shape)
        std_np = std.reshape(xg.shape)
        cm = axes[idx, 0].contourf(xg, yg, mean_np, levels=20, cmap="viridis")
        axes[idx, 0].set_title(f"{name} Mean")
        plt.colorbar(cm, ax=axes[idx, 0])
        cs = axes[idx, 1].contourf(xg, yg, std_np, levels=20, cmap="hot")
        axes[idx, 1].set_title(f"{name} Std Dev")
        plt.colorbar(cs, ax=axes[idx, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_2d_mcmc_contours_comparison.png", dpi=150, bbox_inches="tight")
    print("Contour comparison saved!")


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = 2000
    x_train = torch.rand(n_train, 1) * 4 - 2
    y_train = torch.rand(n_train, 1) * 4 - 2
    X_train = torch.cat([x_train, y_train], dim=1).to(device)
    z_train = target_function(x_train, y_train).to(device)

    n_test = 60
    x_test = torch.linspace(-2, 2, n_test)
    y_test = torch.linspace(-2, 2, n_test)
    X_grid, Y_grid = torch.meshgrid(x_test, y_test, indexing="ij")
    X_eval = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1).to(device)
    Z_true = target_function(X_grid.to(device), Y_grid.to(device)).cpu()

    model = DeterministicKAN2D(layers_hidden=[2, 16, 16, 1], grid_size=8, spline_order=3).to(device)
    map_losses = train_map(model, X_train, z_train)

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Posterior")
    plt.title("MAP Training Loss (2D)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_2d_map_loss.png", dpi=150, bbox_inches="tight")

    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, sampler in SAMPLER_REGISTRY.items():
        print(f"\nRunning sampler: {key}")
        result = sampler(model, X_train, z_train)
        print(f"{result.name} acceptance rate: {result.acceptance_rate:.2%}")
        stats[result.name] = posterior_stats(model, result.samples, X_eval)

    visualise_surfaces(X_grid, Y_grid, Z_true, stats)
    visualise_contours(X_grid, Y_grid, Z_true, stats)

    for name, (mean, _) in stats.items():
        mse = torch.mean((mean.reshape(n_test, n_test) - Z_true) ** 2).item()
        mae = torch.mean(torch.abs(mean.reshape(n_test, n_test) - Z_true)).item()
        print(f"{name} -> MSE: {mse:.6f}, MAE: {mae:.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("KAN 2D MCMC Refactored Playground")
    print("=" * 60)

    run_experiment()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

"""
Refactored 1D KAN MCMC experimentation playground.

This module structures the workflow into modular components so that
different samplers (e.g. Metropolis-Hastings, Gibbs, NUTS) can be plugged in
and compared more easily. It reuses the deterministic KAN from kan1d.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import trange

from kan1d import DeterministicKAN1D, target_function

OUTPUT_DIR = Path(__file__).parent / "results_mcmc_1d_refactored"
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
    progress = trange(epochs, desc="MAP Training 1D", leave=True)
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


def sampler_random_walk_metropolis(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_samples: int = 600,
    burn_in: int = 300,
    step_size: float = 0.005,
    noise_var: float = 0.05,
    prior_var: float = 1.0,
) -> SamplerResult:
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()

    def log_post(theta_vec):
        vector_to_parameters(theta_vec, model.parameters())
        with torch.no_grad():
            preds = model(x)
            sse = torch.sum((preds - y) ** 2)
            prior = torch.sum(theta_vec ** 2) / prior_var
            return -0.5 * sse / noise_var - 0.5 * prior

    current_lp = log_post(current)
    samples: List[torch.Tensor] = []
    accept = 0
    total = burn_in + n_samples
    for step in trange(total, desc="RWM 1D", leave=True):
        proposal = current + step_size * torch.randn_like(current)
        proposal_lp = log_post(proposal)
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
    step_size: float = 0.01,
    n_leapfrog: int = 15,
    noise_var: float = 0.05,
    prior_var: float = 1.0,
) -> SamplerResult:
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()

    def logp_and_grad(theta_vec):
        vector_to_parameters(theta_vec, model.parameters())
        model.zero_grad()
        nlp = negative_log_posterior(model, x, y, noise_var, prior_var)
        grads = torch.autograd.grad(nlp, model.parameters())
        grad_vec = -parameters_to_vector(grads)
        return (-nlp).detach(), grad_vec.detach()

    current_logp, current_grad = logp_and_grad(current)
    samples: List[torch.Tensor] = []
    accept = 0
    total = burn_in + n_samples
    for step in trange(total, desc="HMC 1D", leave=True):
        theta = current.clone()
        grad = current_grad.clone()
        momentum = torch.randn_like(theta)
        current_H = -current_logp + 0.5 * torch.sum(momentum ** 2)

        theta_new = theta.clone()
        momentum_new = momentum.clone()
        momentum_new = momentum_new + 0.5 * step_size * grad
        for l in range(n_leapfrog):
            theta_new = theta_new + step_size * momentum_new
            logp_new, grad_new = logp_and_grad(theta_new)
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
    return preds.mean(0).cpu().squeeze().numpy(), preds.std(0).cpu().squeeze().numpy()


def visualise_predictions(
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    x_test_np = x_test.cpu().squeeze().numpy()
    y_test_np = y_test.cpu().squeeze().numpy()
    x_train_np = x_train.cpu().squeeze().numpy()
    y_train_np = y_train.cpu().squeeze().numpy()

    plt.figure(figsize=(12, 6))
    for idx, (name, (mean, std)) in enumerate(results.items(), start=1):
        plt.subplot(len(results), 1, idx)
        plt.plot(x_test_np, y_test_np, label="Ground Truth", color="black", linewidth=2)
        plt.plot(x_test_np, mean, label=f"{name} Mean", linewidth=2)
        plt.fill_between(x_test_np, mean - 2 * std, mean + 2 * std, alpha=0.2)
        plt.scatter(x_train_np, y_train_np, s=10, alpha=0.3)
        plt.title(f"{name} Posterior Predictive")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_1d_mcmc_comparison.png", dpi=150, bbox_inches="tight")
    print("Posterior predictive comparison saved!")


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.linspace(-2, 2, 512).unsqueeze(-1).to(device)
    y_train = target_function(x_train).to(device)
    x_test = torch.linspace(-2.5, 2.5, 400).unsqueeze(-1).to(device)
    y_test = target_function(x_test.cpu()).to(device)

    model = DeterministicKAN1D(n_layers=2, n_basis=8, spline_order=3).to(device)
    map_losses = train_map(model, x_train, y_train)

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Posterior")
    plt.title("MAP Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_1d_map_loss.png", dpi=150, bbox_inches="tight")

    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, sampler in SAMPLER_REGISTRY.items():
        print(f"\nRunning sampler: {key}")
        result = sampler(model, x_train, y_train)
        print(f"{result.name} acceptance rate: {result.acceptance_rate:.2%}")
        stats[result.name] = posterior_stats(model, result.samples, x_test)

    visualise_predictions(x_test, y_test, x_train, y_train, stats)

    for name, (mean, std) in stats.items():
        y_np = y_test.cpu().squeeze().numpy()
        mse = float(np.mean((mean - y_np) ** 2))
        mae = float(np.mean(np.abs(mean - y_np)))
        print(f"{name} -> MSE: {mse:.6f}, MAE: {mae:.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("KAN 1D MCMC Refactored Playground")
    print("=" * 60)

    run_experiment()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

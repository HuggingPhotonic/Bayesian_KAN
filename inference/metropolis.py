from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import trange

from inference.common import (
    negative_log_posterior,
    posterior_stats,
)


@dataclass
class MetropolisConfig:
    n_samples: int = 800
    burn_in: int = 600
    step_size: float = 2e-4
    noise_var: float = 0.05
    prior_var: float = 1.0


@dataclass
class MCMCResult:
    samples: torch.Tensor
    acceptance_rate: float
    statistics: Dict[str, Any]
    mean: torch.Tensor
    std: torch.Tensor


def run_metropolis(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    config: MetropolisConfig,
) -> MCMCResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    map_progress = trange(500, desc="Metropolis MAP warmup", leave=False)
    map_losses = []
    for epoch in map_progress:
        optimizer.zero_grad()
        nlp = negative_log_posterior(
            model, X_train, y_train, config.noise_var, config.prior_var
        )
        nlp.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        map_losses.append(nlp.detach().cpu())
        map_progress.set_postfix(loss=nlp.item())

    plt.figure(figsize=(8, 3))
    plt.plot(torch.stack(map_losses).numpy())
    plt.xlabel("Epoch")
    plt.ylabel("Negative log posterior")
    plt.title("Metropolis MAP warmup")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    map_plot = plt.gcf()

    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    with torch.no_grad():
        current_lp = -negative_log_posterior(
            model, X_train, y_train, config.noise_var, config.prior_var
        )
    samples = []
    accept = 0
    total = config.burn_in + config.n_samples

    for step in trange(total, desc="Metropolis", leave=True):
        proposal = current + config.step_size * torch.randn_like(current)
        vector_to_parameters(proposal, model.parameters())
        with torch.no_grad():
            proposal_lp = -negative_log_posterior(
                model, X_train, y_train, config.noise_var, config.prior_var
            )
        log_alpha = proposal_lp - current_lp
        if torch.log(torch.rand(1, device=current.device)) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())
        if step >= config.burn_in:
            samples.append(current.clone())

    acceptance_rate = accept / total
    sample_tensor = torch.stack(samples) if samples else torch.empty(0)
    mean, std = posterior_stats(model, sample_tensor, X_eval)
    output_dir = Path("photonic_version/results/hardware_coherent_mcmc")
    output_dir.mkdir(parents=True, exist_ok=True)
    map_plot.savefig(output_dir / "map_warmup.png", dpi=150)
    plt.close(map_plot)
    return MCMCResult(
        samples=sample_tensor,
        acceptance_rate=acceptance_rate,
        statistics={
            "steps": total,
            "burn_in": config.burn_in,
            "step_size": config.step_size,
        },
        mean=mean.cpu(),
        std=std.cpu(),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import trange

from inference.common import (
    negative_log_posterior,
    log_posterior_and_grad,
    posterior_stats,
)


@dataclass
class HMCConfig:
    n_samples: int = 800
    burn_in: int = 600
    step_size: float = 5e-4
    n_leapfrog: int = 30
    noise_var: float = 0.05
    prior_var: float = 1.0


@dataclass
class HMCResult:
    samples: torch.Tensor
    acceptance_rate: float
    statistics: Dict[str, Any]
    mean: torch.Tensor
    std: torch.Tensor


def run_hmc(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    config: HMCConfig,
) -> HMCResult:
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    current_logp, current_grad = log_posterior_and_grad(
        model, current, X_train, y_train, config.noise_var, config.prior_var
    )

    samples = []
    accept = 0
    total = config.burn_in + config.n_samples

    for step in trange(total, desc="HMC", leave=True):
        theta = current.clone()
        grad = current_grad.clone()
        momentum = torch.randn_like(theta)
        current_H = -current_logp + 0.5 * torch.sum(momentum**2)

        theta_new = theta.clone()
        momentum_new = momentum.clone()
        momentum_new = momentum_new + 0.5 * config.step_size * grad
        for l in range(config.n_leapfrog):
            theta_new = theta_new + config.step_size * momentum_new
            logp_new, grad_new = log_posterior_and_grad(
                model,
                theta_new,
                X_train,
                y_train,
                config.noise_var,
                config.prior_var,
            )
            if l != config.n_leapfrog - 1:
                momentum_new = momentum_new + config.step_size * grad_new
        momentum_new = momentum_new + 0.5 * config.step_size * grad_new
        momentum_new = -momentum_new

        new_H = -logp_new + 0.5 * torch.sum(momentum_new**2)
        log_alpha = -(new_H - current_H)
        if torch.log(torch.rand(1, device=current.device)) < log_alpha:
            current = theta_new.detach()
            current_logp = logp_new.detach()
            current_grad = grad_new.detach()
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())

        if step >= config.burn_in:
            samples.append(current.clone())

    acceptance_rate = accept / total
    sample_tensor = torch.stack(samples) if samples else torch.empty(0)
    mean, std = posterior_stats(model, sample_tensor, X_eval)
    return HMCResult(
        samples=sample_tensor,
        acceptance_rate=acceptance_rate,
        statistics={
            "steps": total,
            "burn_in": config.burn_in,
            "step_size": config.step_size,
            "n_leapfrog": config.n_leapfrog,
        },
        mean=mean.cpu(),
        std=std.cpu(),
    )


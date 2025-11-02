from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import trange

from inference.common import (
    negative_log_posterior,
    posterior_stats,
)


@dataclass
class LaplaceConfig:
    map_epochs: int = 500
    map_lr: float = 5e-4
    weight_decay: float = 2e-4
    noise_var: float = 0.05
    prior_var: float = 1.0
    damping: float = 1e-2
    n_samples: int = 200
    grad_clip: float = 1.0


@dataclass
class LaplaceResult:
    map_losses: torch.Tensor
    samples: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    statistics: Dict[str, Any]


def train_map(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: LaplaceConfig,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.map_lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    losses = []
    progress = trange(config.map_epochs, desc="Laplace MAP", leave=True)
    for epoch in progress:
        optimizer.zero_grad()
        nlp = negative_log_posterior(
            model, X_train, y_train, config.noise_var, config.prior_var
        )
        nlp.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(nlp.detach().cpu())
        progress.set_postfix(
            loss=nlp.detach().item(),
            lr=optimizer.param_groups[0]["lr"],
        )
    return torch.stack(losses)


def hessian_diag(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: LaplaceConfig,
) -> torch.Tensor:
    loss = negative_log_posterior(
        model, X_train, y_train, config.noise_var, config.prior_var
    )
    params = [p for p in model.parameters()]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = parameters_to_vector(grads)
    diag = []
    for i in range(grad_vec.numel()):
        comp = grad_vec[i]
        second = torch.autograd.grad(comp, params, retain_graph=True)
        diag.append(parameters_to_vector(second)[i])
    return torch.stack(diag).detach()


def laplace_samples(
    model: torch.nn.Module,
    theta_map: torch.Tensor,
    var_diag: torch.Tensor,
    X_eval: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    preds = []
    std = torch.sqrt(var_diag.clamp(min=1e-6))
    with torch.no_grad():
        for _ in range(n_samples):
            sample_vec = theta_map + torch.randn_like(theta_map) * std
            vector_to_parameters(sample_vec, model.parameters())
            output = model(X_eval)
            if isinstance(output, tuple):
                output = output[0]
            preds.append(output)
        vector_to_parameters(theta_map, model.parameters())
    return torch.stack(preds)


def run_laplace(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    config: LaplaceConfig,
) -> LaplaceResult:
    map_losses = train_map(model, X_train, y_train, config)
    theta_map = parameters_to_vector(model.parameters()).detach()
    diag = hessian_diag(model, X_train, y_train, config)
    var_diag = 1.0 / (diag + config.damping)
    preds = laplace_samples(model, theta_map, var_diag, X_eval, config.n_samples)
    mean, std = preds.mean(0).cpu(), preds.std(0).cpu()
    return LaplaceResult(
        map_losses=map_losses,
        samples=preds.cpu(),
        mean=mean,
        std=std,
        statistics={"damping": config.damping, "n_samples": config.n_samples},
    )

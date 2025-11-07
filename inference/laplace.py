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
    var_clip: float = 1.0


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
    if config.map_epochs <= 0:
        return torch.empty(0)
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
    """
    Compute diagonal Hessian approximation using empirical Fisher information.
    More stable and efficient than exact diagonal Hessian computation.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    fisher_diag = torch.zeros(n_params, device=X_train.device)

    # Use mini-batches for memory efficiency
    batch_size = min(256, len(X_train))
    n_batches = (len(X_train) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]

        model.zero_grad()
        loss = negative_log_posterior(
            model, X_batch, y_batch, config.noise_var, config.prior_var
        )
        grads = torch.autograd.grad(loss, params, create_graph=False)
        grad_vec = parameters_to_vector(grads)
        fisher_diag += grad_vec ** 2

    fisher_diag /= n_batches
    model.train()
    return fisher_diag.detach()


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
    var_diag = torch.clamp(var_diag, min=1e-8, max=config.var_clip)
    preds = laplace_samples(model, theta_map, var_diag, X_eval, config.n_samples)
    mean, std = preds.mean(0).cpu(), preds.std(0).cpu()
    return LaplaceResult(
        map_losses=map_losses,
        samples=preds.cpu(),
        mean=mean,
        std=std,
        statistics={"damping": config.damping, "n_samples": config.n_samples},
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn
from tqdm.auto import trange


@dataclass
class VIConfig:
    epochs: int = 2000
    lr: float = 5e-4
    weight_decay: float = 1e-4
    kl_max: float = 1e-3
    kl_warmup_epochs: int = 300
    scheduler_step: int = 300
    scheduler_gamma: float = 0.5
    n_samples: int = 8
    grad_clip: float = 1.0
    eval_samples: int = 200


@dataclass
class VIResult:
    model: torch.nn.Module
    losses: torch.Tensor
    recon_losses: torch.Tensor
    kl_terms: torch.Tensor
    history: Dict[str, Any]
    mean: torch.Tensor
    std: torch.Tensor


def train_vi(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    config: VIConfig,
) -> VIResult:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    mse_loss = nn.MSELoss()

    losses = []
    recon_losses = []
    kl_terms = []

    progress = trange(config.epochs, desc="VI Training", leave=True)
    for epoch in progress:
        model.train()
        optimizer.zero_grad()

        preds, kl = model(X_train, sample=True, n_samples=config.n_samples)
        recon = mse_loss(preds, y_train)
        kl_weight = config.kl_max * min(1.0, (epoch + 1) / config.kl_warmup_epochs)
        loss = recon + kl_weight * kl / X_train.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        losses.append(loss.detach().cpu())
        recon_losses.append(recon.detach().cpu())
        kl_terms.append(kl.detach().cpu())
        progress.set_postfix(
            loss=float(loss),
            recon=float(recon),
            kl=float(kl),
            beta=kl_weight,
            lr=optimizer.param_groups[0]["lr"],
        )

    model.eval()
    with torch.no_grad():
        preds = []
        for _ in range(config.eval_samples):
            pred, _ = model(X_eval, sample=True, n_samples=1)
            preds.append(pred)
        stacked = torch.stack(preds)
        mean = stacked.mean(0)
        std = stacked.std(0)

    return VIResult(
        model=model,
        losses=torch.stack(losses),
        recon_losses=torch.stack(recon_losses),
        kl_terms=torch.stack(kl_terms),
        history={
            "kl_max": config.kl_max,
            "kl_warmup_epochs": config.kl_warmup_epochs,
            "eval_samples": config.eval_samples,
        },
        mean=mean.cpu(),
        std=std.cpu(),
    )

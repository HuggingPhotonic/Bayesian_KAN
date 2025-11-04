"""
Parameter noise robustness experiment using Metropolis MCMC inference.

This script mirrors the deterministic vs Bayesian comparison but swaps the
variational model for a Metropolis sampler.  It trains a deterministic KAN and
an MCMC-backed KAN (with MAP warm-up), injects Gaussian noise into parameters,
and records/visualises how predictions degrade under perturbations.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn.utils import vector_to_parameters

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import set_seed
from inference.laplace import LaplaceConfig, train_map
from inference.metropolis import MetropolisConfig, run_metropolis
from models.kan import build_kan
from noise_in_parameters import (
    DatasetBundle,
    NoiseExperimentSettings,
    SweepEntry,
    _parameter_scale,
    _resolve_device,
    make_dataset,
    run_noise_sweep,
)
from training.trainer import DeterministicConfig, train_deterministic


@dataclass
class MCMCSettings(NoiseExperimentSettings):
    """Extension of the base settings with Metropolis-specific knobs."""

    results_folder: str = "parameter_noise_mcmc"
    metropolis_samples: int = 800
    metropolis_burn_in: int = 600
    metropolis_step_size: float = 2e-4
    mcmc_noise_var: float = 0.05
    mcmc_prior_var: float = 1.0
    mcmc_eval_samples: int = 200
    map_epochs: int = 800
    map_lr: float = 5e-4
    map_weight_decay: float = 2e-4


def _plot_mse_curve_mcmc(
    deterministic: Sequence[SweepEntry],
    mcmc: Sequence[SweepEntry],
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    det_noise = [entry.noise_std for entry in deterministic]
    det_mse = [entry.metrics["mse"] for entry in deterministic]
    mcmc_noise = [entry.noise_std for entry in mcmc]
    mcmc_mse = [entry.metrics["mse"] for entry in mcmc]
    plt.plot(det_noise, det_mse, marker="o", label="Deterministic")
    plt.plot(mcmc_noise, mcmc_mse, marker="^", label="Metropolis MCMC")
    plt.xlabel("Parameter noise std (relative)")
    plt.ylabel("MSE on evaluation grid")
    plt.title("Prediction error under parameter noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_predictions_1d_mcmc(
    bundle: DatasetBundle,
    deterministic: SweepEntry,
    mcmc_entry: SweepEntry,
    title: str,
    output_path: Path,
) -> None:
    x = bundle.X_eval.squeeze(-1).cpu().numpy()
    target = bundle.y_eval.squeeze(-1).cpu().numpy()
    det_mean = deterministic.mean.squeeze(-1).numpy()
    mcmc_mean = mcmc_entry.mean.squeeze(-1).numpy()
    mcmc_std = mcmc_entry.std.squeeze(-1).numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, target, label="Target", color="black", linewidth=1.5)
    plt.plot(x, det_mean, label="Deterministic mean", color="#1f77b4", linewidth=1.2)
    plt.plot(x, mcmc_mean, label="MCMC mean", color="#2ca02c", linewidth=1.2)
    plt.fill_between(
        x,
        mcmc_mean - 2 * mcmc_std,
        mcmc_mean + 2 * mcmc_std,
        color="#2ca02c",
        alpha=0.2,
        label="MCMC ±2σ",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _subset_samples(samples: torch.Tensor, max_samples: int) -> torch.Tensor:
    if samples.numel() == 0:
        raise ValueError("MCMC produced no samples; cannot evaluate robustness.")
    if samples.shape[0] <= max_samples:
        return samples
    return samples[:max_samples]


def evaluate_mcmc_with_noise(
    template: torch.nn.Module,
    samples: torch.Tensor,
    bundle: DatasetBundle,
    noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    preds: List[torch.Tensor] = []
    for vec in samples:
        vector_to_parameters(vec.clone(), template.parameters())
        if noise_std > 0:
            for name, param in template.named_parameters():
                if "log_var" in name:
                    continue
                scale = _parameter_scale(param)
                param.data.add_(torch.randn_like(param) * noise_std * scale)
        output, _ = template(bundle.X_eval, sample=False)
        preds.append(output.detach().cpu())

    stacked = torch.stack(preds)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    target = bundle.y_eval.detach().cpu()
    errors = mean - target
    mse = float(torch.mean(errors ** 2).item())
    mae = float(torch.mean(errors.abs()).item())
    max_abs = float(torch.max(errors.abs()).item())
    metrics = {"mse": mse, "mae": mae, "max_abs_err": max_abs}
    return mean, std, metrics


def main() -> None:
    settings = MCMCSettings()
    device = _resolve_device(settings.device)
    set_seed(settings.seed)

    output_dir = Path(__file__).resolve().parent / "results" / settings.results_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = make_dataset(settings, device)

    # Deterministic baseline
    det_model = build_kan(
        input_dim=bundle.X_train.shape[1],
        basis_name="bspline",
        layer_sizes=settings.layer_sizes,
        variational=False,
        **settings.basis_kwargs,
    ).to(device)
    det_config = DeterministicConfig(
        epochs=1500,
        lr=5e-4,
        weight_decay=1e-4,
        scheduler_step=400,
        scheduler_gamma=0.5,
        grad_clip=1.0,
    )
    det_losses = train_deterministic(det_model, bundle.X_train, bundle.y_train, det_config)
    det_sweep = run_noise_sweep(
        det_model,
        bundle,
        settings.param_noise_levels,
        variational=False,
        eval_samples=1,
        device=device,
    )

    # MCMC model with MAP warm-up
    mcmc_model = build_kan(
        input_dim=bundle.X_train.shape[1],
        basis_name="bspline",
        layer_sizes=settings.layer_sizes,
        variational=False,
        **settings.basis_kwargs,
    ).to(device)
    laplace_cfg = LaplaceConfig(
        map_epochs=settings.map_epochs,
        map_lr=settings.map_lr,
        weight_decay=settings.map_weight_decay,
        noise_var=settings.mcmc_noise_var,
        prior_var=settings.mcmc_prior_var,
        grad_clip=1.0,
    )
    map_losses = train_map(mcmc_model, bundle.X_train, bundle.y_train, laplace_cfg)
    map_state = {k: v.detach().clone() for k, v in mcmc_model.state_dict().items()}

    metropolis_cfg = MetropolisConfig(
        n_samples=settings.metropolis_samples,
        burn_in=settings.metropolis_burn_in,
        step_size=settings.metropolis_step_size,
        noise_var=settings.mcmc_noise_var,
        prior_var=settings.mcmc_prior_var,
    )
    mcmc_result = run_metropolis(
        mcmc_model,
        bundle.X_train,
        bundle.y_train,
        bundle.X_eval,
        metropolis_cfg,
    )
    samples = _subset_samples(
        mcmc_result.samples.to(device),
        settings.mcmc_eval_samples,
    )

    eval_model = build_kan(
        input_dim=bundle.X_train.shape[1],
        basis_name="bspline",
        layer_sizes=settings.layer_sizes,
        variational=False,
        **settings.basis_kwargs,
    ).to(device)
    eval_model.load_state_dict(map_state)

    mcmc_sweep: List[SweepEntry] = []
    for level in settings.param_noise_levels:
        eval_model.load_state_dict(map_state)
        mean, std, metrics = evaluate_mcmc_with_noise(
            eval_model,
            samples,
            bundle,
            float(level),
        )
        mcmc_sweep.append(
            SweepEntry(noise_std=float(level), mean=mean, std=std, metrics=metrics)
        )

    _plot_mse_curve_mcmc(det_sweep, mcmc_sweep, output_dir / "mse_vs_parameter_noise.png")
    if bundle.X_eval.shape[1] == 1:
        baseline_det = det_sweep[0]
        baseline_mcmc = mcmc_sweep[0]
        worst_det = det_sweep[-1]
        worst_mcmc = mcmc_sweep[-1]
        _plot_predictions_1d_mcmc(
            bundle,
            baseline_det,
            baseline_mcmc,
            title="Predictions without parameter noise",
            output_path=output_dir / "predictions_baseline.png",
        )
        _plot_predictions_1d_mcmc(
            bundle,
            worst_det,
            worst_mcmc,
            title=f"Predictions with parameter noise std={settings.param_noise_levels[-1]:.2f}",
            output_path=output_dir / "predictions_noisy.png",
        )

    # Training/metropolis diagnostics
    plt.figure(figsize=(7, 4))
    plt.plot(det_losses.cpu().numpy())
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Deterministic training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "deterministic_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(map_losses.cpu().numpy())
    plt.xlabel("Epoch")
    plt.ylabel("Negative log posterior")
    plt.title("MAP warm-up (Metropolis)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_warmup_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "settings": {
            "target": settings.target,
            "dim": settings.dim,
            "n_train": settings.n_train,
            "train_noise_std": settings.train_noise_std,
            "layer_sizes": list(settings.layer_sizes),
            "basis_kwargs": dict(settings.basis_kwargs),
            "param_noise_levels": list(settings.param_noise_levels),
            "metropolis": {
                "n_samples": settings.metropolis_samples,
                "burn_in": settings.metropolis_burn_in,
                "step_size": settings.metropolis_step_size,
                "noise_var": settings.mcmc_noise_var,
                "prior_var": settings.mcmc_prior_var,
            },
        },
        "deterministic": [
            {"noise_std": entry.noise_std, **entry.metrics} for entry in det_sweep
        ],
        "mcmc": [
            {"noise_std": entry.noise_std, **entry.metrics} for entry in mcmc_sweep
        ],
        "mcmc_acceptance_rate": float(mcmc_result.acceptance_rate),
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Results saved under {output_dir}")
    print(f"Metric summary: {metrics_path}")


if __name__ == "__main__":
    main()

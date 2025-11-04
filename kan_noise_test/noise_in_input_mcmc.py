"""
Input noise robustness experiment using Metropolis MCMC inference.

This script mirrors the deterministic vs VI comparison but replaces the
variational model with a MAP-warm-started Metropolis sampler to study how
posterior averaging copes with input perturbations.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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
from targets import resolve_target
from training.trainer import DeterministicConfig, train_deterministic
from noise_in_parameters_vi import (
    DatasetBundle,
    NoiseExperimentSettings,
    _resolve_device,
    make_dataset,
)
from noise_in_input_vi import (
    InputNoiseEntry,
    InputNoiseSettings,
    add_noise_to_inputs,
    evaluate_deterministic_input_noise,
    plot_mse_curve,
    plot_predictions_1d,
    prepare_noisy_inputs,
)


@dataclass
class MCMCInputSettings(InputNoiseSettings):
    """Extension with Metropolis-specific controls."""

    results_folder: str = "input_noise_mcmc"
    metropolis_samples: int = 800
    metropolis_burn_in: int = 600
    metropolis_step_size: float = 2e-4
    mcmc_noise_var: float = 0.05
    mcmc_prior_var: float = 1.0
    mcmc_eval_samples: int = 200
    map_epochs: int = 800
    map_lr: float = 5e-4
    map_weight_decay: float = 2e-4


def _subset_samples(samples: torch.Tensor, max_samples: int) -> torch.Tensor:
    if samples.numel() == 0:
        raise ValueError("MCMC produced no samples; cannot evaluate robustness.")
    if samples.shape[0] <= max_samples:
        return samples
    return samples[:max_samples]


def evaluate_mcmc_input_noise(
    template: torch.nn.Module,
    samples: torch.Tensor,
    bundle: DatasetBundle,
    noise_std: float,
    settings: MCMCInputSettings,
    target_fn,
    inputs_override: torch.Tensor | None = None,
) -> InputNoiseEntry:
    if inputs_override is not None:
        X_noisy = inputs_override.to(bundle.X_eval.device)
    else:
        X_noisy = add_noise_to_inputs(
            bundle.X_eval,
            noise_std,
            domain=bundle.domain,
            clamp=settings.clamp_to_domain,
        )
    preds: List[torch.Tensor] = []
    for vec in samples:
        vector_to_parameters(vec.clone(), template.parameters())
        output, _ = template(X_noisy, sample=False)
        preds.append(output.detach().cpu())
    stacked = torch.stack(preds)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    target = target_fn(X_noisy)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    errors = mean - target.detach().cpu()
    mse = float(torch.mean(errors ** 2))
    mae = float(torch.mean(errors.abs()))
    max_abs = float(torch.max(errors.abs()))
    metrics = {"mse": mse, "mae": mae, "max_abs_err": max_abs}
    return InputNoiseEntry(
        noise_std=float(noise_std),
        inputs=X_noisy.detach().cpu(),
        target=target.detach().cpu(),
        mean=mean,
        std=std,
        metrics=metrics,
    )


def run_mcmc_noise_sweep(
    template: torch.nn.Module,
    samples: torch.Tensor,
    bundle: DatasetBundle,
    noise_levels: Sequence[float],
    settings: MCMCInputSettings,
    target_fn,
    prepared_inputs: Dict[float, torch.Tensor] | None = None,
) -> List[InputNoiseEntry]:
    entries: List[InputNoiseEntry] = []
    for level in noise_levels:
        override = None
        if prepared_inputs is not None and level in prepared_inputs:
            override = prepared_inputs[level]
        entry = evaluate_mcmc_input_noise(
            template,
            samples,
            bundle,
            level,
            settings,
            target_fn,
            inputs_override=override,
        )
        entries.append(entry)
    return entries


def main() -> None:
    settings = MCMCInputSettings()
    device = _resolve_device(settings.device)
    set_seed(settings.seed)

    output_dir = Path(__file__).resolve().parent / "results" / settings.results_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = make_dataset(settings, device)
    _, target_fn = resolve_target(settings.target, settings.dim)

    prepared_inputs = prepare_noisy_inputs(bundle, settings.input_noise_levels, settings)

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

    det_sweep = [
        evaluate_deterministic_input_noise(
            det_model,
            bundle,
            level,
            target_fn,
            settings=settings,
            inputs_override=prepared_inputs.get(level),
        )
        for level in settings.input_noise_levels
    ]

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

    mcmc_sweep = run_mcmc_noise_sweep(
        eval_model,
        samples,
        bundle,
        settings.input_noise_levels,
        settings,
        target_fn,
        prepared_inputs=prepared_inputs,
    )

    plot_mse_curve(det_sweep, mcmc_sweep, output_dir / "mse_vs_input_noise.png")
    if bundle.X_eval.shape[1] == 1:
        baseline_det = det_sweep[0]
        baseline_mcmc = mcmc_sweep[0]
        worst_det = det_sweep[-1]
        worst_mcmc = mcmc_sweep[-1]
        plot_predictions_1d(
            bundle,
            baseline_det,
            baseline_mcmc,
            title="Predictions without input noise",
            output_path=output_dir / "predictions_baseline.png",
        )
        plot_predictions_1d(
            bundle,
            worst_det,
            worst_mcmc,
            title=f"Predictions with input noise std={settings.input_noise_levels[-1]:.2f}",
            output_path=output_dir / "predictions_noisy.png",
        )

    # Training and MAP diagnostics
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
            "layer_sizes": list(settings.layer_sizes),
            "basis_kwargs": dict(settings.basis_kwargs),
            "input_noise_levels": list(settings.input_noise_levels),
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

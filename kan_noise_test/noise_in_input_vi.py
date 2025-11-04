"""
Input noise robustness experiment for KAN models (deterministic vs VI).

This script trains both a deterministic and a variational Kolmogorov–Arnold
Network with B-spline basis, then perturbs evaluation inputs with Gaussian
noise to compare robustness and generalisation behaviour.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import set_seed
from inference.vi import VIConfig, train_vi
from models.kan import build_kan
from targets import resolve_target
from training.trainer import DeterministicConfig, train_deterministic
from noise_in_parameters_vi import (
    DatasetBundle,
    NoiseExperimentSettings,
    _resolve_device,
    make_dataset,
)


@dataclass
class InputNoiseSettings(NoiseExperimentSettings):
    """Configuration for input-noise robustness experiments."""

    input_noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2)
    results_folder: str = "input_noise_vi"
    clamp_to_domain: bool = True


@dataclass
class InputNoiseEntry:
    noise_std: float
    inputs: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    metrics: Dict[str, float]


def add_noise_to_inputs(
    X: torch.Tensor,
    noise_std: float,
    *,
    domain: Sequence[Tuple[float, float]],
    clamp: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if noise_std <= 0:
        return X.clone()
    if generator is not None:
        noise = torch.randn(
            X.shape,
            device=X.device,
            dtype=X.dtype,
            generator=generator,
        ) * noise_std
    else:
        noise = torch.randn_like(X) * noise_std
    noisy = X + noise
    if clamp and domain:
        lows = torch.tensor([low for low, _ in domain], device=X.device)
        highs = torch.tensor([high for _, high in domain], device=X.device)
        noisy = torch.max(torch.min(noisy, highs), lows)
    return noisy


def evaluate_deterministic_input_noise(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    noise_std: float,
    target_fn,
    *,
    settings: InputNoiseSettings,
    inputs_override: Optional[torch.Tensor] = None,
) -> InputNoiseEntry:
    model.eval()
    if inputs_override is not None:
        X_noisy = inputs_override.to(bundle.X_eval.device)
    else:
        X_noisy = add_noise_to_inputs(
            bundle.X_eval,
            noise_std,
            domain=bundle.domain,
            clamp=settings.clamp_to_domain,
        )
    with torch.no_grad():
        mean, _ = model(X_noisy, sample=False)
    target = target_fn(X_noisy)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    metrics = _compute_metrics(mean, target)
    return InputNoiseEntry(
        noise_std=float(noise_std),
        inputs=X_noisy.detach().cpu(),
        target=target.detach().cpu(),
        mean=mean.detach().cpu(),
        std=torch.zeros_like(mean).detach().cpu(),
        metrics=metrics,
    )


def evaluate_variational_input_noise(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    noise_std: float,
    target_fn,
    *,
    eval_samples: int,
    settings: InputNoiseSettings,
    inputs_override: Optional[torch.Tensor] = None,
) -> InputNoiseEntry:
    model.eval()
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
    with torch.no_grad():
        for _ in range(eval_samples):
            pred, _ = model(X_noisy, sample=True, n_samples=1)
            preds.append(pred)
    stacked = torch.stack(preds)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    target = target_fn(X_noisy)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    metrics = _compute_metrics(mean, target)
    return InputNoiseEntry(
        noise_std=float(noise_std),
        inputs=X_noisy.detach().cpu(),
        target=target.detach().cpu(),
        mean=mean.detach().cpu(),
        std=std.detach().cpu(),
        metrics=metrics,
    )


def _compute_metrics(preds: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    errors = preds - target
    mse = float(torch.mean(errors ** 2))
    mae = float(torch.mean(errors.abs()))
    max_abs = float(torch.max(errors.abs()))
    return {"mse": mse, "mae": mae, "max_abs_err": max_abs}


def run_input_noise_sweep(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    noise_levels: Sequence[float],
    *,
    variational: bool,
    eval_samples: int,
    settings: InputNoiseSettings,
    target_fn,
    prepared_inputs: Optional[Dict[float, torch.Tensor]] = None,
) -> List[InputNoiseEntry]:
    entries: List[InputNoiseEntry] = []
    for level in noise_levels:
        override = None
        if prepared_inputs is not None and level in prepared_inputs:
            override = prepared_inputs[level]
        if variational:
            entry = evaluate_variational_input_noise(
                model,
                bundle,
                level,
                target_fn,
                eval_samples=eval_samples,
                settings=settings,
                inputs_override=override,
            )
        else:
            entry = evaluate_deterministic_input_noise(
                model,
                bundle,
                level,
                target_fn,
                settings=settings,
                inputs_override=override,
            )
        entries.append(entry)
    return entries


def plot_mse_curve(
    deterministic: Sequence[InputNoiseEntry],
    bayesian: Sequence[InputNoiseEntry],
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    det_noise = [entry.noise_std for entry in deterministic]
    det_mse = [entry.metrics["mse"] for entry in deterministic]
    bayes_noise = [entry.noise_std for entry in bayesian]
    bayes_mse = [entry.metrics["mse"] for entry in bayesian]
    plt.plot(det_noise, det_mse, marker="o", label="Deterministic")
    plt.plot(bayes_noise, bayes_mse, marker="s", label="Bayesian (VI)")
    plt.xlabel("Input noise std")
    plt.ylabel("MSE on evaluation grid")
    plt.title("Prediction error under input noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions_1d(
    bundle: DatasetBundle,
    deterministic: InputNoiseEntry,
    bayesian: InputNoiseEntry,
    title: str,
    output_path: Path,
) -> None:
    if bundle.X_eval.shape[1] != 1:
        return
    target_x, target_y = _sorted_arrays(deterministic.inputs, deterministic.target)
    det_x, det_y = _sorted_arrays(deterministic.inputs, deterministic.mean)
    bayes_x, bayes_mean = _sorted_arrays(bayesian.inputs, bayesian.mean)
    _, bayes_std = _sorted_arrays(bayesian.inputs, bayesian.std)

    plt.figure(figsize=(8, 5))
    plt.plot(target_x, target_y, label="Target", color="black", linewidth=1.5)
    plt.plot(det_x, det_y, label="Deterministic mean", color="#1f77b4", linewidth=1.2)
    plt.plot(bayes_x, bayes_mean, label="Bayesian mean", color="#d62728", linewidth=1.2)
    plt.fill_between(
        bayes_x,
        bayes_mean - 2 * bayes_std,
        bayes_mean + 2 * bayes_std,
        color="#d62728",
        alpha=0.2,
        label="Bayesian ±2σ",
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


def _sorted_arrays(inputs: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    arr_x = inputs.squeeze(-1).numpy()
    arr_y = values.squeeze(-1).numpy()
    order = arr_x.argsort()
    return arr_x[order], arr_y[order]


def prepare_noisy_inputs(
    bundle: DatasetBundle,
    noise_levels: Sequence[float],
    settings: InputNoiseSettings,
) -> Dict[float, torch.Tensor]:
    prepared: Dict[float, torch.Tensor] = {}
    for idx, level in enumerate(noise_levels):
        if level <= 0:
            prepared[level] = bundle.X_eval.clone()
            continue
        generator = torch.Generator(device=bundle.X_eval.device)
        generator.manual_seed(settings.seed + idx * 17)
        noisy = add_noise_to_inputs(
            bundle.X_eval,
            level,
            domain=bundle.domain,
            clamp=settings.clamp_to_domain,
            generator=generator,
        )
        prepared[level] = noisy
    return prepared


def main() -> None:
    settings = InputNoiseSettings()
    device = _resolve_device(settings.device)
    set_seed(settings.seed)

    output_dir = Path(__file__).resolve().parent / "results" / settings.results_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = make_dataset(settings, device)
    _, target_fn = resolve_target(settings.target, settings.dim)

    prepared_inputs = prepare_noisy_inputs(bundle, settings.input_noise_levels, settings)

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

    det_sweep = run_input_noise_sweep(
        det_model,
        bundle,
        settings.input_noise_levels,
        variational=False,
        eval_samples=1,
        settings=settings,
        target_fn=target_fn,
        prepared_inputs=prepared_inputs,
    )

    vi_model = build_kan(
        input_dim=bundle.X_train.shape[1],
        basis_name="bspline",
        layer_sizes=settings.layer_sizes,
        variational=True,
        **settings.basis_kwargs,
    ).to(device)
    vi_config = VIConfig(
        epochs=2000,
        lr=5e-4,
        weight_decay=1e-4,
        kl_max=1e-3,
        kl_warmup_epochs=300,
        scheduler_step=300,
        scheduler_gamma=0.5,
        n_samples=8,
        grad_clip=1.0,
        eval_samples=settings.vi_eval_samples,
    )
    vi_result = train_vi(
        vi_model,
        bundle.X_train,
        bundle.y_train,
        bundle.X_eval,
        vi_config,
    )

    vi_sweep = run_input_noise_sweep(
        vi_result.model,
        bundle,
        settings.input_noise_levels,
        variational=True,
        eval_samples=settings.vi_eval_samples,
        settings=settings,
        target_fn=target_fn,
        prepared_inputs=prepared_inputs,
    )

    plot_mse_curve(det_sweep, vi_sweep, output_dir / "mse_vs_input_noise.png")
    baseline_det = det_sweep[0]
    baseline_vi = vi_sweep[0]
    worst_det = det_sweep[-1]
    worst_vi = vi_sweep[-1]
    plot_predictions_1d(
        bundle,
        baseline_det,
        baseline_vi,
        title="Predictions without input noise",
        output_path=output_dir / "predictions_baseline.png",
    )
    plot_predictions_1d(
        bundle,
        worst_det,
        worst_vi,
        title=f"Predictions with input noise std={settings.input_noise_levels[-1]:.2f}",
        output_path=output_dir / "predictions_noisy.png",
    )

    # Training curves
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
    plt.plot(vi_result.losses.cpu().numpy(), label="ELBO")
    plt.plot(vi_result.recon_losses.cpu().numpy(), label="Reconstruction")
    plt.plot(vi_result.kl_terms.cpu().numpy(), label="KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss terms")
    plt.title("Variational training curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "vi_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "settings": {
            "target": settings.target,
            "dim": settings.dim,
            "n_train": settings.n_train,
            "layer_sizes": list(settings.layer_sizes),
            "basis_kwargs": dict(settings.basis_kwargs),
            "input_noise_levels": list(settings.input_noise_levels),
        },
        "deterministic": [
            {"noise_std": entry.noise_std, **entry.metrics} for entry in det_sweep
        ],
        "bayesian": [
            {"noise_std": entry.noise_std, **entry.metrics} for entry in vi_sweep
        ],
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Results saved under {output_dir}")
    print(f"Metric summary: {metrics_path}")


if __name__ == "__main__":
    main()

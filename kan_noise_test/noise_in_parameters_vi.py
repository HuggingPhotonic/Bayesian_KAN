"""
Parameter noise robustness experiment for KAN models.

This script trains both a deterministic and a Bayesian (VI) Kolmogorov–Arnold
Network using the B-spline basis, then injects Gaussian noise into the learned
parameters to probe robustness and generalisation.  It produces quantitative
metrics and visualisations comparing how each model reacts to parameter noise.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import set_seed
from inference.vi import VIConfig, train_vi
from models.kan import build_kan
from targets import get_function_info, resolve_target
from training.trainer import DeterministicConfig, train_deterministic

# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class NoiseExperimentSettings:
    """Top-level knobs for the parameter-noise robustness study."""

    target: str = "sin_cos"
    dim: int = 1
    n_train: int = 512
    train_noise_std: float = 0.05
    eval_points_1d: int = 400
    eval_points_2d: int = 80
    layer_sizes: Sequence[int] = (64, 64, 1)
    basis_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"grid_size": 48, "spline_order": 5, "prior_scale": 0.5}
    )
    param_noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2)
    vi_eval_samples: int = 200
    seed: int = 42
    device: Optional[str] = None


@dataclass
class DatasetBundle:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_eval: torch.Tensor
    y_eval: torch.Tensor
    domain: Sequence[Tuple[float, float]]
    grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    eval_shape: Optional[Tuple[int, ...]] = None


@dataclass
class SweepEntry:
    noise_std: float
    mean: torch.Tensor
    std: torch.Tensor
    metrics: Dict[str, float]


# =============================================================================
# Dataset utilities
# =============================================================================


def _resolve_device(device: Optional[str]) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return resolved


def _sample_uniform(domain: Sequence[Tuple[float, float]], n_samples: int, device: torch.device) -> torch.Tensor:
    lows = torch.tensor([interval[0] for interval in domain], device=device)
    highs = torch.tensor([interval[1] for interval in domain], device=device)
    return torch.rand(n_samples, len(domain), device=device) * (highs - lows) + lows


def make_dataset(settings: NoiseExperimentSettings, device: torch.device) -> DatasetBundle:
    target_dim, target_fn = resolve_target(settings.target, settings.dim)
    try:
        info = get_function_info(settings.target)
        raw_domain = info["domain"]
        domain = [tuple(bounds) for bounds in raw_domain]
    except ValueError:
        domain = [(-1.0, 1.0) for _ in range(target_dim)]

    X_train = _sample_uniform(domain, settings.n_train, device=device)
    y_train = target_fn(X_train)
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(-1)
    else:
        y_train = y_train.view(y_train.size(0), -1)
        if y_train.size(1) != 1:
            raise ValueError("Only scalar targets are supported for this experiment.")
    if settings.train_noise_std > 0:
        y_train = y_train + torch.randn_like(y_train) * settings.train_noise_std
    y_train = y_train.contiguous()

    if target_dim == 1:
        xs = torch.linspace(domain[0][0], domain[0][1], settings.eval_points_1d, device=device).unsqueeze(-1)
        X_eval = xs
        grid = None
        eval_shape = (settings.eval_points_1d,)
    elif target_dim == 2:
        xs = torch.linspace(domain[0][0], domain[0][1], settings.eval_points_2d, device=device)
        ys = torch.linspace(domain[1][0], domain[1][1], settings.eval_points_2d, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        X_eval = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        grid = (grid_x, grid_y)
        eval_shape = grid_x.shape
    else:
        raise NotImplementedError("Parameter-noise test currently supports only 1D or 2D targets.")

    y_eval = target_fn(X_eval)
    if y_eval.dim() == 1:
        y_eval = y_eval.unsqueeze(-1)
    else:
        y_eval = y_eval.view(y_eval.size(0), -1)
        if y_eval.size(1) != 1:
            raise ValueError("Only scalar targets are supported for this experiment.")
    y_eval = y_eval.contiguous()

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        domain=domain,
        grid=grid,
        eval_shape=eval_shape,
    )


# =============================================================================
# Evaluation helpers
# =============================================================================


def _parameter_scale(param: torch.Tensor) -> torch.Tensor:
    # Use std as baseline, fall back to mean absolute value, then unit scale.
    values = param.detach().float().view(-1)
    if values.numel() == 0:
        return torch.tensor(1.0, device=param.device)
    std = values.std(unbiased=False)
    if torch.isnan(std) or std <= 1e-8:
        mean_abs = values.abs().mean()
        if torch.isnan(mean_abs) or mean_abs <= 1e-8:
            return torch.tensor(1.0, device=param.device)
        return mean_abs.to(param.device)
    return std.to(param.device)


def perturb_model_parameters(
    model: torch.nn.Module,
    noise_std: float,
    *,
    skip_substrings: Iterable[str] = ("log_var",),
) -> torch.nn.Module:
    perturbed = copy.deepcopy(model)
    perturbed.eval()
    with torch.no_grad():
        for name, param in perturbed.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in skip_substrings):
                continue
            scale = _parameter_scale(param)
            noise = torch.randn_like(param) * noise_std * scale
            param.add_(noise)
    return perturbed


def evaluate_model(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    variational: bool,
    eval_samples: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    model.eval()
    X_eval = bundle.X_eval
    y_eval = bundle.y_eval
    with torch.no_grad():
        if variational:
            preds: List[torch.Tensor] = []
            for _ in range(eval_samples):
                pred, _ = model(X_eval, sample=True, n_samples=1)
                preds.append(pred)
            stacked = torch.stack(preds, dim=0)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0)
        else:
            mean, _ = model(X_eval, sample=False)
            std = torch.zeros_like(mean)

    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    errors = mean - y_eval
    mse = float(torch.mean(errors ** 2).item())
    mae = float(torch.mean(errors.abs()).item())
    max_abs = float(torch.max(errors.abs()).item())
    metrics = {"mse": mse, "mae": mae, "max_abs_err": max_abs}
    return mean_cpu, std_cpu, metrics


def run_noise_sweep(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    noise_levels: Sequence[float],
    *,
    variational: bool,
    eval_samples: int,
    device: torch.device,
) -> List[SweepEntry]:
    results: List[SweepEntry] = []
    base_mean, base_std, base_metrics = evaluate_model(
        model, bundle, variational=variational, eval_samples=eval_samples
    )
    results.append(SweepEntry(noise_std=float(noise_levels[0]), mean=base_mean, std=base_std, metrics=base_metrics))

    for level in noise_levels[1:]:
        candidate = perturb_model_parameters(model, level)
        candidate = candidate.to(device)
        mean, std, metrics = evaluate_model(
            candidate, bundle, variational=variational, eval_samples=eval_samples
        )
        results.append(SweepEntry(noise_std=float(level), mean=mean, std=std, metrics=metrics))
    return results


# =============================================================================
# Visualisation helpers
# =============================================================================


def plot_mse_curve(
    deterministic: Sequence[SweepEntry],
    bayesian: Sequence[SweepEntry],
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    det_noise = [entry.noise_std for entry in deterministic]
    det_mse = [entry.metrics["mse"] for entry in deterministic]
    bayes_noise = [entry.noise_std for entry in bayesian]
    bayes_mse = [entry.metrics["mse"] for entry in bayesian]
    plt.plot(det_noise, det_mse, marker="o", label="Deterministic")
    plt.plot(bayes_noise, bayes_mse, marker="s", label="Bayesian (VI)")
    plt.xlabel("Parameter noise std (relative)")
    plt.ylabel("MSE on evaluation grid")
    plt.title("Prediction error under parameter noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions_1d(
    bundle: DatasetBundle,
    deterministic: SweepEntry,
    bayesian: SweepEntry,
    title: str,
    output_path: Path,
) -> None:
    x = bundle.X_eval.squeeze(-1).cpu().numpy()
    target = bundle.y_eval.squeeze(-1).cpu().numpy()
    det_mean = deterministic.mean.squeeze(-1).numpy()
    bayes_mean = bayesian.mean.squeeze(-1).numpy()
    bayes_std = bayesian.std.squeeze(-1).numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, target, label="Target", color="black", linewidth=1.5)
    plt.plot(x, det_mean, label="Deterministic mean", color="#1f77b4", linewidth=1.2)
    plt.plot(x, bayes_mean, label="Bayesian mean", color="#d62728", linewidth=1.2)
    plt.fill_between(
        x,
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


# =============================================================================
# Main execution
# =============================================================================


def main() -> None:
    settings = NoiseExperimentSettings()
    device = _resolve_device(settings.device)
    set_seed(settings.seed)

    output_dir = Path(__file__).resolve().parent / "results" / "parameter_noise"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("======making dataset======")
    bundle = make_dataset(settings, device)
    print("======making dataset finish=====")

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
    
    print("========train deterministic model======")
    det_losses = train_deterministic(det_model, bundle.X_train, bundle.y_train, det_config)

    det_sweep = run_noise_sweep(
        det_model,
        bundle,
        settings.param_noise_levels,
        variational=False,
        eval_samples=1,
        device=device,
    )

    # Bayesian (Variational Inference)
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
    
    print("========train bayesian model========")
    vi_result = train_vi(
        vi_model,
        bundle.X_train,
        bundle.y_train,
        bundle.X_eval,
        vi_config,
    )
    vi_sweep = run_noise_sweep(
        vi_result.model,
        bundle,
        settings.param_noise_levels,
        variational=True,
        eval_samples=settings.vi_eval_samples,
        device=device,
    )

    print("========visualization========")
    # Visualisations
    plot_mse_curve(det_sweep, vi_sweep, output_dir / "mse_vs_parameter_noise.png")
    if bundle.X_eval.shape[1] == 1:
        baseline_det = det_sweep[0]
        baseline_vi = vi_sweep[0]
        worst_det = det_sweep[-1]
        worst_vi = vi_sweep[-1]
        plot_predictions_1d(
            bundle,
            baseline_det,
            baseline_vi,
            title="Predictions without parameter noise",
            output_path=output_dir / "predictions_baseline.png",
        )
        plot_predictions_1d(
            bundle,
            worst_det,
            worst_vi,
            title=f"Predictions with parameter noise std={settings.param_noise_levels[-1]:.2f}",
            output_path=output_dir / "predictions_noisy.png",
        )

    # Store training curves
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

    # Persist metrics
    summary = {
        "settings": {
            "target": settings.target,
            "dim": settings.dim,
            "n_train": settings.n_train,
            "train_noise_std": settings.train_noise_std,
            "layer_sizes": list(settings.layer_sizes),
            "basis_kwargs": settings.basis_kwargs,
            "param_noise_levels": list(settings.param_noise_levels),
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

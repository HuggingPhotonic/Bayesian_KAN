from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def plot_training_curves(
    losses: Sequence[float] | torch.Tensor,
    *,
    recon_losses: Optional[Sequence[float] | torch.Tensor] = None,
    kl_terms: Optional[Sequence[float] | torch.Tensor] = None,
    title: str = "Training Curves",
    output_path: Path | None = None,
) -> Path:
    losses_np = _to_numpy(torch.as_tensor(losses))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses_np, label="Loss", linewidth=1.5)

    if recon_losses is not None:
        ax.plot(_to_numpy(torch.as_tensor(recon_losses)),
                label="Reconstruction", linewidth=1.0, linestyle="--")
    if kl_terms is not None:
        ax.plot(_to_numpy(torch.as_tensor(kl_terms)),
                label="KL", linewidth=1.0, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path is None:
        output_path = Path("training_curves.png")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_training_curve_panels(
    losses: Sequence[float] | torch.Tensor,
    recon_losses: Sequence[float] | torch.Tensor,
    kl_terms: Sequence[float] | torch.Tensor,
    *,
    title: str = "VI Training Breakdown",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    loss_np = _to_numpy(torch.as_tensor(losses))
    recon_np = _to_numpy(torch.as_tensor(recon_losses))
    kl_np = _to_numpy(torch.as_tensor(kl_terms))

    epochs = np.arange(len(loss_np))

    metric_specs = [
        ("training_curve_loss", loss_np, "Total Loss", "tab:blue"),
        ("training_curve_reconstruction", recon_np, "Reconstruction (MSE)", "tab:green"),
        ("training_curve_kl", kl_np, "KL Divergence", "tab:orange"),
    ]

    if output_dir is None:
        output_dir = Path("training_curves_panels")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for key, values, label, color in metric_specs:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(epochs, values, color=color, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        metric_path = output_dir / f"{key}.png"
        fig.savefig(metric_path, dpi=150)
        plt.close(fig)
        paths[key] = metric_path

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    fig.suptitle(title)
    for ax, (_, values, label, color) in zip(axes, metric_specs):
        ax.plot(epochs, values, color=color, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    fig.tight_layout()
    composite_path = output_dir / "training_curves_triptych.png"
    fig.savefig(composite_path, dpi=150)
    plt.close(fig)
    paths["training_curve_triptych"] = composite_path

    return paths


def plot_1d_regression(
    x_eval: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mean: torch.Tensor | np.ndarray,
    std: Optional[torch.Tensor | np.ndarray] = None,
    *,
    title: str = "1D Regression",
    output_path: Path | None = None,
) -> Path:
    x_np = _to_numpy(x_eval).reshape(-1)
    target_np = _to_numpy(target).reshape(-1)
    mean_np = _to_numpy(mean).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_np, target_np, label="Target", linewidth=2.0, alpha=0.8)
    ax.plot(x_np, mean_np, label="Prediction", linewidth=2.0)

    if std is not None:
        std_np = _to_numpy(std).reshape(-1)
        ax.fill_between(
            x_np,
            mean_np - 2 * std_np,
            mean_np + 2 * std_np,
            color="tab:blue",
            alpha=0.2,
            label="±2 std",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path is None:
        output_path = Path("regression_1d.png")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_2d_surfaces(
    x_grid: torch.Tensor | np.ndarray,
    y_grid: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    std: Optional[torch.Tensor | np.ndarray] = None,
    *,
    title_target: str = "Target Surface",
    title_prediction: str = "Predicted Surface",
    title_uncertainty: str = "Predictive Std",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    x_np = _to_numpy(x_grid)
    y_np = _to_numpy(y_grid)
    target_np = _to_numpy(target)
    pred_np = _to_numpy(prediction)
    paths: dict[str, Path] = {}

    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _surface(data: np.ndarray, title: str, filename: str):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_np, y_np, data, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    paths["target"] = _surface(target_np, title_target, "surface_target.png")
    paths["prediction"] = _surface(pred_np, title_prediction, "surface_prediction.png")

    if std is not None:
        std_np = _to_numpy(std)
        paths["uncertainty"] = _surface(std_np, title_uncertainty, "surface_uncertainty.png")
    return paths

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, Tuple, Literal

import torch

from config import ExperimentConfig, default_config
from targets.functions import get_target_function
from models.kan import build_kan
from inference.vi import train_vi, VIConfig, VIResult
from inference.laplace import (
    run_laplace,
    LaplaceConfig,
    LaplaceResult,
    train_map,
)
from inference.metropolis import (
    run_metropolis,
    MetropolisConfig,
    MCMCResult,
)
from inference.hmc import (
    run_hmc,
    HMCConfig,
    HMCResult,
)
from visualization import (
    plot_training_curves,
    plot_1d_regression,
    plot_2d_surfaces,
)

MethodLiteral = Literal["vi", "laplace", "metropolis", "hmc"]


@dataclass
class DatasetConfig:
    n_train: int = 512
    noise_std: float = 0.05
    range_min: float = -1.0
    range_max: float = 1.0
    eval_points_1d: int = 400
    eval_points_2d: int = 60
    seed_offset: int = 0  # allows per-run perturbation of global seed


@dataclass
class DatasetBundle:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_eval: torch.Tensor
    y_eval: torch.Tensor
    grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    eval_shape: Optional[Tuple[int, ...]] = None


@dataclass
class TrainerConfig:
    method: MethodLiteral = "vi"
    dim: int = 1
    target: str = "sin_cos"
    basis: str = "bspline"
    layer_sizes: Sequence[int] = (32, 32, 1)
    basis_kwargs: Dict[str, float] = field(default_factory=dict)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vi: VIConfig = field(default_factory=VIConfig)
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    metropolis: MetropolisConfig = field(default_factory=MetropolisConfig)
    hmc: HMCConfig = field(default_factory=HMCConfig)
    pretrain_map: bool = True
    results_subdir: Optional[str] = None


def _generate_dataset(
    exp_cfg: ExperimentConfig,
    trainer_cfg: TrainerConfig,
) -> DatasetBundle:
    data_cfg = trainer_cfg.dataset
    dim = trainer_cfg.dim
    target_fn = get_target_function(dim, trainer_cfg.target)

    low, high = data_cfg.range_min, data_cfg.range_max
    device = exp_cfg.device
    if data_cfg.seed_offset:
        torch.manual_seed(exp_cfg.seed + data_cfg.seed_offset)

    X_train = (torch.rand(data_cfg.n_train, dim, device=device) * (high - low) + low)
    y_clean = target_fn(X_train)
    noise = torch.randn_like(y_clean) * data_cfg.noise_std
    y_train = y_clean + noise

    if dim == 1:
        x_eval = torch.linspace(low, high, data_cfg.eval_points_1d, device=device).unsqueeze(-1)
        y_eval = target_fn(x_eval)
        bundle = DatasetBundle(
            X_train=X_train,
            y_train=y_train,
            X_eval=x_eval,
            y_eval=y_eval,
            grid=None,
            eval_shape=(data_cfg.eval_points_1d,),
        )
    elif dim == 2:
        lin = torch.linspace(low, high, data_cfg.eval_points_2d, device=device)
        grid_x, grid_y = torch.meshgrid(lin, lin, indexing="xy")
        X_eval = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        y_eval = target_fn(X_eval)
        bundle = DatasetBundle(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            grid=(grid_x, grid_y),
            eval_shape=grid_x.shape,
        )
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}")

    # restore original seed to avoid side-effects for subsequent calls
    torch.manual_seed(exp_cfg.seed)
    return bundle


def _plot_predictions(
    trainer_cfg: TrainerConfig,
    bundle: DatasetBundle,
    mean: torch.Tensor,
    std: torch.Tensor,
    exp_cfg: ExperimentConfig,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    if trainer_cfg.dim == 1:
        path = plot_1d_regression(
            bundle.X_eval.cpu(),
            bundle.y_eval.cpu(),
            mean,
            std,
            title=f"{trainer_cfg.method.upper()} | {trainer_cfg.basis} | 1D",
            output_path=exp_cfg.results_dir / "prediction_1d.png",
        )
        plots["prediction"] = str(path)
    else:
        if bundle.grid is None or bundle.eval_shape is None:
            raise RuntimeError("2D experiment requires evaluation grid.")
        grid_x, grid_y = bundle.grid
        mean_grid = mean.view(bundle.eval_shape)
        std_grid = std.view(bundle.eval_shape)
        target_grid = bundle.y_eval.view(bundle.eval_shape)
        paths = plot_2d_surfaces(
            grid_x.cpu().numpy(),
            grid_y.cpu().numpy(),
            target_grid.cpu().numpy(),
            mean_grid.cpu().numpy(),
            std_grid.cpu().numpy(),
            title_target=f"Target | {trainer_cfg.target}",
            title_prediction=f"{trainer_cfg.method.upper()} Prediction",
            title_uncertainty=f"{trainer_cfg.method.upper()} Std",
            output_dir=exp_cfg.results_dir,
        )
        for name, path in paths.items():
            plots[f"surface_{name}"] = str(path)
    return plots


def run_experiment(trainer_cfg: TrainerConfig) -> Dict[str, Dict[str, str]]:
    if trainer_cfg.dim not in (1, 2):
        raise ValueError("Only 1D or 2D experiments are currently supported.")
    if trainer_cfg.layer_sizes[-1] != 1:
        raise ValueError("Final layer size must be 1 for scalar regression targets.")

    results_subdir = trainer_cfg.results_subdir or f"{trainer_cfg.method}_{trainer_cfg.basis}_{trainer_cfg.dim}d"
    exp_cfg = default_config(results_subdir=results_subdir)
    bundle = _generate_dataset(exp_cfg, trainer_cfg)

    variational = trainer_cfg.method == "vi"
    model = build_kan(
        input_dim=trainer_cfg.dim,
        basis_name=trainer_cfg.basis,
        layer_sizes=trainer_cfg.layer_sizes,
        variational=variational,
        **trainer_cfg.basis_kwargs,
    ).to(exp_cfg.device)

    summary: Dict[str, object] = {
        "method": trainer_cfg.method,
        "basis": trainer_cfg.basis,
        "dimension": trainer_cfg.dim,
        "target": trainer_cfg.target,
        "layer_sizes": list(trainer_cfg.layer_sizes),
        "basis_kwargs": trainer_cfg.basis_kwargs,
        "dataset": asdict(trainer_cfg.dataset),
    }
    plots: Dict[str, str] = {}

    if trainer_cfg.method == "vi":
        vi_result: VIResult = train_vi(
            model, bundle.X_train, bundle.y_train, bundle.X_eval, trainer_cfg.vi
        )
        curves_path = plot_training_curves(
            vi_result.losses,
            recon_losses=vi_result.recon_losses,
            kl_terms=vi_result.kl_terms,
            title="VI Training",
            output_path=exp_cfg.results_dir / "training_curves.png",
        )
        plots["training_curves"] = str(curves_path)
        mean, std = vi_result.mean, vi_result.std
        summary.update(
            {
                "final_loss": float(vi_result.losses[-1]),
                "final_reconstruction": float(vi_result.recon_losses[-1]),
                "final_kl": float(vi_result.kl_terms[-1]),
            }
        )
    elif trainer_cfg.method == "laplace":
        laplace_result: LaplaceResult = run_laplace(
            model, bundle.X_train, bundle.y_train, bundle.X_eval, trainer_cfg.laplace
        )
        curves_path = plot_training_curves(
            laplace_result.map_losses,
            title="Laplace MAP Training",
            output_path=exp_cfg.results_dir / "map_losses.png",
        )
        plots["map_losses"] = str(curves_path)
        mean, std = laplace_result.mean, laplace_result.std
        summary.update(
            {
                "map_final_loss": float(laplace_result.map_losses[-1]),
                "damping": trainer_cfg.laplace.damping,
                "laplace_samples": trainer_cfg.laplace.n_samples,
            }
        )
    elif trainer_cfg.method in ("metropolis", "hmc"):
        map_losses = None
        if trainer_cfg.pretrain_map:
            map_losses = train_map(
                model, bundle.X_train, bundle.y_train, trainer_cfg.laplace
            )
            curves_path = plot_training_curves(
                map_losses,
                title="MAP Warmup",
                output_path=exp_cfg.results_dir / "map_warmup.png",
            )
            plots["map_warmup"] = str(curves_path)
            summary["map_final_loss"] = float(map_losses[-1])

        if trainer_cfg.method == "metropolis":
            mcmc_result: MCMCResult = run_metropolis(
                model,
                bundle.X_train,
                bundle.y_train,
                bundle.X_eval,
                trainer_cfg.metropolis,
            )
            mean, std = mcmc_result.mean, mcmc_result.std
            summary.update(
                {
                    "acceptance_rate": float(mcmc_result.acceptance_rate),
                    "step_size": trainer_cfg.metropolis.step_size,
                    "burn_in": trainer_cfg.metropolis.burn_in,
                    "posterior_samples": mcmc_result.samples.shape[0],
                }
            )
        else:
            hmc_result: HMCResult = run_hmc(
                model,
                bundle.X_train,
                bundle.y_train,
                bundle.X_eval,
                trainer_cfg.hmc,
            )
            mean, std = hmc_result.mean, hmc_result.std
            summary.update(
                {
                    "acceptance_rate": float(hmc_result.acceptance_rate),
                    "step_size": trainer_cfg.hmc.step_size,
                    "burn_in": trainer_cfg.hmc.burn_in,
                    "n_leapfrog": trainer_cfg.hmc.n_leapfrog,
                    "posterior_samples": hmc_result.samples.shape[0],
                }
            )
    else:
        raise ValueError(f"Unsupported method '{trainer_cfg.method}'.")

    prediction_plots = _plot_predictions(trainer_cfg, bundle, mean, std, exp_cfg)
    plots.update(prediction_plots)

    summary["mean_std"] = float(std.mean())
    summary["max_std"] = float(std.max())

    summary_path = exp_cfg.results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    plots["summary"] = str(summary_path)

    return {"summary": summary, "plots": plots}

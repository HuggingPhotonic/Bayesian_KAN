from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, Tuple, Literal

import torch
import torch.nn as nn

from config import ExperimentConfig, default_config, set_seed
from targets import (
    resolve_target,
    get_function_info,
)
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
    plot_training_curve_panels,
    plot_1d_regression,
    plot_2d_surfaces,
)

MethodLiteral = Literal["vi", "laplace", "metropolis", "hmc", "deterministic"]


@dataclass
class DatasetConfig:
    n_train: int = 512
    n_eval: Optional[int] = None
    noise_std: float = 0.05
    range_min: Optional[float] = -1.0
    range_max: Optional[float] = 1.0
    eval_points_1d: int = 400
    eval_points_2d: int = 60
    seed_offset: int = 0  # allows per-run perturbation of global seed
    use_function_domain: bool = True
    custom_domain: Optional[Sequence[Tuple[float, float]]] = None


@dataclass
class DatasetBundle:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_eval: torch.Tensor
    y_eval: torch.Tensor
    grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    eval_shape: Optional[Tuple[int, ...]] = None
    domain: Optional[Sequence[Tuple[float, float]]] = None


@dataclass
class DeterministicConfig:
    epochs: int = 1500
    lr: float = 5e-4
    weight_decay: float = 1e-4
    scheduler_step: int = 400
    scheduler_gamma: float = 0.5
    grad_clip: float = 1.0


@dataclass
class TrainerConfig:
    method: MethodLiteral = "vi"
    dim: Optional[int] = None
    target: str = "sin_cos"
    basis: str = "bspline"
    layer_sizes: Sequence[int] = (32, 32, 1)
    basis_kwargs: Dict[str, float] = field(default_factory=dict)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vi: VIConfig = field(default_factory=VIConfig)
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    metropolis: MetropolisConfig = field(default_factory=MetropolisConfig)
    hmc: HMCConfig = field(default_factory=HMCConfig)
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)
    pretrain_map: bool = True
    device: Optional[str] = None  # "cpu", "cuda", "auto"/None
    seed: int = 42
    results_subdir: Optional[str] = None


def _generate_dataset(
    exp_cfg: ExperimentConfig,
    trainer_cfg: TrainerConfig,
) -> DatasetBundle:
    data_cfg = trainer_cfg.dataset
    device = exp_cfg.device

    target_dim, target_fn = resolve_target(trainer_cfg.target, trainer_cfg.dim)
    info = None
    try:
        info = get_function_info(trainer_cfg.target)
    except ValueError:
        info = None

    if data_cfg.custom_domain is not None:
        domain = list(data_cfg.custom_domain)
    elif data_cfg.use_function_domain and info is not None:
        domain = info["domain"]
    else:
        if data_cfg.range_min is None or data_cfg.range_max is None:
            raise ValueError("range_min and range_max must be provided when not using function domain.")
        domain = [[data_cfg.range_min, data_cfg.range_max] for _ in range(target_dim)]

    if len(domain) != target_dim:
        raise ValueError(
            f"Domain specification for '{trainer_cfg.target}' has {len(domain)} entries, "
            f"but target dimension is {target_dim}."
        )

    def _sample(n_samples: int) -> torch.Tensor:
        lows = torch.tensor([d[0] for d in domain], device=device)
        highs = torch.tensor([d[1] for d in domain], device=device)
        return torch.rand(n_samples, target_dim, device=device) * (highs - lows) + lows

    if data_cfg.seed_offset:
        set_seed(exp_cfg.seed + data_cfg.seed_offset)

    X_train = _sample(data_cfg.n_train)
    y_train = target_fn(X_train)
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(-1)
    else:
        y_train = y_train.view(y_train.size(0), -1)
        if y_train.size(1) != 1:
            raise ValueError(
                f"Target '{trainer_cfg.target}' returns {y_train.size(1)} outputs; "
                "multi-output targets are not supported."
            )
    if data_cfg.noise_std > 0:
        y_train = y_train + torch.randn_like(y_train) * data_cfg.noise_std
    y_train = y_train.contiguous()

    if target_dim == 1:
        low, high = domain[0]
        X_eval = torch.linspace(low, high, data_cfg.eval_points_1d, device=device).unsqueeze(-1)
        grid = None
        eval_shape: Optional[Tuple[int, ...]] = (data_cfg.eval_points_1d,)
    elif target_dim == 2:
        xs = torch.linspace(domain[0][0], domain[0][1], data_cfg.eval_points_2d, device=device)
        ys = torch.linspace(domain[1][0], domain[1][1], data_cfg.eval_points_2d, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        X_eval = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        grid = (grid_x, grid_y)
        eval_shape = grid_x.shape
    else:
        n_eval = data_cfg.n_eval or min(4096, max(data_cfg.n_train, 1024))
        X_eval = _sample(n_eval)
        grid = None
        eval_shape = None

    y_eval = target_fn(X_eval)
    if y_eval.dim() == 1:
        y_eval = y_eval.unsqueeze(-1)
    else:
        y_eval = y_eval.view(y_eval.size(0), -1)
        if y_eval.size(1) != 1:
            raise ValueError(
                f"Target '{trainer_cfg.target}' returns {y_eval.size(1)} outputs; "
                "multi-output targets are not supported."
            )
    y_eval = y_eval.contiguous()

    set_seed(exp_cfg.seed)
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        grid=grid,
        eval_shape=eval_shape,
        domain=domain,
    )


def train_deterministic(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: DeterministicConfig,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = nn.MSELoss()
    losses = []

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        preds, _ = model(X_train, sample=False)
        loss = loss_fn(preds, y_train)
        loss.backward()
        if config.grad_clip is not None and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().cpu())
    return torch.stack(losses)


def _plot_predictions(
    trainer_cfg: TrainerConfig,
    bundle: DatasetBundle,
    mean: torch.Tensor,
    std: torch.Tensor,
    exp_cfg: ExperimentConfig,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    dim = bundle.X_train.shape[1]
    if dim == 1:
        path = plot_1d_regression(
            bundle.X_eval.cpu(),
            bundle.y_eval.cpu(),
            mean,
            std,
            title=f"{trainer_cfg.method.upper()} | {trainer_cfg.basis} | 1D",
            output_path=exp_cfg.results_dir / "prediction_1d.png",
        )
        plots["prediction"] = str(path)
    elif dim == 2:
        if bundle.grid is None or bundle.eval_shape is None:
            return plots
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
    # Higher-dimensional targets are not visualised.
    return plots


def run_experiment(trainer_cfg: TrainerConfig) -> Dict[str, Dict[str, str]]:
    if trainer_cfg.layer_sizes[-1] != 1:
        raise ValueError("Final layer size must be 1 for scalar regression targets.")

    target_dim, _ = resolve_target(trainer_cfg.target, trainer_cfg.dim)
    results_subdir = trainer_cfg.results_subdir or f"{trainer_cfg.method}_{trainer_cfg.basis}_{target_dim}d"
    exp_cfg = default_config(
        results_subdir=results_subdir,
        device=trainer_cfg.device,
        seed=trainer_cfg.seed,
    )
    bundle = _generate_dataset(exp_cfg, trainer_cfg)

    variational = trainer_cfg.method == "vi"
    model = build_kan(
        input_dim=bundle.X_train.shape[1],
        basis_name=trainer_cfg.basis,
        layer_sizes=trainer_cfg.layer_sizes,
        variational=variational,
        **trainer_cfg.basis_kwargs,
    ).to(exp_cfg.device)

    dataset_summary = asdict(trainer_cfg.dataset)
    if bundle.domain is not None:
        dataset_summary["effective_domain"] = [list(interval) for interval in bundle.domain]

    summary: Dict[str, object] = {
        "method": trainer_cfg.method,
        "basis": trainer_cfg.basis,
        "dimension": bundle.X_train.shape[1],
        "target": trainer_cfg.target,
        "layer_sizes": list(trainer_cfg.layer_sizes),
        "basis_kwargs": trainer_cfg.basis_kwargs,
        "dataset": dataset_summary,
        "device": str(exp_cfg.device),
    }
    plots: Dict[str, str] = {}
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

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
        panel_paths = plot_training_curve_panels(
            losses=vi_result.losses,
            recon_losses=vi_result.recon_losses,
            kl_terms=vi_result.kl_terms,
            title="VI Training Breakdown",
            output_dir=exp_cfg.results_dir / "training_curves_panels",
        )
        plots.update({key: str(path) for key, path in panel_paths.items()})
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
    elif trainer_cfg.method == "deterministic":
        losses = train_deterministic(
            model, bundle.X_train, bundle.y_train, trainer_cfg.deterministic
        )
        curves_path = plot_training_curves(
            losses,
            title="Deterministic KAN Training",
            output_path=exp_cfg.results_dir / "training_curves.png",
        )
        plots["training_curves"] = str(curves_path)
        summary["final_loss"] = float(losses[-1])
        model.eval()
        with torch.no_grad():
            mean, _ = model(bundle.X_eval, sample=False)
        std = torch.zeros_like(mean)
    else:
        raise ValueError(f"Unsupported method '{trainer_cfg.method}'.")

    if mean is None or std is None:
        raise RuntimeError("Inference outputs were not produced.")

    mean = mean.detach().cpu()
    std = std.detach().cpu()
    y_eval = bundle.y_eval.detach().cpu()

    prediction_plots = _plot_predictions(trainer_cfg, bundle, mean, std, exp_cfg)
    plots.update(prediction_plots)

    pred_flat = mean.view(mean.size(0), -1)
    target_flat = y_eval.view(y_eval.size(0), -1)
    errors = pred_flat - target_flat
    mse = float(torch.mean(errors ** 2))
    mae = float(torch.mean(errors.abs()))
    max_abs_err = float(torch.max(errors.abs()))
    metrics = {"mse": mse, "mae": mae, "max_abs_err": max_abs_err}

    summary["mean_std"] = float(std.mean())
    summary["max_std"] = float(std.max())
    summary.update(metrics)

    summary_path = exp_cfg.results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    plots["summary"] = str(summary_path)

    return {"summary": summary, "plots": plots, "metrics": metrics}

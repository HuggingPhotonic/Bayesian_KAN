from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt

from models.kan import BASIS_REGISTRY
from targets import TARGET_DIMS
from training import DatasetConfig, TrainerConfig, run_experiment

AVAILABLE_METHODS = {"vi", "laplace", "metropolis", "hmc", "deterministic"}


def _parse_list(arg: str) -> List[str]:
    return [item.strip() for item in arg.split(",") if item.strip()]


def _parse_basis_kwargs(pairs: Sequence[str]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid basis kwarg '{item}', expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        kwargs[key] = parsed
    return kwargs


def _parse_layer_sizes(arg: str) -> List[int]:
    sizes = _parse_list(arg)
    if not sizes:
        raise argparse.ArgumentTypeError("Layer sizes must contain at least one value.")
    try:
        return [int(x) for x in sizes]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layer sizes must be integers.") from exc


def _parse_target_dims(overrides: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in overrides:
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid target-dim override '{item}'. Expected format 'name:dim'."
            )
        name, dim = item.split(":", 1)
        name = name.strip()
        try:
            mapping[name] = int(dim.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid dimension '{dim}' for target '{name}'."
            ) from exc
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Bayesian KAN methods across multiple target functions."
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="sin_cos,wave",
        help="Comma-separated list of target function names.",
    )
    parser.add_argument(
        "--target-dim",
        action="append",
        default=[],
        help="Override the dimension for specific targets (format: name:dim).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="vi,laplace,metropolis,hmc",
        help="Comma-separated list of Bayesian methods to evaluate.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Exclude the deterministic KAN baseline from the comparison.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="bspline",
        choices=sorted(BASIS_REGISTRY.keys()),
        help="Basis function for KAN layers.",
    )
    parser.add_argument(
        "--layer-sizes",
        type=str,
        default="32,32,1",
        help="Comma-separated hidden sizes including final output (default: 32,32,1).",
    )
    parser.add_argument("--n-train", type=int, default=512, help="Training sample count.")
    parser.add_argument("--n-eval", type=int, default=None, help="Evaluation sample count for dim>2.")
    parser.add_argument("--eval-points-1d", type=int, default=400, help="Evaluation grid points for 1D targets.")
    parser.add_argument("--eval-points-2d", type=int, default=60, help="Evaluation grid resolution per axis for 2D targets.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Observation noise.")
    parser.add_argument(
        "--basis-kw",
        action="append",
        default=[],
        help="Additional basis kwargs as key=value (can be repeated).",
    )
    parser.add_argument(
        "--no-function-domain",
        action="store_true",
        help="Ignore builtin function domain metadata and use range_min/range_max.",
    )
    parser.add_argument("--range-min", type=float, default=-1.0, help="Minimum input value if no domain is used.")
    parser.add_argument("--range-max", type=float, default=1.0, help="Maximum input value if no domain is used.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g., 'cpu', 'cuda', 'cuda:0', or 'auto'). Default: auto-detect.",
    )
    parser.add_argument("--dim", type=int, default=None, help="Input dimensionality. If omitted, inferred from target.")
    parser.add_argument("--results-subdir", type=str, default=None, help="Optional results sub-directory name.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--no-map-warmup",
        action="store_true",
        help="Disable MAP warm-up before MCMC sampling.",
    )
    parser.add_argument("--vi-epochs", type=int, default=None, help="Override VI epochs.")
    parser.add_argument("--vi-lr", type=float, default=None, help="Override VI learning rate.")
    parser.add_argument("--vi-kl-max", type=float, default=None, help="Override VI KL max weight.")
    parser.add_argument("--vi-weight-decay", type=float, default=None, help="Override VI weight decay.")
    parser.add_argument("--vi-kl-warmup", type=int, default=None, help="Override VI KL warm-up epochs.")
    parser.add_argument("--vi-scheduler-step", type=int, default=None, help="Override VI scheduler step size.")
    parser.add_argument("--vi-scheduler-gamma", type=float, default=None, help="Override VI scheduler gamma.")
    parser.add_argument("--vi-n-samples", type=int, default=None, help="Override VI training sample count.")
    parser.add_argument("--vi-grad-clip", type=float, default=None, help="Override VI gradient clip norm.")
    parser.add_argument("--vi-eval-samples", type=int, default=None, help="Override VI evaluation samples.")
    parser.add_argument("--laplace-epochs", type=int, default=None, help="Override Laplace MAP epochs.")
    parser.add_argument("--laplace-lr", type=float, default=None, help="Override Laplace MAP learning rate.")
    parser.add_argument("--laplace-damping", type=float, default=None, help="Override Laplace damping.")
    parser.add_argument("--laplace-weight-decay", type=float, default=None, help="Override Laplace weight decay.")
    parser.add_argument("--laplace-noise-var", type=float, default=None, help="Override Laplace noise variance.")
    parser.add_argument("--laplace-prior-var", type=float, default=None, help="Override Laplace prior variance.")
    parser.add_argument("--laplace-n-samples", type=int, default=None, help="Override Laplace predictive samples.")
    parser.add_argument("--laplace-grad-clip", type=float, default=None, help="Override Laplace gradient clip.")
    parser.add_argument("--mcmc-samples", type=int, default=None, help="Override Metropolis/HMC posterior samples.")
    parser.add_argument("--mcmc-burn-in", type=int, default=None, help="Override Metropolis/HMC burn-in steps.")
    parser.add_argument("--mcmc-step-size", type=float, default=None, help="Override Metropolis step size.")
    parser.add_argument("--mcmc-noise-var", type=float, default=None, help="Override Metropolis noise variance.")
    parser.add_argument("--mcmc-prior-var", type=float, default=None, help="Override Metropolis prior variance.")
    parser.add_argument("--hmc-step-size", type=float, default=None, help="Override HMC step size.")
    parser.add_argument("--hmc-leapfrog", type=int, default=None, help="Override HMC leapfrog steps.")
    parser.add_argument("--hmc-noise-var", type=float, default=None, help="Override HMC noise variance.")
    parser.add_argument("--hmc-prior-var", type=float, default=None, help="Override HMC prior variance.")
    parser.add_argument("--det-epochs", type=int, default=None, help="Override deterministic epochs.")
    parser.add_argument("--det-lr", type=float, default=None, help="Override deterministic learning rate.")
    parser.add_argument("--det-weight-decay", type=float, default=None, help="Override deterministic weight decay.")
    parser.add_argument("--det-scheduler-step", type=int, default=None, help="Override deterministic scheduler step.")
    parser.add_argument("--det-scheduler-gamma", type=float, default=None, help="Override deterministic scheduler gamma.")
    parser.add_argument("--det-grad-clip", type=float, default=None, help="Override deterministic gradient clip.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Optional root directory for benchmark artefacts (defaults to results/benchmark/<timestamp>).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    targets = _parse_list(args.targets)
    if not targets:
        raise ValueError("No targets specified.")

    target_dims_override = _parse_target_dims(args.target_dim)

    requested_methods = [m.lower() for m in _parse_list(args.methods)]
    for method in requested_methods:
        if method not in AVAILABLE_METHODS - {"deterministic"}:
            raise ValueError(f"Unsupported method '{method}'. Available: {AVAILABLE_METHODS}.")

    methods: List[str] = list(dict.fromkeys(requested_methods))
    if not args.no_deterministic and "deterministic" not in methods:
        methods.append("deterministic")

    basis_kwargs = _parse_basis_kwargs(args.basis_kw)
    layer_sizes = tuple(_parse_layer_sizes(args.layer_sizes))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = (
        Path(args.results_root).expanduser()
        if args.results_root
        else Path("results") / "benchmark" / timestamp
    )
    results_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    per_target_metrics: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for target in targets:
        dims = TARGET_DIMS.get(target)
        if not dims:
            raise ValueError(f"Unknown target '{target}'.")
        if target in target_dims_override:
            dim = target_dims_override[target]
            if dim not in dims:
                raise ValueError(
                    f"Target '{target}' does not support dimension {dim}. Available: {dims}."
                )
        else:
            if len(dims) > 1:
                raise ValueError(
                    f"Target '{target}' supports multiple dimensions {dims}. "
                    "Please specify one using --target-dim."
                )
            dim = dims[0]

        for method in methods:
            trainer_cfg = TrainerConfig(
                method=method,
                dim=args.dim if args.dim is not None else dim,
                target=target,
                basis=args.basis,
                layer_sizes=layer_sizes,
                basis_kwargs=basis_kwargs.copy(),
                dataset=DatasetConfig(
                    n_train=args.n_train,
                    n_eval=args.n_eval,
                    noise_std=args.noise_std,
                    eval_points_1d=args.eval_points_1d,
                    eval_points_2d=args.eval_points_2d,
                    range_min=args.range_min,
                    range_max=args.range_max,
                    use_function_domain=not args.no_function_domain,
                ),
                device=args.device,
                seed=args.seed,
                results_subdir=args.results_subdir or f"benchmark/{timestamp}/{target}/{method}",
                pretrain_map=not args.no_map_warmup,
            )
            # Apply method-specific overrides
            if args.vi_epochs is not None:
                trainer_cfg.vi.epochs = args.vi_epochs
            if args.vi_lr is not None:
                trainer_cfg.vi.lr = args.vi_lr
            if args.vi_kl_max is not None:
                trainer_cfg.vi.kl_max = args.vi_kl_max
            if args.vi_weight_decay is not None:
                trainer_cfg.vi.weight_decay = args.vi_weight_decay
            if args.vi_kl_warmup is not None:
                trainer_cfg.vi.kl_warmup_epochs = args.vi_kl_warmup
            if args.vi_scheduler_step is not None:
                trainer_cfg.vi.scheduler_step = args.vi_scheduler_step
            if args.vi_scheduler_gamma is not None:
                trainer_cfg.vi.scheduler_gamma = args.vi_scheduler_gamma
            if args.vi_n_samples is not None:
                trainer_cfg.vi.n_samples = args.vi_n_samples
            if args.vi_grad_clip is not None:
                trainer_cfg.vi.grad_clip = args.vi_grad_clip
            if args.vi_eval_samples is not None:
                trainer_cfg.vi.eval_samples = args.vi_eval_samples

            if args.laplace_epochs is not None:
                trainer_cfg.laplace.map_epochs = args.laplace_epochs
            if args.laplace_lr is not None:
                trainer_cfg.laplace.map_lr = args.laplace_lr
            if args.laplace_damping is not None:
                trainer_cfg.laplace.damping = args.laplace_damping
            if args.laplace_weight_decay is not None:
                trainer_cfg.laplace.weight_decay = args.laplace_weight_decay
            if args.laplace_noise_var is not None:
                trainer_cfg.laplace.noise_var = args.laplace_noise_var
            if args.laplace_prior_var is not None:
                trainer_cfg.laplace.prior_var = args.laplace_prior_var
            if args.laplace_n_samples is not None:
                trainer_cfg.laplace.n_samples = args.laplace_n_samples
            if args.laplace_grad_clip is not None:
                trainer_cfg.laplace.grad_clip = args.laplace_grad_clip

            if args.mcmc_samples is not None:
                trainer_cfg.metropolis.n_samples = args.mcmc_samples
                trainer_cfg.hmc.n_samples = args.mcmc_samples
            if args.mcmc_burn_in is not None:
                trainer_cfg.metropolis.burn_in = args.mcmc_burn_in
                trainer_cfg.hmc.burn_in = args.mcmc_burn_in
            if args.mcmc_step_size is not None:
                trainer_cfg.metropolis.step_size = args.mcmc_step_size
            if args.mcmc_noise_var is not None:
                trainer_cfg.metropolis.noise_var = args.mcmc_noise_var
            if args.mcmc_prior_var is not None:
                trainer_cfg.metropolis.prior_var = args.mcmc_prior_var
            if args.hmc_step_size is not None:
                trainer_cfg.hmc.step_size = args.hmc_step_size
            if args.hmc_leapfrog is not None:
                trainer_cfg.hmc.n_leapfrog = args.hmc_leapfrog
            if args.hmc_noise_var is not None:
                trainer_cfg.hmc.noise_var = args.hmc_noise_var
            if args.hmc_prior_var is not None:
                trainer_cfg.hmc.prior_var = args.hmc_prior_var

            if args.det_epochs is not None:
                trainer_cfg.deterministic.epochs = args.det_epochs
            if args.det_lr is not None:
                trainer_cfg.deterministic.lr = args.det_lr
            if args.det_weight_decay is not None:
                trainer_cfg.deterministic.weight_decay = args.det_weight_decay
            if args.det_scheduler_step is not None:
                trainer_cfg.deterministic.scheduler_step = args.det_scheduler_step
            if args.det_scheduler_gamma is not None:
                trainer_cfg.deterministic.scheduler_gamma = args.det_scheduler_gamma
            if args.det_grad_clip is not None:
                trainer_cfg.deterministic.grad_clip = args.det_grad_clip

            result = run_experiment(trainer_cfg)
            summary = result["summary"]
            metrics = result.get("metrics", {})

            dataset_info = summary.get("dataset", {})
            row: Dict[str, object] = {
                "target": target,
                "dimension": summary.get("dimension"),
                "method": summary.get("method"),
                "basis": summary.get("basis"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "max_abs_err": metrics.get("max_abs_err"),
                "mean_std": summary.get("mean_std"),
                "max_std": summary.get("max_std"),
                "n_train": dataset_info.get("n_train"),
                "noise_std": dataset_info.get("noise_std"),
                "effective_domain": dataset_info.get("effective_domain"),
            }

            for key in (
                "final_loss",
                "final_reconstruction",
                "final_kl",
                "map_final_loss",
                "laplace_samples",
                "damping",
                "acceptance_rate",
                "step_size",
                "burn_in",
                "n_leapfrog",
                "posterior_samples",
            ):
                if key in summary:
                    row[key] = summary[key]

            rows.append(row)
            per_target_metrics[target].append(row)

    # Write CSV
    csv_path = results_root / "benchmark_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Visualisations
    for target, entries in per_target_metrics.items():
        methods_order = [entry["method"] for entry in entries]
        mses = [entry["mse"] for entry in entries]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(methods_order, mses, color="tab:blue")
        ax.set_title(f"MSE Comparison | {target}")
        ax.set_ylabel("MSE")
        ax.set_xlabel("Method")
        ax.grid(alpha=0.3, axis="y")
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(results_root / f"{target}_mse.png", dpi=150)
        plt.close(fig)

    print(f"Benchmark complete. Metrics written to {csv_path}")
    print(f"Visualisations stored under {results_root}")


if __name__ == "__main__":
    main()

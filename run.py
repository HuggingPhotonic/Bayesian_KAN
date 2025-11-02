from __future__ import annotations

import argparse
import json
from typing import List

from models.kan import BASIS_REGISTRY
from training import DatasetConfig, TrainerConfig, run_experiment


def _parse_layer_sizes(arg: str) -> List[int]:
    try:
        sizes = [int(x.strip()) for x in arg.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layer sizes must be comma separated integers.") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("At least one layer size must be provided.")
    return sizes


def _parse_basis_kwargs(pairs: List[str]) -> dict:
    kwargs = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid basis kwarg '{item}', expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KAN experiments with reusable components.")
    parser.add_argument(
        "--method",
        choices=["vi", "laplace", "metropolis", "hmc", "deterministic"],
        required=True,
        help="Inference method to use.",
    )
    parser.add_argument(
        "--basis",
        choices=sorted(BASIS_REGISTRY.keys()),
        default="bspline",
        help="Basis function family for KAN layers.",
    )
    parser.add_argument("--dim", type=int, default=None, help="Input dimensionality. If omitted, inferred from target.")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target function identifier (default sin_cos for 1D, wave for 2D).",
    )
    parser.add_argument(
        "--layer-sizes",
        type=_parse_layer_sizes,
        default=[32, 32, 1],
        help="Comma-separated hidden layer sizes including final output (default: 32,32,1).",
    )
    parser.add_argument("--results-subdir", type=str, default=None, help="Optional results sub-directory name.")

    parser.add_argument("--n-train", type=int, default=512, help="Number of noisy training samples.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Observation noise standard deviation.")
    parser.add_argument("--range-min", type=float, default=-1.0, help="Minimum coordinate value.")
    parser.add_argument("--range-max", type=float, default=1.0, help="Maximum coordinate value.")
    parser.add_argument("--eval-points-1d", type=int, default=400, help="Evaluation points for 1D grid.")
    parser.add_argument("--eval-points-2d", type=int, default=60, help="Evaluation resolution per axis for 2D grid.")
    parser.add_argument("--n-eval", type=int, default=None, help="Evaluation samples for dimensions > 2.")
    parser.add_argument(
        "--no-function-domain",
        action="store_true",
        help="Ignore builtin function domain metadata and use range_min/range_max instead.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g., 'cpu', 'cuda', 'cuda:0', or 'auto'). Default: auto-detect.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument(
        "--basis-kw",
        action="append",
        default=[],
        help="Additional basis keyword arguments as key=value pairs (can repeat).",
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
    parser.add_argument("--laplace-damping", type=float, default=None, help="Override Laplace damping for covariance.")
    parser.add_argument("--laplace-weight-decay", type=float, default=None, help="Override Laplace weight decay.")
    parser.add_argument("--laplace-noise-var", type=float, default=None, help="Override Laplace noise variance.")
    parser.add_argument("--laplace-prior-var", type=float, default=None, help="Override Laplace prior variance.")
    parser.add_argument("--laplace-n-samples", type=int, default=None, help="Override Laplace predictive sample count.")
    parser.add_argument("--laplace-grad-clip", type=float, default=None, help="Override Laplace gradient clip norm.")

    parser.add_argument("--mcmc-samples", type=int, default=None, help="MCMC posterior samples after burn-in.")
    parser.add_argument("--mcmc-burn-in", type=int, default=None, help="MCMC burn-in steps.")
    parser.add_argument("--mcmc-step-size", type=float, default=None, help="Step size for Metropolis sampler.")
    parser.add_argument("--mcmc-noise-var", type=float, default=None, help="Override Metropolis noise variance.")
    parser.add_argument("--mcmc-prior-var", type=float, default=None, help="Override Metropolis prior variance.")

    parser.add_argument("--hmc-step-size", type=float, default=None, help="Step size for HMC.")
    parser.add_argument("--hmc-leapfrog", type=int, default=None, help="Number of leapfrog steps for HMC.")
    parser.add_argument("--hmc-noise-var", type=float, default=None, help="Override HMC noise variance.")
    parser.add_argument("--hmc-prior-var", type=float, default=None, help="Override HMC prior variance.")

    parser.add_argument(
        "--no-map-warmup",
        action="store_true",
        help="Disable MAP warm-up before MCMC sampling.",
    )
    parser.add_argument("--det-epochs", type=int, default=None, help="Override deterministic epochs.")
    parser.add_argument("--det-lr", type=float, default=None, help="Override deterministic learning rate.")
    parser.add_argument("--det-weight-decay", type=float, default=None, help="Override deterministic weight decay.")
    parser.add_argument("--det-scheduler-step", type=int, default=None, help="Override deterministic scheduler step.")
    parser.add_argument("--det-scheduler-gamma", type=float, default=None, help="Override deterministic scheduler gamma.")
    parser.add_argument("--det-grad-clip", type=float, default=None, help="Override deterministic gradient clip.")

    return parser


def main(args: argparse.Namespace) -> None:
    dim = args.dim
    target = args.target
    if target is None:
        if dim in (None, 1):
            target = "sin_cos"
            dim = dim or 1
        elif dim == 2:
            target = "wave"
        else:
            raise ValueError("Please provide --target for dimensions other than 1 or 2.")
    basis_kwargs = _parse_basis_kwargs(args.basis_kw)
    dataset_cfg = DatasetConfig(
        n_train=args.n_train,
        n_eval=args.n_eval,
        noise_std=args.noise_std,
        range_min=args.range_min,
        range_max=args.range_max,
        eval_points_1d=args.eval_points_1d,
        eval_points_2d=args.eval_points_2d,
        use_function_domain=not args.no_function_domain,
    )

    trainer_cfg = TrainerConfig(
        method=args.method,
        dim=args.dim if args.dim is not None else dim,
        target=target,
        basis=args.basis,
        layer_sizes=tuple(args.layer_sizes),
        basis_kwargs=basis_kwargs,
        dataset=dataset_cfg,
        device=args.device,
        seed=args.seed,
        results_subdir=args.results_subdir,
        pretrain_map=not args.no_map_warmup,
    )

    if args.method == "vi":
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
    if args.method == "laplace":
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
    if args.method == "metropolis":
        if args.mcmc_samples is not None:
            trainer_cfg.metropolis.n_samples = args.mcmc_samples
        if args.mcmc_burn_in is not None:
            trainer_cfg.metropolis.burn_in = args.mcmc_burn_in
        if args.mcmc_step_size is not None:
            trainer_cfg.metropolis.step_size = args.mcmc_step_size
        if args.mcmc_noise_var is not None:
            trainer_cfg.metropolis.noise_var = args.mcmc_noise_var
        if args.mcmc_prior_var is not None:
            trainer_cfg.metropolis.prior_var = args.mcmc_prior_var
    if args.method == "hmc":
        if args.mcmc_samples is not None:
            trainer_cfg.hmc.n_samples = args.mcmc_samples
        if args.mcmc_burn_in is not None:
            trainer_cfg.hmc.burn_in = args.mcmc_burn_in
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
    print(json.dumps(result["summary"], indent=2))
    if "metrics" in result:
        print("Metrics:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")
    print("Generated artefacts:")
    for name, path in sorted(result["plots"].items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())

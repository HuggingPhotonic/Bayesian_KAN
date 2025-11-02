# Bayesian KAN Experiments

This repository contains a modular implementation of Kolmogorov–Arnold Networks
with Bayesian inference backends (variational inference, Laplace, Metropolis,
HMC) and deterministic baselines.  The codebase is organised so that both
single-run experiments and automated benchmark sweeps share the same
infrastructure.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is listed as a dependency; feel free to adjust the version / CUDA build
to match your environment.

## Quick Start

### Single Experiment

Run a VI experiment on the 1D `sin_cos` target:

```bash
python run.py \
  --method vi \
  --basis bspline \
  --dim 1 \
  --targets sin_cos \
  --layer-sizes 64,64,1 \
  --n-train 512 \
  --eval-points-1d 400 \
  --basis-kw grid_size=50 \
  --basis-kw spline_order=10 \
  --basis-kw prior_scale=0.5
```

Common CLI flags include:

- `--method {vi, laplace, metropolis, hmc, deterministic}`
- `--basis {bspline, chebyshev, wavelet, rbf}`
- `--basis-kw key=value` (repeatable) for per-basis tuning
- Dataset controls: `--n-train`, `--eval-points-1d`, `--eval-points-2d`, `--n-eval`
- Inference overrides (e.g. `--vi-epochs`, `--laplace-damping`, `--mcmc-step-size`)

Outputs (summaries, plots) are written to `results/<method>_<basis>_<dim>d/`.

### Benchmark Sweep

Use the provided shell script to run a batch of targets × methods:

```bash
bash run_benchmark.sh
```

The script exposes every hyperparameter via environment variables.  For
example, to change the spline grid size and disable MAP warm-up:

```bash
BASIS_GRID_SIZE=80 \
NO_MAP_WARMUP=1 \
bash run_benchmark.sh
```

You can still append extra flags, which are forwarded to `benchmark.py`:

```bash
bash run_benchmark.sh --targets sin_cos --methods vi,laplace
```

Benchmark artefacts (CSV metric tables + bar charts) are stored under
`results/benchmark/<timestamp>/`.

## Code Structure

- `targets/` — analytic target functions and metadata (domains, descriptions)
- `bases/` — basis function implementations (B-spline, Chebyshev, Wavelet, RBF)
- `models/` — KAN model builder that stitches basis layers
- `training/` — dataset generation, training loops, metrics aggregation
- `inference/` — VI / Laplace / Metropolis / HMC algorithms
- `visualization/` — plotting utilities (training curves, 1D & 2D visualisations)
- `run.py` — single experiment entrypoint
- `benchmark.py` — multi-target benchmarking entrypoint
- `run_benchmark.sh` — convenient wrapper around `benchmark.py`

## Customising Experiments

1. **Basis Hyperparameters**  
   Every basis accepts `--basis-kw`:
   - B-spline: `grid_size`, `spline_order`, `prior_scale`
   - Chebyshev: `degree`, `prior_scale`
   - Wavelet: `levels`, `prior_scale`
   - RBF: `num_centers`, `prior_scale`

2. **Inference Hyperparameters**  
   For VI you can tune learning rate, KL warm-up schedule, sample counts, etc.
   Laplace exposes MAP optimiser settings (`--laplace-lr`, `--laplace-epochs`,
   `--laplace-damping` …).  Metropolis / HMC accept step-size, burn-in, noise /
   prior variance overrides.

3. **Deterministic Baseline**  
   Select `--method deterministic` to train a standard KAN.  Scheduler,
   learning rate, epochs, and gradient clipping can all be changed from the CLI.

## Example: Custom Benchmark

```bash
BASIS_TYPE=wavelet \
WAVELET_LEVELS=4 \
WAVELET_PRIOR_SCALE=0.3 \
TRAIN_SAMPLES=1024 \
VI_EPOCHS=3000 \
bash run_benchmark.sh --targets sin_cos,wave --methods vi,hmc
```

This launches VI and HMC runs on two targets with the specified wavelet basis
and training schedule.

## Packaging

The optional `setup.py` is provided if you need to install the project as a
package:

```bash
pip install -e .
```

## License

MIT License (see `LICENSE` if present; otherwise adapt as needed).

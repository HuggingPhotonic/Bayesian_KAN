#!/usr/bin/env bash

# Simple wrapper to launch the benchmark suite with default arguments.
# Extra CLI flags can be supplied when invoking this script; they will be
# forwarded to benchmark.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-512}"
EVAL_POINTS_1D="${EVAL_POINTS_1D:-400}"
EVAL_POINTS_2D="${EVAL_POINTS_2D:-60}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
VI_EPOCHS="${VI_EPOCHS:-2000}"
VI_LR="${VI_LR:-5e-4}"
VI_WEIGHT_DECAY="${VI_WEIGHT_DECAY:-1e-4}"
VI_KL_MAX="${VI_KL_MAX:-1e-3}"
VI_KL_WARMUP="${VI_KL_WARMUP:-300}"
VI_SCHEDULER_STEP="${VI_SCHEDULER_STEP:-300}"
VI_SCHEDULER_GAMMA="${VI_SCHEDULER_GAMMA:-0.5}"
VI_N_SAMPLES="${VI_N_SAMPLES:-8}"
VI_GRAD_CLIP="${VI_GRAD_CLIP:-1.0}"
VI_EVAL_SAMPLES="${VI_EVAL_SAMPLES:-200}"
LAPLACE_EPOCHS="${LAPLACE_EPOCHS:-500}"
LAPLACE_LR="${LAPLACE_LR:-5e-4}"
LAPLACE_WEIGHT_DECAY="${LAPLACE_WEIGHT_DECAY:-2e-4}"
LAPLACE_NOISE_VAR="${LAPLACE_NOISE_VAR:-0.05}"
LAPLACE_PRIOR_VAR="${LAPLACE_PRIOR_VAR:-1.0}"
LAPLACE_DAMPING="${LAPLACE_DAMPING:-1e-2}"
LAPLACE_N_SAMPLES="${LAPLACE_N_SAMPLES:-200}"
LAPLACE_GRAD_CLIP="${LAPLACE_GRAD_CLIP:-1.0}"
MCMC_SAMPLES="${MCMC_SAMPLES:-800}"
MCMC_BURN_IN="${MCMC_BURN_IN:-600}"
MCMC_STEP_SIZE="${MCMC_STEP_SIZE:-2e-4}"
MCMC_NOISE_VAR="${MCMC_NOISE_VAR:-0.05}"
MCMC_PRIOR_VAR="${MCMC_PRIOR_VAR:-1.0}"
HMC_STEP_SIZE="${HMC_STEP_SIZE:-5e-4}"
HMC_LEAPFROG="${HMC_LEAPFROG:-30}"
HMC_NOISE_VAR="${HMC_NOISE_VAR:-0.05}"
HMC_PRIOR_VAR="${HMC_PRIOR_VAR:-1.0}"
DET_EPOCHS="${DET_EPOCHS:-1500}"
DET_LR="${DET_LR:-5e-4}"
DET_WEIGHT_DECAY="${DET_WEIGHT_DECAY:-1e-4}"
DET_SCHEDULER_STEP="${DET_SCHEDULER_STEP:-400}"
DET_SCHEDULER_GAMMA="${DET_SCHEDULER_GAMMA:-0.5}"
DET_GRAD_CLIP="${DET_GRAD_CLIP:-1.0}"
BASIS_TYPE="${BASIS_TYPE:-bspline}"
BASIS_GRID_SIZE="${BASIS_GRID_SIZE:-50}"
BASIS_SPLINE_ORDER="${BASIS_SPLINE_ORDER:-10}"
BASIS_PRIOR_SCALE="${BASIS_PRIOR_SCALE:-0.5}"
CHEBYSHEV_DEGREE="${CHEBYSHEV_DEGREE:-20}"
CHEBYSHEV_PRIOR_SCALE="${CHEBYSHEV_PRIOR_SCALE:-0.5}"
WAVELET_LEVELS="${WAVELET_LEVELS:-3}"
WAVELET_PRIOR_SCALE="${WAVELET_PRIOR_SCALE:-0.5}"
RBF_NUM_CENTERS="${RBF_NUM_CENTERS:-32}"
RBF_PRIOR_SCALE="${RBF_PRIOR_SCALE:-0.5}"
NO_MAP_WARMUP="${NO_MAP_WARMUP:-0}"

CMD=(
  python3 "${REPO_ROOT}/benchmark.py"
  --targets "sin_cos,wave"
  --methods "vi,laplace,metropolis,hmc"
  --basis "${BASIS_TYPE}"
  --layer-sizes "64,64,1"
  --n-train "${TRAIN_SAMPLES}"
  --eval-points-1d "${EVAL_POINTS_1D}"
  --eval-points-2d "${EVAL_POINTS_2D}"
  --n-eval "${EVAL_SAMPLES}"
  --device "auto"
  --seed 42
  --vi-epochs "${VI_EPOCHS}"
  --vi-lr "${VI_LR}"
  --vi-weight-decay "${VI_WEIGHT_DECAY}"
  --vi-kl-max "${VI_KL_MAX}"
  --vi-kl-warmup "${VI_KL_WARMUP}"
  --vi-scheduler-step "${VI_SCHEDULER_STEP}"
  --vi-scheduler-gamma "${VI_SCHEDULER_GAMMA}"
  --vi-n-samples "${VI_N_SAMPLES}"
  --vi-grad-clip "${VI_GRAD_CLIP}"
  --vi-eval-samples "${VI_EVAL_SAMPLES}"
  --laplace-epochs "${LAPLACE_EPOCHS}"
  --laplace-lr "${LAPLACE_LR}"
  --laplace-weight-decay "${LAPLACE_WEIGHT_DECAY}"
  --laplace-noise-var "${LAPLACE_NOISE_VAR}"
  --laplace-prior-var "${LAPLACE_PRIOR_VAR}"
  --laplace-damping "${LAPLACE_DAMPING}"
  --laplace-n-samples "${LAPLACE_N_SAMPLES}"
  --laplace-grad-clip "${LAPLACE_GRAD_CLIP}"
  --mcmc-samples "${MCMC_SAMPLES}"
  --mcmc-burn-in "${MCMC_BURN_IN}"
  --mcmc-step-size "${MCMC_STEP_SIZE}"
  --mcmc-noise-var "${MCMC_NOISE_VAR}"
  --mcmc-prior-var "${MCMC_PRIOR_VAR}"
  --hmc-step-size "${HMC_STEP_SIZE}"
  --hmc-leapfrog "${HMC_LEAPFROG}"
  --hmc-noise-var "${HMC_NOISE_VAR}"
  --hmc-prior-var "${HMC_PRIOR_VAR}"
  --det-epochs "${DET_EPOCHS}"
  --det-lr "${DET_LR}"
  --det-weight-decay "${DET_WEIGHT_DECAY}"
  --det-scheduler-step "${DET_SCHEDULER_STEP}"
  --det-scheduler-gamma "${DET_SCHEDULER_GAMMA}"
  --det-grad-clip "${DET_GRAD_CLIP}"
)

case "${BASIS_TYPE}" in
  bspline)
    CMD+=(--basis-kw "grid_size=${BASIS_GRID_SIZE}")
    CMD+=(--basis-kw "spline_order=${BASIS_SPLINE_ORDER}")
    CMD+=(--basis-kw "prior_scale=${BASIS_PRIOR_SCALE}")
    ;;
  chebyshev)
    CMD+=(--basis-kw "degree=${CHEBYSHEV_DEGREE}")
    CMD+=(--basis-kw "prior_scale=${CHEBYSHEV_PRIOR_SCALE}")
    ;;
  wavelet)
    CMD+=(--basis-kw "levels=${WAVELET_LEVELS}")
    CMD+=(--basis-kw "prior_scale=${WAVELET_PRIOR_SCALE}")
    ;;
  rbf)
    CMD+=(--basis-kw "num_centers=${RBF_NUM_CENTERS}")
    CMD+=(--basis-kw "prior_scale=${RBF_PRIOR_SCALE}")
    ;;
esac

if [[ "${NO_MAP_WARMUP}" != "0" ]]; then
  CMD+=(--no-map-warmup)
fi

CMD+=("$@")

"${CMD[@]}"

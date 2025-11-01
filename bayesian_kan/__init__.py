"""
Bayesian Kolmogorov-Arnold Networks (B-KAN)
A complete implementation with MCMC, VI, and Laplace approximation
"""

from .config import DEVICE, get_device
from .bspline import BSplineBasis
from .model import BayesianKANEdge, BayesianKANLayer, BayesianKAN
from .variational_inference import VariationalInference
from .mcmc_sampling import MCMCSampler
from .laplace_approximation import LaplaceApproximation
from .visualization import BayesianKANVisualizer
from .utils import (
    generate_synthetic_data,
    set_seed,
    compute_metrics,
    create_data_loaders
)

__version__ = "1.0.0"
__author__ = "Bayesian KAN Team"

__all__ = [
    # Core components
    'BayesianKAN',
    'BayesianKANLayer',
    'BayesianKANEdge',
    'BSplineBasis',
    
    # Inference methods
    'VariationalInference',
    'MCMCSampler',
    'LaplaceApproximation',
    
    # Visualization
    'BayesianKANVisualizer',
    
    # Utilities
    'generate_synthetic_data',
    'set_seed',
    'compute_metrics',
    'create_data_loaders',
    'get_device',
    'DEVICE',
]
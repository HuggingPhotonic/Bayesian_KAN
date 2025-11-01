"""
Bayesian KAN Layers and Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .bspline import BSplineBasis


class BayesianKANEdge(nn.Module):
    """Single Bayesian KAN edge with probabilistic spline parameters"""
    
    def __init__(self, n_basis: int = 10, degree: int = 3,
                 prior_scale: float = 1.0):
        """
        Initialize Bayesian KAN edge
        
        Args:
            n_basis: number of B-spline basis functions
            degree: B-spline degree
            prior_scale: scale of the prior distribution
        """
        super().__init__()
        self.n_basis = n_basis
        self.degree = degree
        
        # Variational parameters for coefficients
        self.coeff_mean = nn.Parameter(torch.randn(n_basis) * 0.1)
        self.coeff_log_var = nn.Parameter(torch.ones(n_basis) * -2)
        
        # Prior
        self.prior_scale = prior_scale
        self.register_buffer('prior_mean', torch.zeros(n_basis))
        self.register_buffer('prior_var', torch.ones(n_basis) * prior_scale**2)
        
    def forward(self, x: torch.Tensor, sample: bool = False,
                n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional sampling
        
        Args:
            x: input tensor
            sample: whether to sample from the posterior
            n_samples: number of samples to draw
            
        Returns:
            (output, kl_divergence)
        """
        # Get basis functions
        basis = BSplineBasis.evaluate_basis(x, self.degree, self.n_basis)
        
        if sample:
            # Sample coefficients
            std = torch.exp(0.5 * self.coeff_log_var)
            eps = torch.randn(n_samples, *std.shape).to(x.device)
            coeffs = self.coeff_mean + eps * std
            
            # Compute outputs for each sample
            outputs = []
            for i in range(n_samples):
                output = torch.sum(basis * coeffs[i], dim=-1)
                outputs.append(output)
            output = torch.stack(outputs).mean(0) if n_samples > 1 else outputs[0]
        else:
            # Use mean coefficients
            output = torch.sum(basis * self.coeff_mean, dim=-1)
        
        # Compute KL divergence
        kl = self.compute_kl()
        
        return output, kl
    
    def compute_kl(self) -> torch.Tensor:
        """Compute KL divergence from prior"""
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            var / self.prior_var + 
            (self.coeff_mean - self.prior_mean)**2 / self.prior_var -
            1 - torch.log(var / self.prior_var)
        )
        return kl


class BayesianKANLayer(nn.Module):
    """Bayesian KAN layer with multiple edges"""
    
    def __init__(self, in_features: int, out_features: int,
                 n_basis: int = 10, degree: int = 3,
                 prior_scale: float = 1.0):
        """
        Initialize Bayesian KAN layer
        
        Args:
            in_features: number of input features
            out_features: number of output features
            n_basis: number of B-spline basis functions
            degree: B-spline degree
            prior_scale: scale of the prior distribution
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create edge grid
        self.edges = nn.ModuleList([
            nn.ModuleList([
                BayesianKANEdge(n_basis, degree, prior_scale)
                for _ in range(in_features)
            ])
            for _ in range(out_features)
        ])
        
    def forward(self, x: torch.Tensor, sample: bool = True,
                n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through layer
        
        Args:
            x: input tensor of shape (batch_size, in_features)
            sample: whether to sample from the posterior
            n_samples: number of samples to draw
            
        Returns:
            (output, total_kl_divergence)
        """
        batch_size = x.shape[0]
        outputs = []
        total_kl = 0
        
        for out_idx in range(self.out_features):
            out_val = 0
            for in_idx in range(self.in_features):
                edge_out, kl = self.edges[out_idx][in_idx](
                    x[:, in_idx], sample, n_samples
                )
                out_val = out_val + edge_out
                total_kl = total_kl + kl
            outputs.append(out_val)
        
        output = torch.stack(outputs, dim=1)
        return output, total_kl


class BayesianKAN(nn.Module):
    """Complete Bayesian KAN network"""
    
    def __init__(self, layer_sizes: List[int], n_basis: int = 10,
                 degree: int = 3, prior_scale: float = 1.0):
        """
        Initialize Bayesian KAN network
        
        Args:
            layer_sizes: list of layer sizes (e.g., [1, 10, 10, 1])
            n_basis: number of B-spline basis functions
            degree: B-spline degree
            prior_scale: scale of the prior distribution
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        
        # Build layers
        self.layers = nn.ModuleList([
            BayesianKANLayer(layer_sizes[i], layer_sizes[i+1],
                           n_basis, degree, prior_scale)
            for i in range(len(layer_sizes) - 1)
        ])
        
        # Track metrics
        self.training_history = {
            'loss': [], 'kl': [], 'likelihood': [],
            'val_loss': [], 'uncertainties': []
        }
        
    def forward(self, x: torch.Tensor, sample: bool = True,
                n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            x: input tensor
            sample: whether to sample from the posterior
            n_samples: number of samples to draw
            
        Returns:
            (output, total_kl_divergence)
        """
        total_kl = 0
        
        for layer in self.layers:
            x, kl = layer(x, sample, n_samples)
            total_kl = total_kl + kl
        
        return x, total_kl
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                 n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation
        
        Args:
            x: input tensor
            n_samples: number of samples for uncertainty estimation
            
        Returns:
            (mean_prediction, std_prediction)
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.forward(x, sample=True, n_samples=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(0)
        std = predictions.std(0)
        
        return mean, std

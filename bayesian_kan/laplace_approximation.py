"""
Laplace Approximation for Bayesian KAN
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from tqdm import tqdm

from .model import BayesianKAN
from .config import DEVICE


class LaplaceApproximation:
    """Laplace approximation for posterior inference"""
    
    @staticmethod
    def approximate(model: BayesianKAN,
                   data: Tuple[torch.Tensor, torch.Tensor],
                   n_iterations: int = 10,
                   prior_weight: float = 0.01) -> Dict:
        """
        Compute Laplace approximation for posterior
        
        Args:
            model: BayesianKAN model
            data: tuple of (X, y) tensors
            n_iterations: number of optimization iterations
            prior_weight: weight for prior term
            
        Returns:
            Dictionary containing MAP estimate and posterior covariance
        """
        X, y = data
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        # First optimize to find MAP estimate
        optimizer = torch.optim.LBFGS(model.parameters(), 
                                      max_iter=20,
                                      line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            pred, kl = model(X, sample=False)
            loss = F.mse_loss(pred, y) + prior_weight * kl
            loss.backward()
            return loss
        
        # Find MAP
        print("Finding MAP estimate...")
        for i in tqdm(range(n_iterations), desc="MAP Optimization"):
            loss = optimizer.step(closure)
        
        print(f"Final MAP loss: {loss.item():.6f}")
        
        # Compute Hessian at MAP
        pred, kl = model(X, sample=False)
        loss = F.mse_loss(pred, y) + prior_weight * kl
        
        # Get parameters
        params = list(model.parameters())
        n_params = sum(p.numel() for p in params)
        
        print(f"Computing Hessian for {n_params} parameters...")
        
        # Compute Hessian (diagonal approximation for efficiency)
        hessian_diag = LaplaceApproximation._compute_diagonal_hessian(
            model, X, y, prior_weight
        )
        
        # Posterior covariance (inverse Hessian)
        posterior_var = 1.0 / (hessian_diag + 1e-6)
        
        return {
            'map_params': [p.clone().cpu() for p in params],
            'posterior_variance': posterior_var.cpu(),
            'hessian_diagonal': hessian_diag.cpu(),
            'map_loss': loss.item()
        }
    
    @staticmethod
    def _compute_diagonal_hessian(model: BayesianKAN,
                                  X: torch.Tensor,
                                  y: torch.Tensor,
                                  prior_weight: float) -> torch.Tensor:
        """
        Compute diagonal of Hessian matrix
        
        Args:
            model: BayesianKAN model
            X: input data
            y: target data
            prior_weight: weight for prior term
            
        Returns:
            Diagonal of Hessian matrix
        """
        params = list(model.parameters())
        hessian_diag = []
        
        # Compute loss
        pred, kl = model(X, sample=False)
        loss = F.mse_loss(pred, y) + prior_weight * kl
        
        # For each parameter
        for p in tqdm(params, desc="Computing Hessian", leave=False):
            # First derivative
            grad = torch.autograd.grad(loss, p, create_graph=True, 
                                      retain_graph=True)[0]
            
            # Diagonal of second derivative (approximate)
            # We compute ||∇²f||_diag ≈ ||∇f||² (Fisher approximation)
            hess_p = grad.flatten() ** 2
            hessian_diag.append(hess_p)
        
        hessian_diag = torch.cat(hessian_diag)
        
        return hessian_diag
    
    @staticmethod
    def sample_from_posterior(map_params: List[torch.Tensor],
                             posterior_var: torch.Tensor,
                             n_samples: int = 100) -> List[List[torch.Tensor]]:
        """
        Sample from Laplace posterior
        
        Args:
            map_params: MAP parameter estimates
            posterior_var: posterior variance (diagonal)
            n_samples: number of samples to draw
            
        Returns:
            List of parameter samples
        """
        samples = []
        
        # Split variance back into parameter shapes
        var_idx = 0
        param_vars = []
        for p in map_params:
            n_elements = p.numel()
            param_var = posterior_var[var_idx:var_idx + n_elements].reshape(p.shape)
            param_vars.append(param_var)
            var_idx += n_elements
        
        # Sample
        for _ in range(n_samples):
            sample = []
            for mean, var in zip(map_params, param_vars):
                std = torch.sqrt(var)
                param_sample = mean + std * torch.randn_like(mean)
                sample.append(param_sample)
            samples.append(sample)
        
        return samples

"""
MCMC (Hamiltonian Monte Carlo) Sampling for Bayesian KAN
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from tqdm import tqdm

from .model import BayesianKAN
from .config import DEVICE


class MCMCSampler:
    """MCMC sampling using Hamiltonian Monte Carlo for Bayesian KAN"""
    
    @staticmethod
    def sample(model: BayesianKAN,
              data: Tuple[torch.Tensor, torch.Tensor],
              n_samples: int = 1000,
              warmup: int = 100,
              step_size: float = 0.01,
              n_leapfrog: int = 5) -> Dict:
        """
        Sample from posterior using Hamiltonian Monte Carlo
        
        Args:
            model: BayesianKAN model
            data: tuple of (X, y) tensors
            n_samples: number of samples to draw
            warmup: number of warmup iterations
            step_size: HMC step size
            n_leapfrog: number of leapfrog steps
            
        Returns:
            Dictionary containing samples and diagnostics
        """
        X, y = data
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        # Initialize samples storage
        param_samples = []
        log_probs = []
        acceptance_rate = []
        
        # Get initial parameters
        params = list(model.parameters())
        
        n_accepted = 0
        
        # HMC sampling
        for iteration in tqdm(range(n_samples + warmup), desc="MCMC Sampling"):
            # Store current parameters
            current_params = [p.clone() for p in params]
            
            # Sample momentum
            momentum = [torch.randn_like(p) for p in params]
            current_momentum = [m.clone() for m in momentum]
            
            # Compute initial energy
            pred, kl = model(X, sample=False)
            likelihood = -F.mse_loss(pred, y, reduction='sum')
            current_energy = -(likelihood - kl)
            current_kinetic = sum(0.5 * torch.sum(m**2) for m in momentum)
            current_total = current_energy + current_kinetic
            
            # Leapfrog integration
            # Half step for momentum
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            
            pred, kl = model(X, sample=False)
            likelihood = -F.mse_loss(pred, y, reduction='sum')
            potential = -(likelihood - kl)
            potential.backward()
            
            for p, m in zip(params, momentum):
                if p.grad is not None:
                    m.data -= step_size * p.grad / 2
            
            # Full steps
            for _ in range(n_leapfrog - 1):
                # Update position
                for p, m in zip(params, momentum):
                    p.data += step_size * m
                
                # Update momentum
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                
                pred, kl = model(X, sample=False)
                likelihood = -F.mse_loss(pred, y, reduction='sum')
                potential = -(likelihood - kl)
                potential.backward()
                
                for p, m in zip(params, momentum):
                    if p.grad is not None:
                        m.data -= step_size * p.grad
            
            # Final position update
            for p, m in zip(params, momentum):
                p.data += step_size * m
            
            # Final momentum half step
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            
            pred, kl = model(X, sample=False)
            likelihood = -F.mse_loss(pred, y, reduction='sum')
            potential = -(likelihood - kl)
            potential.backward()
            
            for p, m in zip(params, momentum):
                if p.grad is not None:
                    m.data -= step_size * p.grad / 2
            
            # Compute new energy
            pred, kl = model(X, sample=False)
            likelihood = -F.mse_loss(pred, y, reduction='sum')
            new_energy = -(likelihood - kl)
            new_kinetic = sum(0.5 * torch.sum(m**2) for m in momentum)
            new_total = new_energy + new_kinetic
            
            # Metropolis acceptance
            accept_prob = torch.exp(-(new_total - current_total))
            
            if torch.rand(1).item() < accept_prob.item():
                # Accept
                n_accepted += 1
            else:
                # Reject: restore parameters
                for p, old_p in zip(params, current_params):
                    p.data = old_p
                new_energy = current_energy
            
            # Store samples after warmup
            if iteration >= warmup:
                param_samples.append([p.clone().cpu() for p in params])
                log_probs.append(-new_energy.item())
                acceptance_rate.append(n_accepted / (iteration + 1))
        
        final_acceptance = n_accepted / (n_samples + warmup)
        
        return {
            'samples': param_samples,
            'log_probs': log_probs,
            'acceptance_rate': acceptance_rate,
            'final_acceptance': final_acceptance
        }
    
    @staticmethod
    def compute_statistics(samples: List[List[torch.Tensor]]) -> Dict:
        """
        Compute statistics from MCMC samples
        
        Args:
            samples: list of parameter samples
            
        Returns:
            Dictionary containing sample statistics
        """
        # Stack all samples
        n_samples = len(samples)
        n_params = len(samples[0])
        
        # Compute mean and std for each parameter
        param_means = []
        param_stds = []
        
        for param_idx in range(n_params):
            param_samples = torch.stack([s[param_idx] for s in samples])
            param_means.append(param_samples.mean(0))
            param_stds.append(param_samples.std(0))
        
        return {
            'means': param_means,
            'stds': param_stds,
            'n_samples': n_samples
        }

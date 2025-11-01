"""
Utility functions for Bayesian KAN
"""

import torch
import numpy as np
from typing import Tuple, Optional

from .config import DEVICE


def generate_synthetic_data(n_samples: int = 1000, 
                           input_dim: int = 1,
                           noise_std: float = 0.1,
                           seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for testing
    
    Args:
        n_samples: number of samples
        input_dim: input dimensionality
        noise_std: standard deviation of noise
        seed: random seed for reproducibility
        
    Returns:
        Tuple of (X, y) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    X = torch.randn(n_samples, input_dim) * 2
    
    if input_dim == 1:
        # 1D function: combination of sinusoids
        y = torch.sin(X) + 0.5 * torch.cos(2 * X) + noise_std * torch.randn_like(X)
    else:
        # Multi-dimensional: nonlinear combination
        y = torch.sum(torch.sin(X), dim=1, keepdim=True) + \
            torch.sum(X**2, dim=1, keepdim=True) * 0.1
        y += noise_std * torch.randn_like(y)
    
    return X.to(DEVICE), y.to(DEVICE)


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   uncertainties: Optional[torch.Tensor] = None) -> dict:
    """
    Compute evaluation metrics
    
    Args:
        predictions: predicted values
        targets: true values
        uncertainties: prediction uncertainties (optional)
        
    Returns:
        Dictionary of metrics
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    # Add uncertainty-based metrics if available
    if uncertainties is not None:
        # Calibration: check if true values fall within uncertainty bounds
        lower = predictions - 2 * uncertainties
        upper = predictions + 2 * uncertainties
        in_bounds = ((targets >= lower) & (targets <= upper)).float().mean().item()
        
        metrics['calibration_95'] = in_bounds
        metrics['mean_uncertainty'] = uncertainties.mean().item()
        metrics['std_uncertainty'] = uncertainties.std().item()
    
    return metrics


def create_data_loaders(X_train: torch.Tensor, 
                       y_train: torch.Tensor,
                       X_val: Optional[torch.Tensor] = None,
                       y_val: Optional[torch.Tensor] = None,
                       batch_size: int = 32,
                       shuffle: bool = True) -> tuple:
    """
    Create PyTorch data loaders
    
    Args:
        X_train: training inputs
        y_train: training targets
        X_val: validation inputs (optional)
        y_val: validation targets (optional)
        batch_size: batch size
        shuffle: whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader) or just train_loader
    """
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    
    if X_val is not None and y_val is not None:
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader
    
    return train_loader, None

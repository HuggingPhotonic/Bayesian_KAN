"""
Variational Inference for Bayesian KAN
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict
from tqdm import tqdm

from .model import BayesianKAN
from .config import DEVICE


class VariationalInference:
    """Variational Inference optimization for Bayesian KAN"""
    
    @staticmethod
    def train(model: BayesianKAN, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: Optional[torch.utils.data.DataLoader] = None,
             epochs: int = 100,
             lr: float = 1e-3,
             kl_weight: float = 1.0) -> Dict:
        """
        Train model using Variational Inference
        
        Args:
            model: BayesianKAN model
            train_loader: training data loader
            val_loader: validation data loader (optional)
            epochs: number of training epochs
            lr: learning rate
            kl_weight: weight for KL divergence term
            
        Returns:
            Dictionary containing training history
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {'loss': [], 'kl': [], 'likelihood': [], 'val_loss': []}
        
        model.train()
        for epoch in tqdm(range(epochs), desc="VI Training"):
            epoch_loss = 0
            epoch_kl = 0
            epoch_likelihood = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                # Forward pass
                pred, kl = model(batch_x, sample=True)
                
                # Compute loss
                likelihood = -F.mse_loss(pred, batch_y, reduction='sum')
                loss = -(likelihood - kl_weight * kl) / len(batch_x)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_kl += kl.item()
                epoch_likelihood += likelihood.item()
            
            # Average over batches
            n_batches = len(train_loader)
            history['loss'].append(epoch_loss / n_batches)
            history['kl'].append(epoch_kl / n_batches)
            history['likelihood'].append(epoch_likelihood / n_batches)
            
            # Validation
            if val_loader is not None:
                val_loss = VariationalInference._validate(model, val_loader)
                history['val_loss'].append(val_loss)
        
        return history
    
    @staticmethod
    def _validate(model: BayesianKAN, 
                 val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validation step
        
        Args:
            model: BayesianKAN model
            val_loader: validation data loader
            
        Returns:
            Average validation loss
        """
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                pred, _ = model(batch_x, sample=False)
                val_loss += F.mse_loss(pred, batch_y).item()
        
        model.train()
        return val_loss / len(val_loader)

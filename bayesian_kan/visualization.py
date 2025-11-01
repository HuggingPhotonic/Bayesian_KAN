"""
Visualization Tools for Bayesian KAN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import os
from datetime import datetime
import json

from .model import BayesianKAN
from .config import DEVICE


class BayesianKANVisualizer:
    """Comprehensive visualization for Bayesian KAN"""
    
    def __init__(self, save_dir: str = "bayesian_kan_results"):
        """
        Initialize visualizer
        
        Args:
            save_dir: directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_training_history(self, history: Dict, save: bool = True):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history.get('loss', []), label='Training Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', 
                          linewidth=2, linestyle='--')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL Divergence
        if 'kl' in history and history['kl']:
            axes[0, 1].plot(history['kl'], label='KL Divergence', 
                          linewidth=2, color='orange')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('KL Divergence')
            axes[0, 1].set_title('KL Divergence Evolution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Likelihood
        if 'likelihood' in history and history['likelihood']:
            axes[1, 0].plot(history['likelihood'], label='Log Likelihood', 
                          linewidth=2, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Log Likelihood')
            axes[1, 0].set_title('Likelihood Evolution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined normalized view
        if 'loss' in history and 'kl' in history and history['loss'] and history['kl']:
            loss_norm = np.array(history['loss']) / (np.max(history['loss']) + 1e-8)
            kl_norm = np.array(history['kl']) / (np.max(history['kl']) + 1e-8)
            axes[1, 1].plot(loss_norm, label='Normalized Loss', linewidth=2)
            axes[1, 1].plot(kl_norm, label='Normalized KL', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Normalized Value')
            axes[1, 1].set_title('Normalized Metrics Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/training_history_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_predictions_from_stats(self,
                                    X: torch.Tensor,
                                    mean: torch.Tensor,
                                    std: torch.Tensor,
                                    y_test: Optional[torch.Tensor] = None,
                                    save: bool = True):
        """Plot predictions using precomputed mean/std values (e.g., from MCMC)."""
        def _to_cpu(tensor):
            if torch.is_tensor(tensor):
                return tensor.detach().cpu()
            return torch.as_tensor(tensor) if tensor is not None else None
        
        X_cpu = _to_cpu(X)
        mean_cpu = _to_cpu(mean)
        std_cpu = _to_cpu(std)
        y_cpu = _to_cpu(y_test)
        
        if mean_cpu.dim() == 1:
            mean_cpu = mean_cpu.unsqueeze(-1)
        if std_cpu.dim() == 1:
            std_cpu = std_cpu.unsqueeze(-1)
        
        if X_cpu.shape[1] == 1:
            sort_idx = X_cpu[:, 0].argsort()
            X_sorted = X_cpu[sort_idx, 0]
            mean_sorted = mean_cpu[sort_idx, 0]
            std_sorted = std_cpu[sort_idx, 0]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(X_sorted, mean_sorted, 'b-', linewidth=2, label='Mean Prediction')
            ax.fill_between(X_sorted,
                            mean_sorted - 2 * std_sorted,
                            mean_sorted + 2 * std_sorted,
                            alpha=0.3,
                            label='±2σ Uncertainty')
            ax.fill_between(X_sorted,
                            mean_sorted - std_sorted,
                            mean_sorted + std_sorted,
                            alpha=0.5,
                            label='±1σ Uncertainty')
            
            if y_cpu is not None:
                ax.scatter(X_cpu[:, 0], y_cpu[:, 0], c='red',
                           alpha=0.5, s=20, label='True Values')
            
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.set_title('Predictions with Uncertainty')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save:
                filename = f"{self.save_dir}/predictions_uncertainty_{self.timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")
            
            plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].hist(std_cpu.flatten().numpy(), bins=50, alpha=0.7,
                         color='blue', edgecolor='black')
            axes[0].set_xlabel('Uncertainty (std)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Prediction Uncertainties')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].scatter(mean_cpu.flatten().numpy(), std_cpu.flatten().numpy(),
                            alpha=0.5, s=10)
            axes[1].set_xlabel('Mean Prediction')
            axes[1].set_ylabel('Uncertainty (std)')
            axes[1].set_title('Prediction Mean vs Uncertainty')
            axes[1].grid(True, alpha=0.3)
            
            if save:
                filename = f"{self.save_dir}/uncertainty_analysis_{self.timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")
            
            plt.show()
    
    def plot_predictions_with_uncertainty(self, model: BayesianKAN,
                                         X_test: torch.Tensor,
                                         y_test: Optional[torch.Tensor] = None,
                                         n_samples: int = 100,
                                         save: bool = True):
        """Plot predictions with uncertainty bands"""
        model.eval()
        X_test = X_test.to(DEVICE)
        
        # Get predictions with uncertainty
        mean, std = model.predict_with_uncertainty(X_test, n_samples)
        mean, std = mean.cpu(), std.cpu()
        X_test = X_test.cpu()
        
        # Sort for better visualization (1D case)
        if X_test.shape[1] == 1:
            sort_idx = X_test[:, 0].argsort()
            X_sorted = X_test[sort_idx, 0]
            mean_sorted = mean[sort_idx, 0]
            std_sorted = std[sort_idx, 0]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot predictions with uncertainty
            ax.plot(X_sorted, mean_sorted, 'b-', linewidth=2, label='Mean Prediction')
            ax.fill_between(X_sorted,
                           mean_sorted - 2*std_sorted,
                           mean_sorted + 2*std_sorted,
                           alpha=0.3, label='±2σ Uncertainty')
            ax.fill_between(X_sorted,
                           mean_sorted - std_sorted,
                           mean_sorted + std_sorted,
                           alpha=0.5, label='±1σ Uncertainty')
            
            # Plot true values if available
            if y_test is not None:
                y_test = y_test.cpu()
                ax.scatter(X_test[:, 0], y_test[:, 0], c='red', 
                         alpha=0.5, s=20, label='True Values')
            
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.set_title('Bayesian KAN Predictions with Uncertainty')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save:
                filename = f"{self.save_dir}/predictions_uncertainty_{self.timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")
            
            plt.show()
        
        # For multi-dimensional inputs
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Uncertainty histogram
            axes[0].hist(std.flatten().numpy(), bins=50, alpha=0.7, 
                        color='blue', edgecolor='black')
            axes[0].set_xlabel('Uncertainty (std)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Prediction Uncertainties')
            axes[0].grid(True, alpha=0.3)
            
            # Mean vs uncertainty scatter
            axes[1].scatter(mean.flatten().numpy(), std.flatten().numpy(),
                          alpha=0.5, s=10)
            axes[1].set_xlabel('Mean Prediction')
            axes[1].set_ylabel('Uncertainty (std)')
            axes[1].set_title('Prediction Mean vs Uncertainty')
            axes[1].grid(True, alpha=0.3)
            
            if save:
                filename = f"{self.save_dir}/uncertainty_analysis_{self.timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")
            
            plt.show()
    
    def plot_parameter_distributions(self, model: BayesianKAN, save: bool = True):
        """Visualize parameter distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        all_means = []
        all_vars = []
        
        # Collect all parameters
        for layer_idx, layer in enumerate(model.layers):
            for out_idx in range(layer.out_features):
                for in_idx in range(layer.in_features):
                    edge = layer.edges[out_idx][in_idx]
                    all_means.extend(edge.coeff_mean.detach().cpu().numpy())
                    all_vars.extend(torch.exp(edge.coeff_log_var).detach().cpu().numpy())
        
        all_means = np.array(all_means)
        all_vars = np.array(all_vars)
        all_stds = np.sqrt(all_vars)
        
        # Mean distribution
        axes[0, 0].hist(all_means, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Parameter Mean')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Parameter Means')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Std distribution
        axes[0, 1].hist(all_stds, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Parameter Std')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Parameter Uncertainties')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean vs Std scatter
        axes[1, 0].scatter(all_means, all_stds, alpha=0.3, s=5)
        axes[1, 0].set_xlabel('Parameter Mean')
        axes[1, 0].set_ylabel('Parameter Std')
        axes[1, 0].set_title('Parameter Mean vs Uncertainty')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Signal-to-noise ratio
        snr = np.abs(all_means) / (all_stds + 1e-8)
        axes[1, 1].hist(np.log10(snr + 1e-8), bins=50, alpha=0.7, 
                       color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('log10(|Mean|/Std)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Parameter Signal-to-Noise Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/parameter_distributions_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def plot_mcmc_diagnostics(self, mcmc_results: Dict, save: bool = True):
        """Plot MCMC convergence diagnostics"""
        if 'samples' not in mcmc_results or not mcmc_results['samples']:
            print("No MCMC samples found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Log probability trace
        if 'log_probs' in mcmc_results and mcmc_results['log_probs']:
            axes[0, 0].plot(mcmc_results['log_probs'], linewidth=0.5)
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Log Probability')
            axes[0, 0].set_title('MCMC Log Probability Trace')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Acceptance rate
        if 'acceptance_rate' in mcmc_results and mcmc_results['acceptance_rate']:
            axes[0, 1].plot(mcmc_results['acceptance_rate'], linewidth=1)
            axes[0, 1].axhline(0.65, color='red', linestyle='--', 
                             alpha=0.5, label='Target (0.65)')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Acceptance Rate')
            axes[0, 1].set_title('HMC Acceptance Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter traces
        samples = mcmc_results['samples']
        if len(samples) > 0:
            param_traces = torch.stack([
                s[0].detach().cpu().flatten()[:5] for s in samples
            ])
            for i in range(min(5, param_traces.shape[1])):
                axes[1, 0].plot(
                    param_traces[:, i].numpy(),
                    linewidth=0.5,
                    alpha=0.7,
                    label=f'Param {i}'
                )
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Parameter Value')
            axes[1, 0].set_title('Parameter Traces (First 5)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Posterior distribution
        if len(samples) > 0:
            param_vals = torch.cat([
                s[0].detach().cpu().flatten() for s in samples
            ]).numpy()
            axes[1, 1].hist(param_vals, bins=50, alpha=0.7, 
                          color='green', edgecolor='black')
            axes[1, 1].set_xlabel('Parameter Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Posterior Parameter Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/mcmc_diagnostics_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    def save_results_summary(self, model: BayesianKAN, 
                            history: Dict, 
                            test_metrics: Dict):
        """Save comprehensive results summary"""
        summary = {
            'timestamp': self.timestamp,
            'architecture': model.layer_sizes,
            'training_history': {k: [float(v) if not isinstance(v, list) else v 
                                    for v in vals] 
                               for k, vals in history.items()},
            'test_metrics': {k: float(v) if torch.is_tensor(v) else v 
                           for k, v in test_metrics.items()},
            'model_info': {
                'n_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(DEVICE)
            }
        }
        
        filename = f"{self.save_dir}/results_summary_{self.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results summary: {filename}")

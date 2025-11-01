"""
B-Spline Basis Functions for KAN
"""

import torch
from typing import Optional


class BSplineBasis:
    """B-spline basis functions for KAN edges"""
    
    @staticmethod
    def B(x: torch.Tensor, i: int, p: int, t: torch.Tensor) -> torch.Tensor:
        """
        Recursive B-spline basis function
        
        Args:
            x: input tensor
            i: knot index
            p: degree
            t: knot vector
            
        Returns:
            B-spline basis value
        """
        if p == 0:
            return ((t[i] <= x) & (x < t[i+1])).float()
        else:
            c1 = (x - t[i]) / (t[i+p] - t[i] + 1e-8)
            c2 = (t[i+p+1] - x) / (t[i+p+1] - t[i+1] + 1e-8)
            return c1 * BSplineBasis.B(x, i, p-1, t) + c2 * BSplineBasis.B(x, i+1, p-1, t)
    
    @staticmethod
    def evaluate_basis(x: torch.Tensor, degree: int = 3, 
                       n_basis: int = 10) -> torch.Tensor:
        """
        Evaluate B-spline basis functions
        
        Args:
            x: input tensor
            degree: B-spline degree
            n_basis: number of basis functions
            
        Returns:
            Basis function values
        """
        device = x.device
        n_knots = n_basis + degree + 1
        t = torch.linspace(0, 1, n_knots).to(device)
        
        # Normalize x to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        basis = []
        for i in range(n_basis):
            basis.append(BSplineBasis.B(x_norm, i, degree, t))
        
        return torch.stack(basis, dim=-1)
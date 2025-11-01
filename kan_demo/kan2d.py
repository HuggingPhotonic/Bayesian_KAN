import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import trange

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== B-Spline Basis ==========
class BSpline(nn.Module):
    """
    B-spline basis implementation
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(BSpline, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Initialize B-spline coefficients
        # Each input-output pair owns grid_size + spline_order control points
        self.coefficients = nn.Parameter(
            torch.randn(in_features, out_features, grid_size + spline_order) * 0.1
        )
        
        # Create a uniform knot vector
        h = 2.0 / grid_size
        self.register_buffer(
            'grid',
            torch.linspace(-1 - spline_order * h, 1 + spline_order * h, 
                          grid_size + 2 * spline_order + 1)
        )
    
    def b_splines(self, x, k=0):
        """
        Recursively compute B-spline basis functions.
        Args:
            x: input values of shape (batch_size, in_features)
            k: spline order
        """
        if k == 0:
            x_expanded = x.unsqueeze(-1)
            # Zero-order B-spline (piecewise constant)
            return ((x_expanded >= self.grid[:-1].unsqueeze(0).unsqueeze(0)) & 
                    (x_expanded < self.grid[1:].unsqueeze(0).unsqueeze(0))).float()
        else:
            # Recursive computation for higher-order splines
            B_prev = self.b_splines(x, k - 1)
            
            # Left term
            left_num = x.unsqueeze(-1) - self.grid[:-k-1].unsqueeze(0).unsqueeze(0)
            left_den = self.grid[k:-1].unsqueeze(0).unsqueeze(0) - self.grid[:-k-1].unsqueeze(0).unsqueeze(0)
            left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
            left = (left_num / left_den) * B_prev[:, :, :-1]
            
            # Right term
            right_num = self.grid[k+1:].unsqueeze(0).unsqueeze(0) - x.unsqueeze(-1)
            right_den = self.grid[k+1:].unsqueeze(0).unsqueeze(0) - self.grid[1:-k].unsqueeze(0).unsqueeze(0)
            right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
            right = (right_num / right_den) * B_prev[:, :, 1:]
            
            return left + right
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: tensor of shape (batch_size, in_features)
        Returns:
            tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Normalize input to [-1, 1]
        x_normalized = torch.tanh(x)
        
        # Compute B-spline bases (batch_size, in_features, num_bases)
        bases = self.b_splines(x_normalized, self.spline_order)
        
        # Weighted sum for every input-output pair
        # bases: (batch_size, in_features, num_bases)
        # coefficients: (in_features, out_features, num_bases)
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        for i in range(self.in_features):
            for j in range(self.out_features):
                # (batch_size, num_bases) * (num_bases,) -> (batch_size,)
                output[:, j] += torch.sum(bases[:, i, :] * self.coefficients[i, j, :], dim=-1)
        
        return output


# ========== KAN Layer ==========
class KANLayer(nn.Module):
    """
    Single KAN layer with spline-based learnable activation
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.spline = BSpline(in_features, out_features, grid_size, spline_order)
        
    def forward(self, x):
        return self.spline(x)


# ========== Full KAN Network ==========
class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network
    Constructed according to the Kolmogorov-Arnold representation theorem
    """
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        """
        layers_hidden: list defining neurons per layer, e.g. [2, 8, 8, 1]
        """
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                KANLayer(layers_hidden[i], layers_hidden[i+1], grid_size, spline_order)
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ========== Define a complex 2D test function ==========
def target_function(x, y):
    """
    Complex 2D function combining multiple nonlinear patterns
    """
    return (torch.sin(np.pi * x) * torch.cos(np.pi * y) + 
            0.3 * torch.exp(-(x**2 + y**2)) + 
            0.2 * x * y +
            0.1 * torch.sin(3 * x) * torch.sin(3 * y))


# ========== Training and visualisation ==========
def train_and_visualize():
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build training data
    n_train = 2000
    x_train = torch.rand(n_train, 1) * 4 - 2  # [-2, 2]
    y_train = torch.rand(n_train, 1) * 4 - 2  # [-2, 2]
    X_train = torch.cat([x_train, y_train], dim=1).to(device)
    z_train = target_function(x_train, y_train).to(device)
    
    # Create evaluation grid
    n_test = 50
    x_test = torch.linspace(-2, 2, n_test)
    y_test = torch.linspace(-2, 2, n_test)
    X_grid, Y_grid = torch.meshgrid(x_test, y_test, indexing='ij')
    X_test = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1).to(device)
    Z_true = target_function(X_grid.to(device), Y_grid.to(device)).cpu()
    
    # Build KAN model
    model = KAN(layers_hidden=[2, 16, 16, 1], grid_size=8, spline_order=3).to(device)
    
    # Optimiser and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Training
    print("\nStart training...")
    epochs = 1000
    losses = []
    
    progress = trange(epochs, desc="Training", leave=True)
    for epoch in progress:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z_pred = model(X_train)
        loss = criterion(z_pred, z_train)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        scheduler.step()

    print("\nOptimised parameters (name -> shape):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {tuple(param.shape)}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        Z_pred = model(X_test).cpu().reshape(n_test, n_test)
    
    # Visualise results
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: true function
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_true, 
                             cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Ground Truth Function')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot 2: KAN prediction
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_pred, 
                             cmap='viridis', alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('KAN Prediction')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot 3: error distribution
    ax3 = fig.add_subplot(133, projection='3d')
    error = torch.abs(Z_true - Z_pred)
    surf3 = ax3.plot_surface(X_grid.cpu(), Y_grid.cpu(), error, 
                             cmap='hot', alpha=0.8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Absolute Error')
    ax3.set_title('Prediction Error')
    plt.colorbar(surf3, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kan_fitting_results.png', dpi=150, bbox_inches='tight')
    print("\n3D visualisations saved!")
    
    # Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('KAN Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'kan_training_loss.png', dpi=150, bbox_inches='tight')
    print("Training loss curve saved!")
    
    # Compute metrics
    mse = torch.mean((Z_true - Z_pred)**2).item()
    mae = torch.mean(torch.abs(Z_true - Z_pred)).item()
    max_error = torch.max(torch.abs(Z_true - Z_pred)).item()
    
    print(f"\nEvaluation metrics:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Maximum Absolute Error: {max_error:.6f}")
    
    # Create contour comparisons
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    levels = 20
    
    # Ground truth contours
    contour1 = axes[0].contourf(X_grid.cpu(), Y_grid.cpu(), Z_true, 
                                 levels=levels, cmap='viridis')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Ground Truth (Contours)')
    plt.colorbar(contour1, ax=axes[0])
    
    # KAN prediction contours
    contour2 = axes[1].contourf(X_grid.cpu(), Y_grid.cpu(), Z_pred, 
                                 levels=levels, cmap='viridis')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('KAN Prediction (Contours)')
    plt.colorbar(contour2, ax=axes[1])
    
    # Error contours
    contour3 = axes[2].contourf(X_grid.cpu(), Y_grid.cpu(), error, 
                                 levels=levels, cmap='hot')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Absolute Error (Contours)')
    plt.colorbar(contour3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kan_contour_comparison.png', dpi=150, bbox_inches='tight')
    print("Contour comparison saved!")
    
    return model, losses, mse, mae, max_error


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Kolmogorov-Arnold Network (KAN) Demo")
    print("="*60)
    
    model, losses, mse, mae, max_error = train_and_visualize()
    
    print("\n" + "="*60)
    print("Training finished!")
    print("="*60)

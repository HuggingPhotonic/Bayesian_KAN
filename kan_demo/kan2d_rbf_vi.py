import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import trange

OUTPUT_DIR = Path(__file__).parent / "results_vi_rbf_2d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class RBFKernelLayer(nn.Module):
    """
    RBF basis with variational coefficients. Centers and log_sigma are trainable.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_centers: int = 16, prior_scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers

        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.log_sigma = nn.Parameter(torch.zeros(num_centers))

        self.coeff_mean = nn.Parameter(
            torch.randn(in_features, out_features, num_centers) * 0.05
        )
        self.coeff_log_var = nn.Parameter(
            torch.full((in_features, out_features, num_centers), -5.0)
        )

        self.register_buffer("prior_mean", torch.zeros_like(self.coeff_mean))
        self.register_buffer("prior_var", torch.ones_like(self.coeff_mean) * prior_scale ** 2)

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def _rbf_basis(self, x):
        # x: (batch, in_features)
        x_expanded = x.unsqueeze(1)  # (batch, 1, in_features)
        centers = self.centers.unsqueeze(0)  # (1, num_centers, in_features)
        sigma = torch.exp(self.log_sigma).unsqueeze(0)  # (1, num_centers)
        dist_sq = torch.sum((x_expanded - centers) ** 2, dim=-1)  # (batch, num_centers)
        rbf_vals = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))
        # same basis for each input feature, but keep dims aligned
        return rbf_vals.unsqueeze(1).repeat(1, self.in_features, 1)

    def forward(self, x, sample=True, n_samples=1):
        basis = self._rbf_basis(x)
        if sample:
            std = torch.exp(0.5 * self.coeff_log_var)
            eps = torch.randn(n_samples, *std.shape, device=x.device)
            coeffs = self.coeff_mean + eps * std
            outputs = []
            for s in range(n_samples):
                out = torch.einsum("bin,ion->bo", basis, coeffs[s])
                outputs.append(out)
            rbf_out = torch.stack(outputs).mean(0)
        else:
            rbf_out = torch.einsum("bin,ion->bo", basis, self.coeff_mean)
        residual = x @ self.base_weight
        return rbf_out + residual, self.kl_divergence()

    def kl_divergence(self):
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0 - torch.log(var / self.prior_var)
        )
        return kl


class BayesianRBFKAN(nn.Module):
    def __init__(self, layers_hidden, num_centers=16, prior_scale=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            RBFKernelLayer(layers_hidden[i], layers_hidden[i + 1],
                           num_centers=num_centers, prior_scale=prior_scale)
            for i in range(len(layers_hidden) - 1)
        ])

    def forward(self, x, sample=True, n_samples=1):
        total_kl = 0.0
        for layer in self.layers:
            x, kl = layer(x, sample=sample, n_samples=n_samples)
            total_kl = total_kl + kl
        return x, total_kl

    def predict_with_uncertainty(self, x, n_samples=100):
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.forward(x, sample=True, n_samples=1)
                preds.append(pred)
        stacked = torch.stack(preds)
        return stacked.mean(0), stacked.std(0)


def target_function(x, y):
    return (torch.sin(np.pi * x) * torch.cos(np.pi * y) +
            0.3 * torch.exp(-(x ** 2 + y ** 2)) +
            0.2 * x * y +
            0.1 * torch.sin(3 * x) * torch.sin(3 * y))


def train_with_variational_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_train = 5000
    x_train = torch.rand(n_train, 1) * 4 - 2
    y_train = torch.rand(n_train, 1) * 4 - 2
    X_train = torch.cat([x_train, y_train], dim=1).to(device)
    z_train = target_function(x_train, y_train).to(device)

    n_test = 100
    x_test = torch.linspace(-2, 2, n_test)
    y_test = torch.linspace(-2, 2, n_test)
    X_grid, Y_grid = torch.meshgrid(x_test, y_test, indexing='ij')
    X_test = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1).to(device)
    Z_true = target_function(X_grid.to(device), Y_grid.to(device)).cpu()

    model = BayesianRBFKAN(layers_hidden=[2, 12, 12, 1],
                           num_centers=24, prior_scale=0.5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    mse_loss = nn.MSELoss()
    kl_max = 2e-3
    kl_warmup_epochs = 400

    print("\nStart VI training (RBF basis)...")
    epochs = 2000
    losses = []

    progress = trange(epochs, desc="RBF VI Training", leave=True)
    for epoch in progress:
        model.train()
        optimizer.zero_grad()

        preds, kl = model(X_train, sample=True, n_samples=10)
        recon = mse_loss(preds, z_train)
        kl_weight = kl_max * min(1.0, (epoch + 1) / kl_warmup_epochs)
        loss = recon + kl_weight * kl / X_train.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        progress.set_postfix(recon=recon.item(),
                             kl=kl.item(),
                             beta=kl_weight,
                             loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])

    print("\nOptimised variational parameters (name -> shape):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {tuple(param.shape)}")

    model.eval()
    with torch.no_grad():
        Z_mean, Z_std = model.predict_with_uncertainty(X_test, n_samples=200)
    Z_mean = Z_mean.cpu().reshape(n_test, n_test)
    Z_std = Z_std.cpu().reshape(n_test, n_test)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_true, cmap='viridis', alpha=0.8)
    ax1.set_title('Ground Truth')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_mean, cmap='viridis', alpha=0.8)
    ax2.set_title('RBF VI Mean')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_std, cmap='hot', alpha=0.8)
    ax3.set_title('Predictive Std Dev')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rbf_vi_surfaces.png', dpi=150, bbox_inches='tight')
    print("\nSurface plots saved!")

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title('RBF VI Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'rbf_vi_loss.png', dpi=150, bbox_inches='tight')
    print("Training loss curve saved!")

    mse = torch.mean((Z_true - Z_mean) ** 2).item()
    mae = torch.mean(torch.abs(Z_true - Z_mean)).item()
    max_error = torch.max(torch.abs(Z_true - Z_mean)).item()
    print(f"\nEvaluation metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Max error: {max_error:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    levels = 20
    contour1 = axes[0].contourf(X_grid.cpu(), Y_grid.cpu(), Z_true, levels=levels, cmap='viridis')
    axes[0].set_title('Ground Truth')
    plt.colorbar(contour1, ax=axes[0])

    contour2 = axes[1].contourf(X_grid.cpu(), Y_grid.cpu(), Z_mean, levels=levels, cmap='viridis')
    axes[1].set_title('RBF Mean')
    plt.colorbar(contour2, ax=axes[1])

    contour3 = axes[2].contourf(X_grid.cpu(), Y_grid.cpu(), Z_std, levels=levels, cmap='hot')
    axes[2].set_title('Predictive Std Dev')
    plt.colorbar(contour3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rbf_vi_contours.png', dpi=150, bbox_inches='tight')
    print("Contour plots saved!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("RBF KAN 2D Variational Inference")
    print("=" * 60)

    train_with_variational_inference()

    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)

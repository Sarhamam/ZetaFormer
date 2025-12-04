"""
Visualization tools for polylipse datasets and curriculum training.

Provides 2D scatter plots with focal centers, decision boundaries,
and trajectory visualizations similar to test.py's zero-set plots.

Author: Enhanced for Noetic Eidos Project
License: MIT
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Dict, List, Tuple

from polylipse_dataset import make_polylipse_dataset, solve_focal_config


def project_to_2d(X: torch.Tensor, n_foci: int = None) -> torch.Tensor:
    """
    Project high-dimensional embedding to 2D for visualization.

    Uses PCA to find the 2 principal components.

    Args:
        X: Input tensor of shape (n_samples, seq_len, d_model) or (n_samples, d_model)
        n_foci: Number of foci (unused, for compatibility)

    Returns:
        X_2d: Tensor of shape (n_samples, 2)
    """
    # Handle sequence dimension if present
    if X.dim() == 3:
        X = X.squeeze(1)  # (n_samples, seq_len, d_model) -> (n_samples, d_model)

    # Simple PCA projection to 2D
    # Center the data
    X_centered = X - X.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Get top 2 eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Take top 2 eigenvectors (largest eigenvalues are at the end)
    top_2_eigenvectors = eigenvectors[:, -2:]

    # Project to 2D
    X_2d = X_centered @ top_2_eigenvectors

    return X_2d


def plot_polylipse_2d(
    n_foci: int,
    observed_kappa: float,
    n_samples: int = 1000,
    focal_radius: float = 2.0,
    show_focal_info: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize a single polylipse dataset in 2D.

    Shows:
    - Data points colored by dominant focus
    - Focal centers marked with stars
    - Circular regions around each focus
    - Focal configuration info

    Args:
        n_foci: Number of foci
        observed_kappa: The κζ used to configure this dataset
        n_samples: Number of samples to generate
        focal_radius: Radius for focal distribution
        show_focal_info: Display focal configuration text
        title: Plot title
        save_path: Optional path to save figure
    """
    # Generate dataset with focal info
    X, y, mask, info = make_polylipse_dataset(
        n_foci=n_foci,
        observed_kappa=observed_kappa,
        n_samples=n_samples,
        focal_radius=focal_radius,
        return_focal_info=True
    )

    # Extract 2D positions (reverse the embedding)
    # For visualization, regenerate the 2D points directly
    angles = torch.tensor(info['angles'])
    weights = torch.tensor(info['weights'])
    centers = torch.tensor(info['centers'])

    # Regenerate 2D data
    torch.manual_seed(42)
    focal_indices = torch.multinomial(weights, n_samples, replacement=True)
    theta = 2 * np.pi * torch.rand(n_samples)
    r = 0.5 + 0.3 * torch.randn(n_samples)
    offsets = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    X_2d = centers[focal_indices] + offsets + 0.05 * torch.randn(n_samples, 2)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color map for different foci
    colors = plt.cm.tab10(np.linspace(0, 1, n_foci))

    # Plot data points by dominant focus
    for i in range(n_foci):
        mask_i = focal_indices == i
        ax.scatter(
            X_2d[mask_i, 0].numpy(),
            X_2d[mask_i, 1].numpy(),
            c=[colors[i]],
            s=20,
            alpha=0.6,
            label=f'Focus {i+1}',
            edgecolors='none'
        )

    # Plot focal centers
    for i in range(n_foci):
        ax.scatter(
            centers[i, 0].numpy(),
            centers[i, 1].numpy(),
            marker='*',
            s=500,
            c=[colors[i]],
            edgecolors='black',
            linewidths=2,
            zorder=10
        )

        # Draw circle around each focus
        circle = Circle(
            (centers[i, 0].item(), centers[i, 1].item()),
            radius=0.8,
            fill=False,
            edgecolor=colors[i],
            linewidth=2,
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(circle)

    # Configure plot
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    if title is None:
        title = f"{n_foci}-Focal Dataset (κζ = {observed_kappa:.3f})"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)

    # Add focal configuration info
    if show_focal_info:
        info_text = f"Focal Configuration:\n"
        info_text += f"  n = {n_foci} foci\n"
        info_text += f"  Target κζ = {observed_kappa:.4f}\n"
        info_text += f"  Actual κζ = {info['kappa_actual']:.4f}\n"
        info_text += f"  M_τ = {info['M_tau']:.4f}\n"
        info_text += f"  M_σ = {info['M_sigma']:.4f}\n\n"
        info_text += f"Angles (°): {[f'{a:.0f}' for a in info['angles_deg']]}\n"
        info_text += f"Weights: {[f'{w:.3f}' for w in info['weights']]}"

        ax.text(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace'
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_curriculum_levels(
    curriculum_metrics,
    save_dir: Optional[str] = None
):
    """
    Create a grid showing all curriculum levels side-by-side.

    Args:
        curriculum_metrics: CurriculumMetrics object
        save_dir: Optional directory to save figures
    """
    from pathlib import Path

    n_levels = len(curriculum_metrics.levels)
    ncols = min(3, n_levels)
    nrows = (n_levels + ncols - 1) // ncols

    fig = plt.figure(figsize=(6*ncols, 6*nrows))

    for idx, level in enumerate(curriculum_metrics.levels, 1):
        ax = fig.add_subplot(nrows, ncols, idx)

        n_foci = level['n_foci']
        target_kappa = level['target_kappa'] if level['target_kappa'] is not None else 1.0
        stabilized_kappa = level['stabilized_kappa']

        # Generate and plot
        if n_foci == 1:
            # Simple circle for n=1
            torch.manual_seed(42)
            theta = 2 * np.pi * torch.rand(500)
            r = 0.5 + 0.3 * torch.randn(500)
            X_2d = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            X_2d += 0.05 * torch.randn_like(X_2d)

            ax.scatter(X_2d[:, 0], X_2d[:, 1], c='steelblue', s=10, alpha=0.6)
            ax.scatter([0], [0], marker='*', s=500, c='gold', edgecolors='black', linewidths=2, zorder=10)

        else:
            # Multi-focal
            angles, weights = solve_focal_config(n_foci, target_kappa)
            centers = 2.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

            torch.manual_seed(42)
            focal_indices = torch.multinomial(weights, 500, replacement=True)
            theta = 2 * np.pi * torch.rand(500)
            r = 0.5 + 0.3 * torch.randn(500)
            offsets = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            X_2d = centers[focal_indices] + offsets + 0.05 * torch.randn(500, 2)

            colors = plt.cm.tab10(np.linspace(0, 1, n_foci))

            for i in range(n_foci):
                mask_i = focal_indices == i
                ax.scatter(
                    X_2d[mask_i, 0].numpy(),
                    X_2d[mask_i, 1].numpy(),
                    c=[colors[i]],
                    s=10,
                    alpha=0.6
                )

            for i in range(n_foci):
                ax.scatter(
                    centers[i, 0].numpy(),
                    centers[i, 1].numpy(),
                    marker='*',
                    s=300,
                    c=[colors[i]],
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=10
                )

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        title = f"Level {idx}: n={n_foci} foci\n"
        if level['target_kappa'] is not None:
            title += f"Config: κζ={target_kappa:.3f}\n"
        title += f"Stable: κζ={stabilized_kappa:.3f}"

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "curriculum_levels_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_kappa_evolution_with_transitions(
    curriculum_metrics,
    save_path: Optional[str] = None
):
    """
    Plot κζ evolution over time with vertical lines marking level transitions.

    Shows how κζ converges at each level and then transitions to next dataset.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Collect all κζ values with epoch numbers
    all_kappa = []
    all_epochs = []
    transition_epochs = []
    transition_labels = []

    epoch_offset = 0
    for i, level in enumerate(curriculum_metrics.levels):
        # Get kappa values for this level
        level_kappa = [k for epoch_list in level['metrics'].epoch_kappa for k in epoch_list]

        # Add to global list
        epochs = np.arange(len(level_kappa)) + epoch_offset
        all_kappa.extend(level_kappa)
        all_epochs.extend(epochs)

        # Mark transition
        if i < len(curriculum_metrics.levels) - 1:
            transition_epochs.append(epoch_offset + len(level_kappa))
            transition_labels.append(f"→ {level['n_foci']+1} foci")

        epoch_offset += len(level_kappa)

    # Plot κζ evolution
    ax.plot(all_epochs, all_kappa, linewidth=1.5, alpha=0.7, color='steelblue', label='κζ (calibrated)')

    # Add moving average
    window = 20
    if len(all_kappa) >= window:
        kappa_smooth = np.convolve(all_kappa, np.ones(window)/window, mode='valid')
        epochs_smooth = all_epochs[window-1:]
        ax.plot(epochs_smooth, kappa_smooth, linewidth=2.5, color='darkblue', label=f'κζ (MA-{window})')

    # Mark transitions
    for trans_epoch, trans_label in zip(transition_epochs, transition_labels):
        ax.axvline(trans_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(
            trans_epoch, ax.get_ylim()[1] * 0.95,
            trans_label,
            rotation=90,
            verticalalignment='top',
            fontsize=10,
            color='red'
        )

    # Mark stable κζ for each level
    epoch_offset = 0
    for level in curriculum_metrics.levels:
        level_kappa = [k for epoch_list in level['metrics'].epoch_kappa for k in epoch_list]
        mid_epoch = epoch_offset + len(level_kappa) // 2

        ax.annotate(
            f"n={level['n_foci']}\nκζ={level['stabilized_kappa']:.3f}",
            xy=(mid_epoch, level['stabilized_kappa']),
            xytext=(0, 20),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
        )

        epoch_offset += len(level_kappa)

    ax.set_xlabel('Training Epoch (across all levels)', fontsize=12)
    ax.set_ylabel('κζ (Kappa-Zeta)', fontsize=12)
    ax.set_title('κζ Evolution Through Adaptive Curriculum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def visualize_focal_config_space(
    n_foci: int = 3,
    kappa_range: Tuple[float, float] = (0.5, 3.0),
    n_points: int = 10
):
    """
    Visualize how focal configurations change across κζ values.

    Shows a grid of polylipse datasets with different κζ values,
    demonstrating the mathematical relationship between κζ and geometry.
    """
    kappa_values = np.linspace(kappa_range[0], kappa_range[1], n_points)

    ncols = min(5, n_points)
    nrows = (n_points + ncols - 1) // ncols

    fig = plt.figure(figsize=(3*ncols, 3*nrows))

    for idx, kappa in enumerate(kappa_values, 1):
        ax = fig.add_subplot(nrows, ncols, idx)

        # Generate focal config
        angles, weights = solve_focal_config(n_foci, kappa)
        centers = 2.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        # Sample some points
        torch.manual_seed(42)
        focal_indices = torch.multinomial(weights, 200, replacement=True)
        theta = 2 * np.pi * torch.rand(200)
        r = 0.5 + 0.3 * torch.randn(200)
        offsets = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        X_2d = centers[focal_indices] + offsets + 0.05 * torch.randn(200, 2)

        # Plot
        colors = plt.cm.tab10(np.linspace(0, 1, n_foci))
        for i in range(n_foci):
            mask_i = focal_indices == i
            ax.scatter(X_2d[mask_i, 0].numpy(), X_2d[mask_i, 1].numpy(),
                      c=[colors[i]], s=5, alpha=0.5)
            ax.scatter(centers[i, 0].numpy(), centers[i, 1].numpy(),
                      marker='*', s=150, c=[colors[i]], edgecolors='black', linewidths=1, zorder=10)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title(f"κζ = {kappa:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Focal Configuration Space (n={n_foci} foci)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================================
# CGD (Conjugate Gradient Descent) Solver Integration
# ============================================================================

def row_normalize(mat, eps=1e-12):
    """Row-normalize a matrix."""
    return mat / (mat.sum(dim=1, keepdim=True) + eps)


def pairwise_sq_dists(X, Y=None):
    """Compute pairwise squared distances. X: (n,d), Y: (m,d) -> (n,m)"""
    if Y is None:
        Y = X
    XX = (X * X).sum(dim=1, keepdim=True)  # (n,1)
    YY = (Y * Y).sum(dim=1, keepdim=True).T  # (1,m)
    return torch.clamp(XX + YY - 2 * X @ Y.T, min=0.0)


def gaussian_kernel_from_points(X, Y, sigma):
    """Gaussian/heat kernel ~ exp(-||x-y||^2/(2σ^2))"""
    D2 = pairwise_sq_dists(X, Y)
    K = torch.exp(-D2 / (2.0 * sigma ** 2))
    return row_normalize(K)


def poisson_kernel_from_points(X, Y, t):
    """Poisson/transport kernel ~ 1/(||x-y||^2 + t^2)"""
    D2 = pairwise_sq_dists(X, Y)
    K = 1.0 / (D2 + t ** 2)
    return row_normalize(K)


def cg_solve(A_apply, AT_apply, y, eta=1e-3, tol=1e-6, max_iter=500):
    """
    Conjugate Gradient solve for (A^T A + eta I) f = A^T y.

    Args:
        A_apply: lambda v -> A v
        AT_apply: lambda v -> A^T v
        y: target vector
        eta: regularization strength
        tol: convergence tolerance
        max_iter: maximum iterations

    Returns:
        f: solution coefficients
    """
    def normal_op(v):
        return AT_apply(A_apply(v)) + eta * v

    f = torch.zeros_like(y)
    r = AT_apply(y) - normal_op(f)
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())

    for it in range(max_iter):
        Ap = normal_op(p)
        denom = torch.dot(p.flatten(), Ap.flatten()) + 1e-12
        alpha = rs_old / denom
        f = f + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r.flatten(), r.flatten())
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return f


def build_dual_kernel_operator(X_train, X_basis=None, sigma=0.5, t=0.5, w=0.5):
    """
    Build dual-kernel operator S = w*Gaussian + (1-w)*Poisson.

    Args:
        X_train: Training points (n, d)
        X_basis: Basis points (m, d). If None, uses X_train.
        sigma: Gaussian bandwidth (τ-kernel)
        t: Poisson scale (σ-kernel)
        w: Mellin mix weight (default 0.5 = balanced τ-σ)

    Returns:
        S: (n, m) operator matrix
        X_basis: basis points used
    """
    if X_basis is None:
        X_basis = X_train

    G = gaussian_kernel_from_points(X_train, X_basis, sigma)
    P = poisson_kernel_from_points(X_train, X_basis, t)
    S = w * G + (1.0 - w) * P

    return S, X_basis


def fit_cgd_polylipse(X_2d, y_labels, sigma=0.5, t=0.5, w=0.5, eta=1e-3):
    """
    Fit CGD solver to polylipse data.

    Args:
        X_2d: (n, 2) 2D points
        y_labels: (n,) class labels (0, 1, 2, ..., n_foci-1)
        sigma: Gaussian bandwidth
        t: Poisson scale
        w: Mellin mix weight
        eta: Regularization strength

    Returns:
        f: (n,) solution coefficients
        S: (n, n) operator matrix
        X_basis: basis points
    """
    # Convert to float64 for numerical stability
    X_2d = X_2d.to(torch.float64)

    # Map labels to signed targets for multi-class
    # For visualization, we'll use binary encoding (in-class vs out)
    # Here, just use the first class as positive for simplicity
    y = (y_labels == 0).to(torch.float64) * 2.0 - 1.0  # {-1, +1}

    # Build dual-kernel operator
    S, X_basis = build_dual_kernel_operator(X_2d, None, sigma, t, w)
    S = S.to(torch.float64)

    # Define operators
    A_apply = lambda v: S @ v
    AT_apply = lambda v: S.T @ v

    # Solve with CG
    f = cg_solve(A_apply, AT_apply, y, eta=eta, tol=1e-8, max_iter=2000)

    return f, S, X_basis


def predict_cgd_scores(X_query, X_basis, f, sigma=0.5, t=0.5, w=0.5):
    """
    Evaluate CGD solution on query points.

    Args:
        X_query: (m, 2) query points
        X_basis: (n, 2) basis points
        f: (n,) solution coefficients
        sigma: Gaussian bandwidth
        t: Poisson scale
        w: Mellin mix weight

    Returns:
        scores: (m,) signed distance from decision boundary
    """
    X_query = X_query.to(torch.float64)
    X_basis = X_basis.to(torch.float64)

    Gq = gaussian_kernel_from_points(X_query, X_basis, sigma)
    Pq = poisson_kernel_from_points(X_query, X_basis, t)
    Sq = (w * Gq + (1.0 - w) * Pq).to(torch.float64)

    g = Sq @ f
    return g.to(torch.float32)


def plot_cgd_polylipse(
    n_foci: int,
    observed_kappa: float,
    n_samples: int = 1000,
    sigma: float = 0.5,
    t: float = 0.5,
    w: float = 0.5,
    eta: float = 1e-2,
    grid_resolution: int = 220,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize CGD solution on polylipse dataset.

    Shows:
    - Heatmap of CGD-learned function over space
    - Zero-level set (decision boundary) as black contour
    - Training points colored by class
    - Focal centers marked with stars

    This demonstrates how the dual-kernel (Gaussian-Poisson) attention
    learns to separate the n-focal geometry.

    Args:
        n_foci: Number of foci
        observed_kappa: κζ ratio for dataset configuration
        n_samples: Number of training samples
        sigma: Gaussian bandwidth (τ-kernel)
        t: Poisson scale (σ-kernel)
        w: Mellin mix (0.5 = balanced τ-σ)
        eta: Regularization strength
        grid_resolution: Grid size for heatmap
        title: Plot title
        save_path: Optional path to save

    Returns:
        fig: matplotlib figure
        f: CGD solution coefficients
        info: focal configuration info
    """
    # Generate polylipse dataset
    X, y, mask, info = make_polylipse_dataset(
        n_foci=n_foci,
        observed_kappa=observed_kappa,
        n_samples=n_samples,
        d_model=32,
        return_focal_info=True
    )

    # Extract 2D points (reverse the embedding)
    # The embedding was: X_embed = X_2d @ W_embed where W_embed is (2, d_model)
    # To invert: X_2d = X_embed @ pinv(W_embed)
    torch.manual_seed(42)
    W_embed = torch.randn(2, 32) / np.sqrt(2)  # (2, 32)
    X_2d = X.squeeze(1) @ torch.linalg.pinv(W_embed)  # (n, 32) @ (32, 2) = (n, 2)
    y_labels = y.squeeze(1)

    # Fit CGD
    f, S, X_basis = fit_cgd_polylipse(X_2d, y_labels, sigma=sigma, t=t, w=w, eta=eta)

    # Create grid for visualization
    gx, gy = torch.meshgrid(
        torch.linspace(-3.5, 3.5, grid_resolution),
        torch.linspace(-3.5, 3.5, grid_resolution),
        indexing="xy"
    )
    Gpts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    scores = predict_cgd_scores(Gpts, X_basis, f, sigma, t, w).reshape(gx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Heatmap
    contourf = ax.contourf(gx.numpy(), gy.numpy(), scores.numpy(),
                           levels=50, cmap="RdBu", alpha=0.7)
    plt.colorbar(contourf, ax=ax, label="CGD score (S f)")

    # Zero-level set (decision boundary)
    ax.contour(gx.numpy(), gy.numpy(), scores.numpy(),
               levels=[0.0], colors='black', linewidths=2.5, linestyles='-')

    # Training points
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_foci, 10)))  # Cycle colors for high n

    # For high focal counts, don't add individual labels
    show_legend = n_foci <= 5

    for i in range(n_foci):
        mask_i = (y_labels == i)
        color_idx = i % 10  # Cycle through 10 colors
        ax.scatter(X_2d[mask_i, 0].numpy(), X_2d[mask_i, 1].numpy(),
                   c=[colors[color_idx]], s=8, alpha=0.5,
                   label=f'Focus {i}' if show_legend else None,
                   edgecolors='none')

    # Focal centers - scale size inversely with n_foci
    centers = info['centers']
    star_size = max(50, 400 / (n_foci ** 0.5))  # Adaptive size
    for i in range(n_foci):
        color_idx = i % 10
        ax.scatter(centers[i, 0], centers[i, 1],
                   marker='*', s=star_size, c=[colors[color_idx]],
                   edgecolors='black', linewidths=1, zorder=10, alpha=0.8)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')

    # Only show legend for low focal counts
    if show_legend:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

    ax.grid(True, alpha=0.2)

    if title is None:
        title = f"CGD Solution: {n_foci}-Focal (κζ={observed_kappa:.3f})\nσ={sigma}, t={t}, w={w}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved CGD visualization: {save_path}")

    return fig, f, info


def plot_cgd_curriculum(
    curriculum_metrics,
    sigma: float = 0.5,
    t: float = 0.5,
    w: float = 0.5,
    eta: float = 1e-2,
    n_samples: int = 800,
    save_dir: Optional[str] = None
):
    """
    Visualize CGD solutions for all curriculum levels.

    Creates a grid showing how the dual-kernel attention learns
    progressively complex geometries (1-focal → 2-focal → ... → n-focal).

    Args:
        curriculum_metrics: CurriculumMetrics object from training
        sigma: Gaussian bandwidth
        t: Poisson scale
        w: Mellin mix
        eta: Regularization
        n_samples: Samples per level
        save_dir: Directory to save individual plots

    Returns:
        figures: List of matplotlib figures (one per level)
    """
    from pathlib import Path

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_levels = len(curriculum_metrics.levels)
    figures = []

    print(f"\nGenerating CGD visualizations for {n_levels} curriculum levels...")

    for i, level in enumerate(curriculum_metrics.levels):
        n_foci = level['n_foci']
        target_kappa = level['target_kappa'] if level['target_kappa'] is not None else 1.0
        stabilized_kappa = level['stabilized_kappa']

        print(f"  Level {i+1}/{n_levels}: n={n_foci} foci, κζ={stabilized_kappa:.3f}")

        title = (f"Level {i+1}: {n_foci}-Focal Geometry\n"
                 f"κζ_target={target_kappa:.3f} → κζ_stable={stabilized_kappa:.3f}")

        save_path = Path(save_dir) / f"cgd_level_{i+1}_n{n_foci}.png" if save_dir else None

        fig, f, info = plot_cgd_polylipse(
            n_foci=n_foci,
            observed_kappa=stabilized_kappa,
            n_samples=n_samples,
            sigma=sigma,
            t=t,
            w=w,
            eta=eta,
            title=title,
            save_path=save_path
        )

        figures.append(fig)
        plt.close(fig)

    print(f"✓ Generated {len(figures)} CGD visualizations")

    return figures


if __name__ == "__main__":
    print("Testing polylipse visualization...")

    # Test 1: Single polylipse plot
    print("\n[Test 1] Single polylipse visualization")
    fig1 = plot_polylipse_2d(
        n_foci=3,
        observed_kappa=1.5,
        n_samples=1000,
        title="Trifocal Dataset Example",
        save_path="./test_polylipse_single.png"
    )
    plt.show()

    # Test 2: Focal config space
    print("\n[Test 2] Focal configuration space")
    fig2 = visualize_focal_config_space(n_foci=3, kappa_range=(0.5, 2.5), n_points=9)
    plt.savefig("./test_focal_config_space.png", dpi=150, bbox_inches='tight')
    print("Saved: ./test_focal_config_space.png")
    plt.show()

    print("\nVisualization tests complete!")

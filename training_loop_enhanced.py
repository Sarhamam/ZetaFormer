"""
Enhanced training loop with ζ-normalization monitoring and logging.

This training loop extends the base implementation to:
1. Track κζ (kappa_zeta) evolution per epoch
2. Log Poisson parameter (β, t) dynamics
3. Monitor ζ-convergence rates
4. Provide rich metrics for analysis and visualization
5. Support both standard and enhanced ZetaBlock

Features:
- Backward compatible with original training loop
- Optional κζ tracking via return_kappa flag
- Per-epoch and per-batch statistics
- Convergence diagnostics
- Parameter drift monitoring

Author: Enhanced by Claude for Noetic Eidos Project
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from zeta_block_enhanced import ZetaBlockEnhanced
from zeta_losses import ZetaLosses


class ZetaTrainingMetrics:
    """
    Container for tracking ζ-normalization metrics during training.

    Tracks:
    - κζ evolution per epoch and batch (both raw and calibrated)
    - Offset dynamics (m* evolution)
    - Poisson parameter (β, t) dynamics
    - Loss components (task, zero-set)
    - Convergence diagnostics
    """

    def __init__(self):
        # Calibrated κζ (control signal)
        self.epoch_kappa: List[List[float]] = []      # Per-epoch calibrated κζ values
        # Raw κζ (science - shows global scale)
        self.epoch_kappa_raw: List[List[float]] = []  # Per-epoch raw κζ values
        # Offset dynamics
        self.epoch_offset: List[List[float]] = []     # Per-epoch offset values

        # Parameter snapshots
        self.epoch_beta: List[np.ndarray] = []        # Per-epoch β snapshots
        self.epoch_t: List[np.ndarray] = []           # Per-epoch t snapshots

        # Loss tracking
        self.epoch_loss: List[float] = []             # Total loss per epoch
        self.epoch_task_loss: List[float] = []        # Task loss per epoch
        self.epoch_zero_loss: List[float] = []        # Zero-set loss per epoch

    def add_epoch(
        self,
        kappa_vals: List[float],
        kappa_raw_vals: List[float],
        offset_vals: List[float],
        beta_vals: np.ndarray,
        t_vals: np.ndarray,
        total_loss: float,
        task_loss: float,
        zero_loss: float,
    ) -> None:
        """Add metrics for a completed epoch."""
        self.epoch_kappa.append(kappa_vals)
        self.epoch_kappa_raw.append(kappa_raw_vals)
        self.epoch_offset.append(offset_vals)
        self.epoch_beta.append(beta_vals.copy())
        self.epoch_t.append(t_vals.copy())
        self.epoch_loss.append(total_loss)
        self.epoch_task_loss.append(task_loss)
        self.epoch_zero_loss.append(zero_loss)

    def get_kappa_convergence(self, window: int = 5) -> float:
        """
        Compute κζ convergence rate over recent epochs.

        Returns: Mean absolute gradient of κζ over last 'window' epochs.
        Lower values indicate convergence to critical line.
        """
        if len(self.epoch_kappa) < 2:
            return float('inf')

        # Flatten recent κζ values
        recent = self.epoch_kappa[-window:] if len(self.epoch_kappa) >= window else self.epoch_kappa
        flat = [k for epoch in recent for k in epoch]

        if len(flat) < 2:
            return float('inf')

        # Compute gradient magnitude
        grad = np.abs(np.gradient(flat))
        return float(np.mean(grad))

    def get_parameter_drift(self) -> Dict[str, float]:
        """
        Compute drift in Poisson parameters over training.

        Returns: Dict with 'beta_drift' and 't_drift' (L2 norm of change)
        """
        if len(self.epoch_beta) < 2:
            return {"beta_drift": 0.0, "t_drift": 0.0}

        beta_start = self.epoch_beta[0]
        beta_end = self.epoch_beta[-1]
        t_start = self.epoch_t[0]
        t_end = self.epoch_t[-1]

        beta_drift = float(np.linalg.norm(beta_end - beta_start))
        t_drift = float(np.linalg.norm(t_end - t_start))

        return {"beta_drift": beta_drift, "t_drift": t_drift}

    def summary(self) -> Dict[str, float]:
        """Generate summary statistics for the entire training run."""
        if len(self.epoch_loss) == 0:
            return {}

        kappa_flat = [k for epoch in self.epoch_kappa for k in epoch]

        return {
            "final_loss": self.epoch_loss[-1],
            "loss_reduction": self.epoch_loss[0] - self.epoch_loss[-1] if len(self.epoch_loss) > 1 else 0.0,
            "final_kappa_mean": float(np.mean(kappa_flat[-100:])) if len(kappa_flat) >= 100 else (float(np.mean(kappa_flat)) if kappa_flat else 0.0),
            "kappa_convergence": self.get_kappa_convergence(),
            **self.get_parameter_drift(),
        }


def train_zeta_block(
    make_dataset_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    d_model: int = 32,
    n_heads: int = 4,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    enable_zeta_norm: bool = True,
    kappa_strength: float = 0.05,
    lambda_zero: float = 0.1,
    eta_zero: float = 0.5,
    log_interval: int = 1,
    verbose: bool = True,
    model: Optional[ZetaBlockEnhanced] = None,
    classifier: Optional[nn.Linear] = None,
    optimizer: Optional[optim.Optimizer] = None,
    existing_metrics: Optional[ZetaTrainingMetrics] = None,
) -> Tuple[ZetaBlockEnhanced, nn.Linear, optim.Optimizer, ZetaTrainingMetrics]:
    """
    Train a ZetaBlock with optional ζ-normalization feedback.

    Args:
        make_dataset_fn: Function returning (X, y, mask) tensors
        d_model: Model dimension
        n_heads: Number of attention heads
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: torch device ('cuda' or 'cpu')
        enable_zeta_norm: Enable ζ-normalization feedback loop
        kappa_strength: Feedback strength for κζ modulation (0.01-0.1 typical)
        lambda_zero: Weight for zero-set constraint loss
        eta_zero: Margin for zero-set loss
        log_interval: Epochs between detailed logging
        verbose: Print training progress

    Returns:
        model: Trained ZetaBlockEnhanced
        classifier: Trained classification head
        metrics: ZetaTrainingMetrics with complete training history

    Example:
       def make_data():
        ...     X = torch.randn(1000, 50, 32)
        ...     y = torch.randint(0, 2, (1000, 50))
        ...     mask = torch.ones(1000, 50, dtype=torch.bool)
        ...     return X, y, mask
       model, clf, metrics = train_zeta_block(make_data, n_epochs=10)
       print(metrics.summary())
    """
    # ----- Data -----
    X, y, mask = make_dataset_fn()
    dataset = TensorDataset(X, y, mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----- Model & losses -----
    if model is None:
        # Create new model
        model = ZetaBlockEnhanced(
            d_model=d_model,
            n_heads=n_heads,
            enable_zeta_norm=enable_zeta_norm,
            kappa_strength=kappa_strength,
        ).to(device)
    else:
        # Continue existing model
        model = model.to(device)

    losses = ZetaLosses(lambda_subm=lambda_zero, eta_zero=eta_zero)

    n_classes = y.max().item() + 1

    if classifier is None:
        classifier = nn.Linear(d_model, n_classes).to(device)
    else:
        classifier = classifier.to(device)

    if optimizer is None:
        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=lr
        )

    # ----- Metrics tracking -----
    if existing_metrics is None:
        metrics = ZetaTrainingMetrics()
    else:
        metrics = existing_metrics  # Continue accumulating

    # ----- Training Loop -----
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_task, total_zero = 0.0, 0.0, 0.0
        epoch_kappa_vals: List[float] = []       # Calibrated κζ (control)
        epoch_kappa_raw_vals: List[float] = []   # Raw κζ (science)
        epoch_offset_vals: List[float] = []      # Offset dynamics
        n_batches = 0

        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            # Forward pass with κζ tracking
            if enable_zeta_norm:
                out, f, tau, sigma, kappa = model(
                    xb,
                    return_components=True,
                    return_kappa=True
                )
                epoch_kappa_vals.append(kappa)
                # Extract raw κζ and offset from model logs
                if hasattr(model, 'kappa_raw_log') and model.kappa_raw_log:
                    epoch_kappa_raw_vals.append(model.kappa_raw_log[-1])
                    epoch_offset_vals.append(model.offset_log[-1] if model.offset_log else 0.0)
                else:
                    epoch_kappa_raw_vals.append(kappa)
                    epoch_offset_vals.append(0.0)
            else:
                out, f, tau, sigma = model(xb, return_components=True)
                epoch_kappa_vals.append(0.0)
                epoch_kappa_raw_vals.append(0.0)
                epoch_offset_vals.append(0.0)

            logits = classifier(out)

            # Loss computation
            L, parts = losses(logits, yb, f, tau, sigma, mb)

            # Optimization step
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += L.item()
            total_task += parts["task"]
            total_zero += parts["zero"]
            n_batches += 1

        # Per-epoch statistics
        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        avg_zero = total_zero / n_batches

        # Snapshot Poisson parameters
        with torch.no_grad():
            beta_snapshot = model.poisson_beta.cpu().numpy()
            t_snapshot = model.poisson_t.cpu().numpy()

        # Record metrics (two-stage κζ tracking)
        metrics.add_epoch(
            kappa_vals=epoch_kappa_vals,           # Calibrated (control)
            kappa_raw_vals=epoch_kappa_raw_vals,   # Raw (science)
            offset_vals=epoch_offset_vals,         # Offset dynamics
            beta_vals=beta_snapshot,
            t_vals=t_snapshot,
            total_loss=avg_loss,
            task_loss=avg_task,
            zero_loss=avg_zero,
        )

        # Logging
        if verbose and (epoch + 1) % log_interval == 0:
            # Calibrated κζ (control signal)
            kappa_mean = np.mean(epoch_kappa_vals) if epoch_kappa_vals else 0.0
            kappa_std = np.std(epoch_kappa_vals) if epoch_kappa_vals else 0.0

            # Raw κζ and offset (science)
            kappa_raw_mean = np.mean(epoch_kappa_raw_vals) if epoch_kappa_raw_vals else 0.0
            offset_final = epoch_offset_vals[-1] if epoch_offset_vals else 0.0

            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {avg_loss:.3f} (task: {avg_task:.3f}, zero: {avg_zero:.3f}) | "
                  f"κζ_cal: {kappa_mean:.4f}±{kappa_std:.4f} | "
                  f"κζ_raw: {kappa_raw_mean:.4f} | "
                  f"m*: {offset_final:.4f} | "
                  f"β: [{beta_snapshot.min():.2f}, {beta_snapshot.max():.2f}] | "
                  f"t: [{t_snapshot.min():.2f}, {t_snapshot.max():.2f}]")

    # Final summary
    if verbose:
        print("\n" + "="*80)
        print("Training Complete - Summary:")
        print("="*80)
        summary = metrics.summary()
        for key, val in summary.items():
            print(f"  {key:20s}: {val:.4f}")
        print("="*80)

    return model, classifier, optimizer, metrics


def train_zeta_block_simple(
    make_dataset_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    d_model: int = 32,
    n_heads: int = 4,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ZetaBlockEnhanced, nn.Linear]:
    """
    Simplified training interface (backward compatible with original).

    Returns only model and classifier, without metrics tracking.
    """
    model, classifier, _ = train_zeta_block(
        make_dataset_fn=make_dataset_fn,
        d_model=d_model,
        n_heads=n_heads,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=False,
    )
    return model, classifier


def save_zeta_model(
    model: ZetaBlockEnhanced,
    classifier: nn.Linear,
    metrics: ZetaTrainingMetrics,
    save_path: str,
    hyperparameters: Optional[Dict] = None,
) -> None:
    """
    Save trained ζ-normalized model with full checkpoint.

    Saves:
    - Model state dict (ZetaBlockEnhanced parameters)
    - Classifier state dict
    - Training metrics (κζ evolution, losses, parameters)
    - Model hyperparameters and configuration
    - ζ-normalization state (offset, EMA state)

    Args:
        model: Trained ZetaBlockEnhanced
        classifier: Trained classification head
        metrics: ZetaTrainingMetrics from training
        save_path: Path to save checkpoint (e.g., './checkpoints/zeta_model.pt')
        hyperparameters: Optional dict of training hyperparameters

    Example:
        model, clf, metrics = train_zeta_block(...)
        save_zeta_model(
            model, clf, metrics,
            './checkpoints/my_model.pt',
            hyperparameters={
                'd_model': 32,
                'n_heads': 4,
                'kappa_strength': 0.05,
                'n_epochs': 20,
            }
        )
    """
    import os
    import pickle

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        # Model architecture
        'model_config': {
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'zeta_s': model.zeta_translator.s if hasattr(model, 'zeta_translator') else 0.5,
            'dynamics_alpha': model.zeta_translator.dynamics_alpha if hasattr(model, 'zeta_translator') else 0.01,
            'kappa_strength': model.kappa_strength,
            'enable_zeta_norm': model.enable_zeta_norm,
            'baseline_window': model.baseline_window,
        },

        # Model weights
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),

        # ζ-normalization state
        'zeta_state': {
            'offset': model.zeta_translator.offset if hasattr(model, 'zeta_translator') else 0.0,
            'kappa_smooth': model.kappa_smooth if hasattr(model, 'kappa_smooth') else None,
            'kappa_log': model.kappa_log.copy() if hasattr(model, 'kappa_log') else [],
            'kappa_raw_log': model.kappa_raw_log.copy() if hasattr(model, 'kappa_raw_log') else [],
            'offset_log': model.offset_log.copy() if hasattr(model, 'offset_log') else [],
            'beta_mean_log': model.beta_mean_log.copy() if hasattr(model, 'beta_mean_log') else [],
            't_mean_log': model.t_mean_log.copy() if hasattr(model, 't_mean_log') else [],
        },

        # Training metrics
        'metrics': {
            'epoch_kappa': metrics.epoch_kappa,
            'epoch_kappa_raw': metrics.epoch_kappa_raw,
            'epoch_offset': metrics.epoch_offset,
            'epoch_beta': [beta.tolist() for beta in metrics.epoch_beta],
            'epoch_t': [t.tolist() for t in metrics.epoch_t],
            'epoch_loss': metrics.epoch_loss,
            'epoch_task_loss': metrics.epoch_task_loss,
            'epoch_zero_loss': metrics.epoch_zero_loss,
            'summary': metrics.summary(),
        },

        # Hyperparameters
        'hyperparameters': hyperparameters or {},

        # Classifier info
        'n_classes': classifier.out_features,
    }

    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"  - Final κζ_cal: {metrics.epoch_kappa[-1][-1] if metrics.epoch_kappa else 0.0:.4f}")
    print(f"  - Final offset m*: {metrics.epoch_offset[-1][-1] if metrics.epoch_offset else 0.0:.4f}")


def load_zeta_model(
    load_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ZetaBlockEnhanced, nn.Linear, Dict, Dict]:
    """
    Load trained ζ-normalized model from checkpoint.

    Args:
        load_path: Path to saved checkpoint
        device: Device to load model onto

    Returns:
        model: Loaded ZetaBlockEnhanced
        classifier: Loaded classification head
        metrics_dict: Dictionary containing training metrics
        hyperparameters: Dictionary of training hyperparameters

    Example:
        model, clf, metrics_dict, hparams = load_zeta_model('./checkpoints/my_model.pt')
        print(f"Loaded model with d_model={hparams['d_model']}")
        print(f"Final loss: {metrics_dict['summary']['final_loss']:.4f}")
    """
    checkpoint = torch.load(load_path, map_location=device)

    # Reconstruct model
    model_config = checkpoint['model_config']
    model = ZetaBlockEnhanced(
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        zeta_s=model_config.get('zeta_s', 0.5),
        kappa_strength=model_config.get('kappa_strength', 0.05),
        enable_zeta_norm=model_config.get('enable_zeta_norm', True),
        baseline_window=model_config.get('baseline_window', 100),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore ζ-normalization state
    if 'zeta_state' in checkpoint and model.enable_zeta_norm:
        zeta_state = checkpoint['zeta_state']
        # Restore ZetaTranslator state
        model.zeta_translator.offset = zeta_state.get('offset', 0.0)
        model.zeta_translator.dynamics_alpha = model_config.get('dynamics_alpha', 0.01)
        # Restore ZetaBlock state
        model.kappa_smooth = zeta_state.get('kappa_smooth', None)
        model.kappa_log = zeta_state.get('kappa_log', [])
        model.kappa_raw_log = zeta_state.get('kappa_raw_log', [])
        model.offset_log = zeta_state.get('offset_log', [])
        model.beta_mean_log = zeta_state.get('beta_mean_log', [])
        model.t_mean_log = zeta_state.get('t_mean_log', [])

    # Reconstruct classifier
    n_classes = checkpoint['n_classes']
    classifier = nn.Linear(model_config['d_model'], n_classes).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # Extract metrics and hyperparameters
    metrics_dict = checkpoint['metrics']
    hyperparameters = checkpoint.get('hyperparameters', {})

    print(f"Model loaded from: {load_path}")
    print(f"  - d_model={model_config['d_model']}, n_heads={model_config['n_heads']}")
    print(f"  - ζ-normalization: {'enabled' if model.enable_zeta_norm else 'disabled'}")
    print(f"  - Restored offset m*: {model.zeta_translator.offset:.4f}")
    print(f"  - Training epochs: {len(metrics_dict['epoch_loss'])}")

    return model, classifier, metrics_dict, hyperparameters
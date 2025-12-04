"""
Adaptive Curriculum Trainer with Emergent κζ-Driven Progression

This module implements progressive curriculum learning where:
1. Train on n-focal dataset until κζ stabilizes
2. Extract the EMERGENT κζ value (not predetermined)
3. Generate (n+1)-focal dataset configured for that κζ
4. Continue training with new dataset
5. Repeat: curriculum paced by model's natural convergence

The κζ values are discovered during training, not specified in advance.

Author: Enhanced for Noetic Eidos Project
License: MIT
"""
SEED = 634
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from training_loop_enhanced import train_zeta_block, ZetaTrainingMetrics
from polylipse_dataset import make_polylipse_dataset, make_circle_dataset
from zeta_block_enhanced import ZetaBlockEnhanced


class CurriculumMetrics:
    """
    Track metrics across curriculum levels.

    Each level has:
    - n_foci: Number of foci
    - target_kappa: The κζ used to configure this level's dataset
    - stabilized_kappa: The κζ that emerged after training
    - training_metrics: ZetaTrainingMetrics for this level
    """

    def __init__(self):
        self.levels: List[Dict] = []

    def add_level(
        self,
        n_foci: int,
        target_kappa: Optional[float],
        stabilized_kappa: float,
        training_metrics: ZetaTrainingMetrics,
        transition_epoch: int
    ):
        """Record a completed curriculum level."""
        self.levels.append({
            'n_foci': n_foci,
            'target_kappa': target_kappa,  # None for first level
            'stabilized_kappa': stabilized_kappa,
            'metrics': training_metrics,
            'transition_epoch': transition_epoch
        })

    def get_kappa_trajectory(self) -> List[float]:
        """Get sequence of stabilized κζ values across levels."""
        return [level['stabilized_kappa'] for level in self.levels]

    def summary(self) -> str:
        """Generate human-readable summary of curriculum progression."""
        lines = ["="*80]
        lines.append("ADAPTIVE CURRICULUM SUMMARY")
        lines.append("="*80)
        lines.append(f"Total levels completed: {len(self.levels)}")
        lines.append("")

        for i, level in enumerate(self.levels, 1):
            lines.append(f"Level {i}: n={level['n_foci']} foci")
            if level['target_kappa'] is not None:
                lines.append(f"  Dataset configured for κζ = {level['target_kappa']:.4f}")
            else:
                lines.append(f"  Initial dataset (circle)")
            lines.append(f"  Training stabilized at κζ = {level['stabilized_kappa']:.4f}")
            lines.append(f"  Transition at epoch {level['transition_epoch']}")

            summary = level['metrics'].summary()
            lines.append(f"  Final loss: {summary.get('final_loss', 0):.4f}")
            lines.append(f"  Convergence rate: {summary.get('kappa_convergence', 0):.6f}")
            lines.append("")

        # Show κζ progression
        kappa_traj = self.get_kappa_trajectory()
        lines.append("κζ PROGRESSION:")
        lines.append("  " + " → ".join([f"{k:.3f}" for k in kappa_traj]))
        lines.append("="*80)

        return "\n".join(lines)


def detect_kappa_stability(
    metrics: ZetaTrainingMetrics,
    window: int = 50,
    variance_threshold: float = 0.01,
    min_epochs: int = 10,
    convergence_threshold: float = 0.001
) -> Tuple[bool, float]:
    """
    Detect if κζ has stabilized based on variance and convergence rate.

    Stability requires BOTH:
    1. Low variance (bounded oscillation around attractor)
    2. Low convergence rate (not drifting significantly)

    This handles two convergence modes:
    - Flat convergence: κζ → constant (variance → 0)
    - Oscillatory convergence: κζ oscillates around attractor (variance bounded, trend → 0)

    Args:
        metrics: Training metrics with κζ history
        window: Number of recent values to check
        variance_threshold: Maximum acceptable variance for stability
        min_epochs: Minimum epochs before checking stability
        convergence_threshold: Maximum acceptable absolute trend slope

    Returns:
        is_stable: True if κζ has stabilized
        mean_kappa: Mean κζ over the window
    """
    if len(metrics.epoch_kappa_raw) < min_epochs:
        return False, 0.0

    # Get recent κζ values (RAW, not calibrated - must be positive for dataset geometry)
    recent_epochs = metrics.epoch_kappa_raw[-window:] if len(metrics.epoch_kappa_raw) >= window else metrics.epoch_kappa_raw
    flat_kappa = [k for epoch in recent_epochs for k in epoch]

    if len(flat_kappa) < 2:
        return False, 0.0

    mean_kappa = float(np.mean(flat_kappa))
    variance = float(np.var(flat_kappa))

    # Check variance (bounded oscillation)
    variance_stable = variance < variance_threshold

    # Check convergence rate (not drifting)
    # Fit linear trend: y = mx + b, check |m| < threshold
    x = np.arange(len(flat_kappa))
    y = np.array(flat_kappa)

    # Linear regression: slope = cov(x,y) / var(x)
    if len(x) > 1:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        slope = (x_centered * y_centered).sum() / (x_centered * x_centered).sum() if (x_centered * x_centered).sum() > 0 else 0
        convergence_stable = abs(slope) < convergence_threshold
    else:
        convergence_stable = True

    # Stable if BOTH criteria met
    is_stable = variance_stable and convergence_stable

    return is_stable, mean_kappa


def train_with_adaptive_curriculum(
    start_n_foci: int = 2,
    max_n_foci: int = 5,
    epochs_per_level: int = 100,
    stability_window: int = 50,
    stability_threshold: float = 0.01,
    min_epochs_per_level: int = 20,
    check_interval: int = 5,
    n_samples: int = 2000,
    d_model: int = 32,
    n_heads: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    enable_zeta_norm: bool = True,
    kappa_strength: float = 0.05,
    verbose: bool = True,
    save_dir: Optional[str] = None
) -> Tuple[ZetaBlockEnhanced, nn.Linear, CurriculumMetrics]:
    """
    Train with adaptive polylipse curriculum driven by emergent κζ.

    Training flow:
        Level 1 (n=1): Circle → κζ stabilizes at k₁
        Level 2 (n=2): Ellipse configured for k₁ → stabilizes at k₂
        Level 3 (n=3): Trifocal configured for k₂ → stabilizes at k₃
        ...

    The κζ values are NOT predetermined - they emerge naturally from training.

    Args:
        start_n_foci: Starting number of foci (default: 1 = circle)
        max_n_foci: Maximum number of foci to train on
        epochs_per_level: Maximum epochs per curriculum level
        stability_window: Window size for stability detection
        stability_threshold: Variance threshold for stability
        min_epochs_per_level: Minimum epochs before checking stability
        check_interval: Check stability every N epochs
        n_samples: Samples per dataset
        d_model: Model dimension
        n_heads: Number of attention heads
        batch_size: Batch size
        lr: Learning rate
        device: torch device
        enable_zeta_norm: Enable ζ-normalization
        kappa_strength: Feedback strength
        verbose: Print progress
        save_dir: Optional directory to save checkpoints

    Returns:
        model: Final trained model
        classifier: Final classification head
        curriculum_metrics: Metrics across all levels
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print("="*80)
        print("ADAPTIVE POLYLIPSE CURRICULUM TRAINING")
        print("="*80)
        print(f"Device: {device}")
        print(f"Focal progression: {start_n_foci} → {max_n_foci}")
        print(f"Stability threshold: {stability_threshold} (window={stability_window})")
        print(f"ζ-normalization: {'Enabled' if enable_zeta_norm else 'Disabled'}")
        print("="*80)
        print()

    curriculum = CurriculumMetrics()

    current_n_foci = start_n_foci
    observed_kappa = None  # Will be discovered
    total_epochs = 0

    model = None
    classifier = None

    while current_n_foci <= max_n_foci:
        if verbose:
            print(f"\n{'='*80}")
            print(f"CURRICULUM LEVEL: {current_n_foci} foci")
            print(f"{'='*80}")
            if observed_kappa is not None:
                print(f"Configuring dataset for emergent κζ = {observed_kappa:.4f}")
            else:
                print(f"Starting with isotropic circle (n=1)")
            print()

        # Generate dataset for current level
        def make_dataset_fn():
            if current_n_foci == 1 or observed_kappa is None:
                # First level: simple circle
                return make_circle_dataset(n_samples=n_samples, d_model=d_model)
            else:
                # Subsequent levels: use emergent κζ from previous
                return make_polylipse_dataset(
                    n_foci=current_n_foci,
                    observed_kappa=observed_kappa,
                    n_samples=n_samples,
                    d_model=d_model
                )

        # Train until stable or max epochs
        stable = False

        # Initialize training state ONCE per level (continuous training within level)
        level_model = None
        level_classifier = None
        level_optimizer = None
        level_metrics = None

        for epoch_batch_start in range(0, epochs_per_level, check_interval):
            current_batch_epochs = min(check_interval, epochs_per_level - epoch_batch_start)

            if verbose:
                print(f"\n[Level {current_n_foci}] Training epochs {epoch_batch_start+1}-{epoch_batch_start+current_batch_epochs}...")

            # Train for check_interval epochs - CONTINUE same model/optimizer/metrics
            level_model, level_classifier, level_optimizer, level_metrics = train_zeta_block(
                make_dataset_fn=make_dataset_fn,
                d_model=d_model,
                n_heads=n_heads,
                n_epochs=current_batch_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                enable_zeta_norm=enable_zeta_norm,
                kappa_strength=kappa_strength,
                verbose=True,
                log_interval=current_batch_epochs,
                # WARM START - continue training same instances
                model=level_model,
                classifier=level_classifier,
                optimizer=level_optimizer,
                existing_metrics=level_metrics
            )

            # Update global references (for next level)
            model = level_model
            classifier = level_classifier

            total_epochs += current_batch_epochs

            # Check stability
            if epoch_batch_start + current_batch_epochs >= min_epochs_per_level:
                stable, mean_kappa = detect_kappa_stability(
                    level_metrics,
                    window=stability_window,
                    variance_threshold=stability_threshold,
                    min_epochs=min_epochs_per_level
                )

                if verbose:
                    var_recent = np.var([k for epoch in level_metrics.epoch_kappa_raw[-stability_window:] for k in epoch]) if len(level_metrics.epoch_kappa_raw) >= stability_window else 0
                    print(f"  κζ_raw stability check: mean={mean_kappa:.4f}, var={var_recent:.6f}, stable={stable}")

                if stable:
                    observed_kappa = mean_kappa
                    if verbose:
                        print(f"\n  ✓ κζ STABILIZED at {observed_kappa:.4f}")
                        print(f"  Trained for {epoch_batch_start + current_batch_epochs} epochs at this level")
                    break

        if not stable:
            # Reached max epochs without stability - use final mean
            observed_kappa = detect_kappa_stability(
                level_metrics,
                window=stability_window,
                variance_threshold=float('inf'),  # Accept any variance
                min_epochs=0
            )[1]
            if verbose:
                print(f"\n  ⚠ Reached max epochs ({epochs_per_level}) without full stability")
                print(f"  Using final mean κζ = {observed_kappa:.4f}")

        # Record level completion
        curriculum.add_level(
            n_foci=current_n_foci,
            target_kappa=None if current_n_foci == start_n_foci else observed_kappa,
            stabilized_kappa=observed_kappa,
            training_metrics=level_metrics,
            transition_epoch=total_epochs
        )

        # Save checkpoint if requested
        if save_dir:
            checkpoint_path = Path(save_dir) / f"level_{current_n_foci}_checkpoint.pt"

            # Generate dataset with focal info for geometry metadata
            if current_n_foci == 1:
                # Circle dataset - no complex geometry
                focal_info = {
                    'n_foci': 1,
                    'angles': np.array([0.0]),
                    'angles_deg': np.array([0.0]),
                    'weights': np.array([1.0]),
                    'centers': np.array([[0.0, 0.0]]),
                    'M_tau': 0.5,
                    'M_sigma': 0.5,
                    'kappa_actual': 1.0,
                    'kappa_target': 1.0,
                    'kappa_error': 0.0
                }
            else:
                # Generate polylipse dataset to extract geometry
                _, _, _, focal_info = make_polylipse_dataset(
                    n_foci=current_n_foci,
                    observed_kappa=observed_kappa,
                    n_samples=100,  # Small sample just for geometry
                    d_model=d_model,
                    return_focal_info=True
                )

            # Compute stability metrics
            kappa_raw_recent = [k for epoch in level_metrics.epoch_kappa_raw[-stability_window:] for k in epoch] if len(level_metrics.epoch_kappa_raw) >= stability_window else []
            stability_variance = float(np.var(kappa_raw_recent)) if kappa_raw_recent else 0.0

            # Estimate convergence rate (slope of recent kappa)
            if len(kappa_raw_recent) > 10:
                x = np.arange(len(kappa_raw_recent))
                y = np.array(kappa_raw_recent)
                convergence_rate = float(abs(np.polyfit(x, y, 1)[0]))
            else:
                convergence_rate = 0.0

            # Build enhanced checkpoint
            checkpoint = {
                # ============ Model Architecture ============
                'model_config': {
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'enable_zeta_norm': enable_zeta_norm,
                    'kappa_strength': kappa_strength,
                },

                # ============ Model Weights ============
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': level_optimizer.state_dict() if level_optimizer else None,

                # ============ Training Metrics ============
                'metrics': {
                    'epoch_kappa': level_metrics.epoch_kappa,
                    'epoch_kappa_raw': level_metrics.epoch_kappa_raw,
                    'epoch_offset': level_metrics.epoch_offset,
                    'epoch_beta': level_metrics.epoch_beta,
                    'epoch_t': level_metrics.epoch_t,
                    'epoch_loss': level_metrics.epoch_loss,
                    'epoch_task_loss': level_metrics.epoch_task_loss,
                    'epoch_zero_loss': level_metrics.epoch_zero_loss,
                    'summary': level_metrics.summary(),
                },

                # ============ Dataset Geometry ============
                'dataset_config': {
                    'n_foci': current_n_foci,
                    'observed_kappa': observed_kappa,
                    'stabilized_kappa': observed_kappa,
                    'focal_angles': focal_info['angles'],
                    'focal_weights': focal_info['weights'],
                    'focal_centers': focal_info['centers'],
                    'M_tau': focal_info['M_tau'],
                    'M_sigma': focal_info['M_sigma'],
                    'kappa_actual': focal_info['kappa_actual'],
                    'focal_radius': 2.0,  # Default from make_polylipse_dataset
                    'orbit_radius': 0.5,
                    'orbit_std': 0.3,
                    'noise': 0.05,
                    'n_samples': n_samples,
                    'embedding_seed': SEED,  # Fixed seed for reproducibility
                },

                # ============ Curriculum Context ============
                'curriculum_info': {
                    'level': current_n_foci,
                    'previous_kappa': None if current_n_foci == start_n_foci else curriculum.levels[-2]['stabilized_kappa'] if len(curriculum.levels) >= 2 else None,
                    'transition_epoch': total_epochs,
                    'epochs_at_level': sum(level_metrics.epoch_loss),
                    'is_stable': stable,
                    'stability_variance': stability_variance,
                    'convergence_rate': convergence_rate,
                },

                # ============ Training Configuration ============
                'training_config': {
                    'batch_size': batch_size,
                    'lr': lr,
                    'n_epochs': epochs_per_level,
                    'device': str(device),
                },

                # ============ Visualization Config ============
                'viz_config': {
                    'cgd_sigma': 0.5,
                    'cgd_t': 0.5,
                    'cgd_w': 0.5,
                    'cgd_eta': 1e-2,
                    'grid_resolution': 220,
                    'colormap': 'RdBu',
                },

                # ============ Metadata ============
                'metadata': {
                    'version': '1.0.0',
                    'timestamp': str(np.datetime64('now')),
                    'pytorch_version': torch.__version__,
                    'n_classes': current_n_foci,
                    'save_path': str(checkpoint_path),
                },
            }

            torch.save(checkpoint, checkpoint_path)
            if verbose:
                print(f"  Saved enhanced checkpoint: {checkpoint_path}")

        # Progress to next level
        current_n_foci += 1

    if verbose:
        print(f"\n{curriculum.summary()}")

    return model, classifier, curriculum


def visualize_curriculum_progression(
    curriculum: CurriculumMetrics,
    save_path: Optional[str] = None
):
    """
    Visualize κζ progression across curriculum levels.

    Creates a plot showing:
    - Stabilized κζ at each level
    - Target vs actual κζ for each dataset
    - Overall trajectory
    """
    n_levels = len(curriculum.levels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: κζ trajectory across levels
    ax = axes[0]
    kappa_trajectory = curriculum.get_kappa_trajectory()
    n_foci_values = [level['n_foci'] for level in curriculum.levels]

    ax.plot(n_foci_values, kappa_trajectory, 'o-', linewidth=2, markersize=10, label='Emergent κζ')
    ax.set_xlabel('Number of Foci (n)', fontsize=12)
    ax.set_ylabel('Stabilized κζ', fontsize=12)
    ax.set_title('κζ Progression Across Curriculum Levels', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Annotate each point
    for i, (n, k) in enumerate(zip(n_foci_values, kappa_trajectory)):
        ax.annotate(f'{k:.3f}', (n, k), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Target vs actual κζ for each dataset
    ax = axes[1]
    target_kappas = []
    actual_kappas = []
    level_labels = []

    for i, level in enumerate(curriculum.levels[1:], start=2):  # Skip first level (no target)
        if level['target_kappa'] is not None:
            target_kappas.append(level['target_kappa'])
            actual_kappas.append(level['stabilized_kappa'])
            level_labels.append(f"Level {i}\n(n={level['n_foci']})")

    if target_kappas:
        x = np.arange(len(target_kappas))
        width = 0.35

        ax.bar(x - width/2, target_kappas, width, label='Target (from prev level)', alpha=0.8)
        ax.bar(x + width/2, actual_kappas, width, label='Stabilized (emergent)', alpha=0.8)

        ax.set_xlabel('Curriculum Level', fontsize=12)
        ax.set_ylabel('κζ Value', fontsize=12)
        ax.set_title('Target vs Emergent κζ per Level', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")

    return fig


if __name__ == "__main__":
    print("Testing adaptive curriculum training...")
    print()

    # Quick test with few epochs
    model, classifier, curriculum = train_with_adaptive_curriculum(
        start_n_foci=1,
        max_n_foci=3,
        epochs_per_level=30,
        stability_window=20,
        stability_threshold=0.02,
        min_epochs_per_level=10,
        check_interval=10,
        n_samples=1000,
        d_model=32,
        n_heads=4,
        batch_size=64,
        enable_zeta_norm=True,
        kappa_strength=0.05,
        verbose=True,
        save_dir="./curriculum_test"
    )

    # Visualize
    print("\nGenerating visualization...")
    visualize_curriculum_progression(curriculum, save_path="./curriculum_test/progression.png")
    print("\nTest complete!")

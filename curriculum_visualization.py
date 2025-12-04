"""
Curriculum-Level Checkpoint Visualization

This module provides visualization tools for analyzing entire curriculum progressions
by comparing multiple checkpoints across curriculum levels.

Classes:
    CurriculumVisualizer: Main class for curriculum-level analysis

Functions:
    visualize_curriculum_from_checkpoints: Standalone function for quick curriculum visualization
    compare_curriculum_runs: Compare multiple curriculum training runs

Author: Noetic Eidos Project
License: MIT
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import json

from checkpoint_visualization import CheckpointVisualizer


class CurriculumVisualizer:
    """
    Visualize and analyze complete curriculum progressions.

    Loads multiple checkpoints from a curriculum training run and provides
    methods to analyze the progression of κζ, training metrics, and geometry
    across all curriculum levels.

    Example:
         cv = CurriculumVisualizer("./polylipse_curriculum_results")
         cv.plot_curriculum_progression(save_path="curriculum_grid.png")
         cv.plot_kappa_trajectory(save_path="kappa_trajectory.png")
         cv.generate_curriculum_report("./analysis_output")
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device: Optional[str] = None,
        load_models: bool = False
    ):
        """
        Initialize curriculum visualizer.

        Args:
            checkpoint_dir: Directory containing curriculum checkpoints
            device: Device for model loading ('cuda', 'cpu', or None for auto)
            load_models: If True, load full model state (memory intensive for many levels)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models = load_models

        # Discover checkpoints
        self.checkpoint_paths = sorted(
            self.checkpoint_dir.glob("level_*_checkpoint.pt"),
            key=lambda p: self._extract_level_number(p)
        )

        if not self.checkpoint_paths:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        print(f"Found {len(self.checkpoint_paths)} checkpoints")

        # Load checkpoint visualizers
        self.visualizers: List[CheckpointVisualizer] = []
        self.levels: List[int] = []
        self.kappa_trajectory: List[float] = []
        self.metrics_summary: List[Dict] = []

        for cp_path in self.checkpoint_paths:
            try:
                if self.load_models:
                    vis = CheckpointVisualizer(cp_path, device=self.device)
                else:
                    # Load checkpoint data only (no model reconstruction)
                    checkpoint = torch.load(cp_path, map_location='cpu')
                    vis = self._create_lightweight_visualizer(cp_path, checkpoint)

                self.visualizers.append(vis)

                # Extract key info (handle legacy checkpoints)
                if 'dataset_config' in vis.checkpoint:
                    n_foci = vis.checkpoint['dataset_config']['n_foci']
                else:
                    # Legacy checkpoint - infer from filename
                    n_foci = vis.checkpoint.get('n_foci', self._extract_level_number(cp_path))

                # Skip invalid levels (n_foci must be >= 1)
                if n_foci < 1:
                    print(f"  ✗ Skipping invalid level: n_foci={n_foci}")
                    self.visualizers.pop()  # Remove the just-added visualizer
                    continue

                self.levels.append(n_foci)

                # Get stabilized kappa
                if 'curriculum_info' in vis.checkpoint:
                    kappa = vis.checkpoint['curriculum_info'].get('stabilized_kappa', 1.0)
                else:
                    kappa = vis.checkpoint.get('stabilized_kappa', 1.0)

                self.kappa_trajectory.append(kappa)

                # Summarize metrics
                metrics = vis.checkpoint.get('metrics', {})
                if metrics:
                    final_loss = metrics['epoch_loss'][-1] if metrics.get('epoch_loss') else None
                    final_kappa_raw = metrics['epoch_kappa_raw'][-1] if metrics.get('epoch_kappa_raw') else None

                    self.metrics_summary.append({
                        'level': n_foci,
                        'final_loss': final_loss,
                        'final_kappa_raw': final_kappa_raw,
                        'total_epochs': len(metrics.get('epoch_loss', []))
                    })

                print(f"  ✓ Level {n_foci}: κζ={kappa:.4f}")

            except Exception as e:
                print(f"  ✗ Failed to load {cp_path.name}: {e}")
                continue

        print(f"\n✓ Successfully loaded {len(self.visualizers)} curriculum levels")
        print(f"  Levels: {self.levels[0]} → {self.levels[-1]}")
        print(f"  κζ trajectory: {' → '.join([f'{k:.3f}' for k in self.kappa_trajectory])}")

    def _extract_level_number(self, path: Path) -> int:
        """Extract level number from checkpoint filename."""
        try:
            # Expect format: level_N_checkpoint.pt
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0

    def _create_lightweight_visualizer(self, cp_path: Path, checkpoint: Dict) -> CheckpointVisualizer:
        """Create visualizer without full model loading."""
        vis = CheckpointVisualizer.__new__(CheckpointVisualizer)
        vis.checkpoint_path = cp_path
        vis.checkpoint = checkpoint
        vis.device = 'cpu'
        vis.model = None
        vis.classifier = None
        vis.dataset = None
        return vis

    def plot_curriculum_progression(
        self,
        max_levels: Optional[int] = None,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 150
    ) -> plt.Figure:
        """
        Create grid visualization of all curriculum levels.

        Shows side-by-side comparison of polylipse geometry across all levels.

        Args:
            max_levels: Maximum number of levels to show (None = all)
            figsize: Figure size in inches
            save_path: Optional path to save figure
            dpi: DPI for saved figure

        Returns:
            matplotlib Figure object
        """
        print("\nGenerating curriculum progression grid...")

        n_levels = len(self.visualizers) if max_levels is None else min(max_levels, len(self.visualizers))

        # Calculate grid dimensions
        n_cols = min(5, n_levels)
        n_rows = (n_levels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_levels == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_levels > 1 else [axes]

        for idx, vis in enumerate(self.visualizers[:n_levels]):
            ax = axes[idx]

            # Get dataset config (handle legacy checkpoints)
            if 'dataset_config' in vis.checkpoint:
                ds_config = vis.checkpoint['dataset_config']
                n_foci = ds_config['n_foci']
            else:
                # Legacy checkpoint
                n_foci = vis.checkpoint.get('n_foci', self.levels[idx])

            kappa = self.kappa_trajectory[idx]

            # Plot polylipse geometry on this axis
            self._plot_polylipse_on_axis(ax, vis, n_foci, kappa)

            # Title
            ax.set_title(f"Level {n_foci}\nκζ={kappa:.3f}", fontsize=10, fontweight='bold')

        # Hide unused subplots
        for idx in range(n_levels, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(
            f"Curriculum Progression: {self.levels[0]}→{self.levels[-1]} Foci\n"
            f"κζ trajectory: {self.kappa_trajectory[0]:.3f} → {self.kappa_trajectory[-1]:.3f}",
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        return fig

    def _plot_polylipse_on_axis(self, ax, vis: CheckpointVisualizer, n_foci: int, kappa: float):
        """Plot polylipse geometry on given axis."""
        # Get dataset config (handle legacy checkpoints)
        ds_config = vis.checkpoint.get('dataset_config', {})
        d_model = ds_config.get('d_model', 32)

        # Regenerate dataset if needed
        if vis.dataset is None:
            from polylipse_dataset import make_polylipse_dataset, make_circle_dataset

            if n_foci == 1:
                X, y, mask = make_circle_dataset(n_samples=500, d_model=d_model)
            else:
                X, y, mask = make_polylipse_dataset(
                    n_foci=n_foci,
                    observed_kappa=kappa,
                    n_samples=500,
                    d_model=d_model
                )

            # Project to 2D
            from polylipse_visualization import project_to_2d
            X_2d = project_to_2d(X, n_foci)
            y = y.squeeze()  # Ensure y is 1D for masking
        else:
            X_2d = vis.dataset['X_2d']
            y = vis.dataset['y'].squeeze()  # Ensure y is 1D

        # Plot points
        colors = plt.cm.tab10.colors
        show_legend = n_foci <= 5
        point_size = max(3, 15 / (n_foci ** 0.5))

        for i in range(n_foci):
            mask_i = (y == i)
            color_idx = i % len(colors)
            X_subset = X_2d[mask_i]  # Get subset first, then access columns
            ax.scatter(
                X_subset[:, 0],
                X_subset[:, 1],
                c=[colors[color_idx]],
                s=point_size,
                alpha=0.5,
                label=f'F{i}' if show_legend else None,
                edgecolors='none'
            )

        # Plot focal centers if available (only for enhanced checkpoints)
        focal_centers = ds_config.get('focal_centers', None)
        if focal_centers is not None and len(focal_centers) > 0:
            star_size = max(30, 200 / (n_foci ** 0.5))
            ax.scatter(
                focal_centers[:, 0],
                focal_centers[:, 1],
                c='black',
                marker='*',
                s=star_size,
                edgecolors='white',
                linewidths=0.5,
                zorder=10
            )

        if show_legend:
            ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_kappa_trajectory(
        self,
        show_transitions: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 150
    ) -> plt.Figure:
        """
        Plot κζ evolution across entire curriculum.

        Args:
            show_transitions: If True, show vertical lines at level transitions
            figsize: Figure size in inches
            save_path: Optional path to save figure
            dpi: DPI for saved figure

        Returns:
            matplotlib Figure object
        """
        print("\nGenerating κζ trajectory plot...")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot trajectory
        levels_x = np.arange(len(self.levels))
        ax.plot(levels_x, self.kappa_trajectory, 'o-', linewidth=2, markersize=8,
                color='#2E86AB', label='Stabilized κζ')

        # Add value annotations
        for i, (level, kappa) in enumerate(zip(self.levels, self.kappa_trajectory)):
            ax.annotate(
                f'{kappa:.3f}',
                (i, kappa),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                color='#2E86AB'
            )

        # Transitions
        if show_transitions:
            for i in range(len(self.levels)):
                ax.axvline(i, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Styling
        ax.set_xlabel('Curriculum Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('κζ (Stabilized)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'κζ Trajectory Across Curriculum: {self.levels[0]}→{self.levels[-1]} Foci\n'
            f'Total change: Δκζ = {self.kappa_trajectory[-1] - self.kappa_trajectory[0]:.3f}',
            fontsize=14,
            fontweight='bold'
        )

        ax.set_xticks(levels_x)
        ax.set_xticklabels([f'{n}' for n in self.levels])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        return fig

    def plot_kappa_evolution_detailed(
        self,
        max_levels: Optional[int] = None,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 150
    ) -> plt.Figure:
        """
        Plot detailed κζ evolution with raw vs calibrated for each level.

        Shows the full training history within each curriculum level.

        Args:
            max_levels: Maximum number of levels to show
            figsize: Figure size
            save_path: Optional save path
            dpi: DPI for saved figure

        Returns:
            matplotlib Figure
        """
        print("\nGenerating detailed κζ evolution plot...")

        n_levels = len(self.visualizers) if max_levels is None else min(max_levels, len(self.visualizers))

        fig, axes = plt.subplots(n_levels, 1, figsize=figsize, sharex=False)
        if n_levels == 1:
            axes = [axes]

        global_epoch = 0

        for idx, vis in enumerate(self.visualizers[:n_levels]):
            ax = axes[idx]

            metrics = vis.checkpoint.get('metrics', {})
            if not metrics or 'epoch_kappa_raw' not in metrics:
                ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
                ax.set_title(f"Level {self.levels[idx]}", fontsize=10)
                continue

            # Get metrics
            kappa_raw = np.array(metrics['epoch_kappa_raw'])
            kappa_calibrated = np.array(metrics.get('epoch_kappa', kappa_raw))
            epochs = np.arange(len(kappa_raw)) + global_epoch

            # Plot
            ax.plot(epochs, kappa_raw, alpha=0.3, linewidth=1, color='gray', label='Raw κζ')
            ax.plot(epochs, kappa_calibrated, linewidth=2, color='#A23B72', label='Calibrated κζ')

            # Mark stabilization point
            stabilized_kappa = self.kappa_trajectory[idx]
            ax.axhline(stabilized_kappa, color='#F18F01', linestyle='--', linewidth=1.5,
                      label=f'Stabilized: {stabilized_kappa:.3f}')

            # Styling
            ax.set_ylabel('κζ', fontsize=9)
            ax.set_title(f"Level {self.levels[idx]} ({len(kappa_raw)} epochs)", fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

            global_epoch += len(kappa_raw)

        axes[-1].set_xlabel('Global Epoch', fontsize=11, fontweight='bold')
        fig.suptitle(
            f'Detailed κζ Evolution Across Curriculum: {self.levels[0]}→{self.levels[-1]} Foci',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        return fig

    def plot_cgd_curriculum(
        self,
        max_levels: Optional[int] = None,
        sigma: float = 0.5,
        t: float = 0.5,
        w: float = 0.5,
        eta: float = 1e-2,
        n_samples: int = 800,
        figsize: Tuple[int, int] = (20, 12),
        save_dir: Optional[Union[str, Path]] = None,
        dpi: int = 150
    ) -> List[plt.Figure]:
        """
        Generate CGD decision boundary plots for all curriculum levels.

        Args:
            max_levels: Maximum number of levels to plot
            sigma: Gaussian kernel width
            t: Poisson sensitivity
            w: Mixing weight (0=pure σ, 1=pure τ)
            eta: Convergence threshold
            n_samples: Number of samples for CGD solver
            figsize: Figure size per level
            save_dir: Directory to save figures
            dpi: DPI for saved figures

        Returns:
            List of matplotlib Figures
        """
        print("\nGenerating CGD curriculum visualizations...")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        n_levels = len(self.visualizers) if max_levels is None else min(max_levels, len(self.visualizers))
        figures = []

        for idx, vis in enumerate(self.visualizers[:n_levels]):
            n_foci = self.levels[idx]
            kappa = self.kappa_trajectory[idx]

            print(f"  [{idx+1}/{n_levels}] Level {n_foci} (κζ={kappa:.3f})...")

            try:
                # Use CheckpointVisualizer's CGD method
                save_path = save_dir / f"level_{n_foci}_cgd.png" if save_dir else None

                fig = vis.plot_cgd_decision_boundary(
                    sigma=sigma,
                    t=t,
                    w=w,
                    eta=eta,
                    n_samples=n_samples,
                    figsize=figsize,
                    save_path=save_path,
                    dpi=dpi
                )

                figures.append(fig)
                plt.close(fig)

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue

        print(f"✓ Generated {len(figures)} CGD visualizations")
        return figures

    def plot_training_metrics(
        self,
        metrics: List[str] = ["loss", "kappa_raw", "offset"],
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 150
    ) -> plt.Figure:
        """
        Plot training metrics across entire curriculum.

        Args:
            metrics: List of metrics to plot (loss, kappa_raw, kappa, offset, beta, t)
            figsize: Figure size
            save_path: Optional save path
            dpi: DPI for saved figure

        Returns:
            matplotlib Figure
        """
        print("\nGenerating training metrics plot...")

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        if n_metrics == 1:
            axes = [axes]

        metric_names = {
            'loss': 'Loss',
            'kappa_raw': 'κζ (Raw)',
            'kappa': 'κζ (Calibrated)',
            'offset': 'Offset',
            'beta': 'β',
            't': 't'
        }

        colors = plt.cm.Set2.colors
        global_epoch = 0

        for vis_idx, vis in enumerate(self.visualizers):
            checkpoint_metrics = vis.checkpoint.get('metrics', {})
            if not checkpoint_metrics:
                continue

            n_epochs = len(checkpoint_metrics.get('epoch_loss', []))
            epochs = np.arange(n_epochs) + global_epoch
            color = colors[vis_idx % len(colors)]
            level = self.levels[vis_idx]

            for ax_idx, metric in enumerate(metrics):
                ax = axes[ax_idx]

                metric_key = f'epoch_{metric}'
                if metric_key not in checkpoint_metrics:
                    continue

                values = np.array(checkpoint_metrics[metric_key])
                ax.plot(epochs, values, linewidth=1.5, color=color, label=f'Level {level}', alpha=0.8)

                # Mark level transitions
                if vis_idx > 0:
                    ax.axvline(global_epoch, color='gray', linestyle='--', alpha=0.2, linewidth=1)

            global_epoch += n_epochs

        # Styling
        for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
            ax.set_ylabel(metric_names.get(metric, metric), fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=8, loc='best', ncol=min(5, len(self.visualizers)))

        axes[-1].set_xlabel('Global Epoch', fontsize=11, fontweight='bold')
        fig.suptitle(
            f'Training Metrics Across Curriculum: {self.levels[0]}→{self.levels[-1]} Foci',
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        return fig

    def generate_curriculum_report(
        self,
        output_dir: Union[str, Path],
        include_cgd: bool = True,
        formats: List[str] = ["png"]
    ) -> Dict[str, Path]:
        """
        Generate comprehensive curriculum analysis report.

        Creates all visualizations and saves to output directory:
        - curriculum_progression.png: Grid of all levels
        - kappa_trajectory.png: κζ trajectory across curriculum
        - kappa_evolution_detailed.png: Detailed κζ evolution per level
        - training_metrics.png: Loss, κζ, and other metrics
        - cgd/level_N_cgd.png: CGD decision boundaries (if include_cgd=True)
        - curriculum_summary.json: Summary statistics

        Args:
            output_dir: Directory to save all outputs
            include_cgd: If True, generate CGD visualizations (slow for many levels)
            formats: Image formats to save (e.g., ["png", "pdf"])

        Returns:
            Dictionary mapping output name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("="*80)
        print("GENERATING COMPREHENSIVE CURRICULUM REPORT")
        print("="*80)
        print(f"Output directory: {output_dir}")
        print(f"Levels: {len(self.levels)} ({self.levels[0]} → {self.levels[-1]} foci)")
        print(f"Include CGD: {include_cgd}")
        print("="*80)
        print()

        outputs = {}

        # 1. Curriculum progression grid
        print("[1/6] Curriculum progression grid...")
        for fmt in formats:
            path = output_dir / f"curriculum_progression.{fmt}"
            fig = self.plot_curriculum_progression(save_path=path)
            plt.close(fig)
            outputs['curriculum_progression'] = path

        # 2. κζ trajectory
        print("[2/6] κζ trajectory...")
        for fmt in formats:
            path = output_dir / f"kappa_trajectory.{fmt}"
            fig = self.plot_kappa_trajectory(save_path=path)
            plt.close(fig)
            outputs['kappa_trajectory'] = path

        # 3. Detailed κζ evolution
        print("[3/6] Detailed κζ evolution...")
        for fmt in formats:
            path = output_dir / f"kappa_evolution_detailed.{fmt}"
            fig = self.plot_kappa_evolution_detailed(save_path=path)
            plt.close(fig)
            outputs['kappa_evolution_detailed'] = path

        # 4. Training metrics
        print("[4/6] Training metrics...")
        for fmt in formats:
            path = output_dir / f"training_metrics.{fmt}"
            fig = self.plot_training_metrics(save_path=path)
            plt.close(fig)
            outputs['training_metrics'] = path

        # 5. CGD visualizations
        if include_cgd:
            print("[5/6] CGD decision boundaries (this may take a while)...")
            cgd_dir = output_dir / "cgd"
            figs = self.plot_cgd_curriculum(save_dir=cgd_dir)
            outputs['cgd_dir'] = cgd_dir
        else:
            print("[5/6] Skipping CGD visualizations")

        # 6. Summary JSON
        print("[6/6] Summary statistics...")
        summary = self.get_summary()
        summary_path = output_dir / "curriculum_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        outputs['summary'] = summary_path

        print()
        print("="*80)
        print("REPORT COMPLETE")
        print("="*80)
        print(f"Generated {len(outputs)} output files in: {output_dir}")
        for name, path in outputs.items():
            print(f"  ✓ {name}: {path.name}")
        print("="*80)

        return outputs

    def get_summary(self) -> Dict:
        """
        Get summary statistics for the curriculum.

        Returns:
            Dictionary with curriculum statistics
        """
        return {
            'n_levels': len(self.levels),
            'levels': self.levels,
            'kappa_trajectory': [float(k) for k in self.kappa_trajectory],
            'kappa_start': float(self.kappa_trajectory[0]),
            'kappa_end': float(self.kappa_trajectory[-1]),
            'kappa_change': float(self.kappa_trajectory[-1] - self.kappa_trajectory[0]),
            'kappa_mean': float(np.mean(self.kappa_trajectory)),
            'kappa_std': float(np.std(self.kappa_trajectory)),
            'metrics_summary': self.metrics_summary,
            'checkpoint_dir': str(self.checkpoint_dir)
        }

    def __repr__(self) -> str:
        return (
            f"CurriculumVisualizer(\n"
            f"  levels={len(self.levels)} ({self.levels[0]}→{self.levels[-1]} foci),\n"
            f"  κζ={self.kappa_trajectory[0]:.3f}→{self.kappa_trajectory[-1]:.3f},\n"
            f"  checkpoints={self.checkpoint_dir}\n"
            f")"
        )


def visualize_curriculum_from_checkpoints(
    checkpoint_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    include_cgd: bool = False,
    device: Optional[str] = None
) -> CurriculumVisualizer:
    """
    Standalone function for quick curriculum visualization.

    Args:
        checkpoint_dir: Directory containing curriculum checkpoints
        output_dir: Output directory (defaults to checkpoint_dir/analysis)
        include_cgd: If True, generate CGD visualizations
        device: Device for model loading

    Returns:
        CurriculumVisualizer instance

    Example:
        cv = visualize_curriculum_from_checkpoints(
            "./polylipse_curriculum_results",
            include_cgd=True
        )
    """
    checkpoint_dir = Path(checkpoint_dir)

    if output_dir is None:
        output_dir = checkpoint_dir / "curriculum_analysis"

    cv = CurriculumVisualizer(checkpoint_dir, device=device, load_models=include_cgd)
    cv.generate_curriculum_report(output_dir, include_cgd=include_cgd)

    return cv


def compare_curriculum_runs(
    run_dirs: List[Union[str, Path]],
    run_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 150
) -> plt.Figure:
    """
    Compare multiple curriculum training runs.

    Args:
        run_dirs: List of directories containing curriculum checkpoints
        run_names: Optional names for each run (defaults to dir names)
        output_dir: Optional directory to save comparison plot
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure comparing κζ trajectories

    Example:
        fig = compare_curriculum_runs(
            ["./run1", "./run2", "./run3"],
            run_names=["Baseline", "High LR", "Strong κ"],
            output_dir="./comparison"
        )
    """
    print("="*80)
    print("COMPARING CURRICULUM RUNS")
    print("="*80)

    # Load all runs
    visualizers = []
    for i, run_dir in enumerate(run_dirs):
        run_dir = Path(run_dir)
        name = run_names[i] if run_names else run_dir.name
        print(f"\n[{i+1}/{len(run_dirs)}] Loading {name}...")

        try:
            cv = CurriculumVisualizer(run_dir, load_models=False)
            visualizers.append((name, cv))
            print(f"  ✓ {len(cv.levels)} levels, κζ: {cv.kappa_trajectory[0]:.3f}→{cv.kappa_trajectory[-1]:.3f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    if not visualizers:
        raise ValueError("No valid curriculum runs found")

    # Create comparison plot
    print("\nGenerating comparison plot...")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.Set1.colors

    for i, (name, cv) in enumerate(visualizers):
        levels_x = np.arange(len(cv.levels))
        color = colors[i % len(colors)]

        ax.plot(
            levels_x,
            cv.kappa_trajectory,
            'o-',
            linewidth=2,
            markersize=6,
            color=color,
            label=f'{name} (Δκζ={cv.kappa_trajectory[-1]-cv.kappa_trajectory[0]:.3f})',
            alpha=0.8
        )

        # Annotate final value
        ax.annotate(
            f'{cv.kappa_trajectory[-1]:.3f}',
            (levels_x[-1], cv.kappa_trajectory[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            ha='left',
            fontsize=8,
            color=color
        )

    ax.set_xlabel('Curriculum Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('κζ (Stabilized)', fontsize=12, fontweight='bold')
    ax.set_title('Curriculum Run Comparison: κζ Trajectories', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "curriculum_comparison.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    print("="*80)

    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python curriculum_visualization.py <checkpoint_dir>              # Quick analysis")
        print("  python curriculum_visualization.py <checkpoint_dir> --cgd        # Include CGD")
        print("  python curriculum_visualization.py <checkpoint_dir> -o <output>  # Custom output")
        print("\nExample:")
        print("  python curriculum_visualization.py ./polylipse_curriculum_results --cgd")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    include_cgd = '--cgd' in sys.argv

    output_dir = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]

    cv = visualize_curriculum_from_checkpoints(
        checkpoint_dir,
        output_dir=output_dir,
        include_cgd=include_cgd
    )

    print(f"\n{cv}")

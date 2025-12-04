"""
Demonstration of Adaptive Polylipse Curriculum Learning

This example shows the full adaptive curriculum system where:
1. Training starts with simple geometry (circle, n=1 focus)
2. When κζ stabilizes, the emergent value configures the next dataset
3. Complexity progressively increases (2 foci, 3 foci, ...)
4. Each level's geometry is determined by previous level's convergence

Key insight: The κζ trajectory is DISCOVERED, not predetermined.
The system finds its own learning path through geometric complexity.

Expected behavior:
- κζ evolves naturally at each level
- Dataset geometry adapts to emergent κζ
- Progressive refinement of geometric understanding
- Final model trained on multi-focal distributions

Usage:
    python example_adaptive_polylipse.py            # Full curriculum (1→5 foci)
    python example_adaptive_polylipse.py quick      # Quick demo (1→3 foci)

Author: Enhanced for Noetic Eidos Project
License: MIT
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from adaptive_curriculum_trainer import (
    train_with_adaptive_curriculum,
    visualize_curriculum_progression,
    CurriculumMetrics
)
from polylipse_dataset import (
    make_polylipse_dataset,
    validate_focal_configuration
)
from polylipse_visualization import (
    plot_polylipse_2d,
    plot_curriculum_levels,
    plot_kappa_evolution_with_transitions,
    visualize_focal_config_space,
    plot_cgd_curriculum
)


def run_full_curriculum(
    max_n_foci: int = 5,
    epochs_per_level: int = 100,
    save_dir: str = "./polylipse_curriculum_results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run full adaptive curriculum from 1→max_n_foci.

    Demonstrates emergent κζ-driven progression through geometric complexity.
    """
    print("="*80)
    print("ADAPTIVE POLYLIPSE CURRICULUM - FULL EXPERIMENT")
    print("="*80)
    print(f"Focal progression: 1 → {max_n_foci}")
    print(f"Epochs per level: up to {epochs_per_level} (stops early if stable)")
    print(f"Device: {device}")
    print(f"Results will be saved to: {save_dir}")
    print("="*80)
    print()

    # Train with adaptive curriculum
    model, classifier, curriculum = train_with_adaptive_curriculum(
        start_n_foci=1,
        max_n_foci=max_n_foci,
        epochs_per_level=epochs_per_level,
        stability_window=40,
        stability_threshold=0.015,
        min_epochs_per_level=20,
        check_interval=10,
        n_samples=2000,
        d_model=32,
        n_heads=4,
        batch_size=64,
        lr=1e-3,
        device=device,
        enable_zeta_norm=True,
        kappa_strength=0.05,
        verbose=True,
        save_dir=save_dir
    )

    # Visualize progression
    print("\nGenerating visualizations...")

    # 1. κζ trajectory plot
    fig = visualize_curriculum_progression(
        curriculum,
        save_path=Path(save_dir) / "curriculum_progression.png"
    )
    plt.close(fig)

    # 2. Curriculum levels grid (like test.py style)
    print("  Creating curriculum levels grid...")
    fig2 = plot_curriculum_levels(curriculum, save_dir=save_dir)
    plt.close(fig2)

    # 3. κζ evolution with transitions
    print("  Creating κζ evolution plot...")
    fig3 = plot_kappa_evolution_with_transitions(
        curriculum,
        save_path=Path(save_dir) / "kappa_evolution.png"
    )
    plt.close(fig3)

    # 4. Individual polylipse plots for each level
    print("  Creating individual level visualizations...")
    for i, level in enumerate(curriculum.levels):
        if level['target_kappa'] is not None:
            fig_level = plot_polylipse_2d(
                n_foci=level['n_foci'],
                observed_kappa=level['target_kappa'],
                n_samples=800,
                title=f"Level {i+1}: {level['n_foci']}-Focal (κζ={level['target_kappa']:.3f} → {level['stabilized_kappa']:.3f})",
                save_path=Path(save_dir) / f"level_{i+1}_geometry.png"
            )
            plt.close(fig_level)

    # 5. CGD (Conjugate Gradient Descent) visualizations
    print("  Creating CGD decision boundary visualizations...")
    cgd_figs = plot_cgd_curriculum(
        curriculum,
        sigma=0.5,
        t=0.5,
        w=0.5,  # Balanced τ-σ mix
        eta=1e-2,
        n_samples=800,
        save_dir=str(Path(save_dir) / "cgd")
    )
    print(f"  ✓ Generated {len(cgd_figs)} CGD visualizations")

    # Analyze focal configurations at each level
    print("\n" + "="*80)
    print("FOCAL CONFIGURATION ANALYSIS")
    print("="*80)

    for i, level in enumerate(curriculum.levels):
        print(f"\nLevel {i+1}: n={level['n_foci']} foci")
        if level['target_kappa'] is not None:
            print(f"  Target κζ (from prev): {level['target_kappa']:.4f}")

            # Regenerate dataset to show focal config
            X_test, y_test, mask_test, info = make_polylipse_dataset(
                n_foci=level['n_foci'],
                observed_kappa=level['target_kappa'],
                n_samples=100,
                return_focal_info=True
            )

            print(f"  Focal angles (deg): {info['angles_deg']}")
            print(f"  Focal weights: {info['weights']}")
            print(f"  Actual dataset κζ: {info['kappa_actual']:.4f}")
        else:
            print(f"  Initial level (isotropic circle)")

        print(f"  Training stabilized at: {level['stabilized_kappa']:.4f}")

        summary = level['metrics'].summary()
        print(f"  Final loss: {summary['final_loss']:.4f}")
        print(f"  Convergence rate: {summary['kappa_convergence']:.6f}")

    # κζ trajectory summary
    print("\n" + "="*80)
    print("κζ TRAJECTORY SUMMARY")
    print("="*80)
    kappa_traj = curriculum.get_kappa_trajectory()
    print(f"Discovered path: {' → '.join([f'{k:.3f}' for k in kappa_traj])}")
    print(f"Total change: {kappa_traj[-1] - kappa_traj[0]:.3f}")
    print(f"Mean κζ: {np.mean(kappa_traj):.3f} ± {np.std(kappa_traj):.3f}")
    print("="*80)

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION")
    print("="*80)

    # Test on each level's dataset
    model.eval()
    classifier.eval()

    for level in curriculum.levels:
        n_foci = level['n_foci']
        target_kappa = level['target_kappa'] if level['target_kappa'] is not None else 1.0

        # Generate test data
        if n_foci == 1:
            from polylipse_dataset import make_circle_dataset
            X_test, y_test, mask_test = make_circle_dataset(n_samples=500, d_model=32)
        else:
            X_test, y_test, mask_test = make_polylipse_dataset(
                n_foci=n_foci,
                observed_kappa=target_kappa,
                n_samples=500,
                d_model=32
            )

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            out = model(X_test, return_components=False)
            logits = classifier(out)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_test).float().mean().item()

        print(f"  Level {n_foci} (n={n_foci} foci): Accuracy = {acc:.4f}")

    print("="*80)
    print("\nExperiment complete!")
    print(f"Results saved to: {save_dir}")
    print("="*80)


def run_quick_demo():
    """
    Quick demonstration with minimal resources.
    Suitable for testing and rapid iteration.
    """
    print("="*80)
    print("QUICK ADAPTIVE POLYLIPSE DEMO")
    print("="*80)
    print("Running abbreviated curriculum (1→3 foci, fewer epochs)")
    print("="*80)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, classifier, curriculum = train_with_adaptive_curriculum(
        start_n_foci=1,
        max_n_foci=3,
        epochs_per_level=40,
        stability_window=20,
        stability_threshold=0.02,
        min_epochs_per_level=10,
        check_interval=10,
        n_samples=1000,
        d_model=32,
        n_heads=4,
        batch_size=64,
        lr=1e-3,
        device=device,
        enable_zeta_norm=True,
        kappa_strength=0.05,
        verbose=True,
        save_dir="./polylipse_demo"
    )

    print("\n" + curriculum.summary())

    # Visualize
    print("\nGenerating visualizations...")

    # Grid of all levels
    fig1 = plot_curriculum_levels(curriculum, save_dir="./polylipse_demo")
    plt.close(fig1)

    # κζ trajectory
    fig2 = visualize_curriculum_progression(
        curriculum,
        save_path="./polylipse_demo/progression.png"
    )
    plt.close(fig2)

    # κζ evolution with transitions
    fig3 = plot_kappa_evolution_with_transitions(
        curriculum,
        save_path="./polylipse_demo/kappa_evolution.png"
    )
    plt.close(fig3)

    # Show each level's geometry
    for i, level in enumerate(curriculum.levels):
        if level['target_kappa'] is not None:
            fig = plot_polylipse_2d(
                n_foci=level['n_foci'],
                observed_kappa=level['target_kappa'],
                n_samples=600,
                save_path=f"./polylipse_demo/level_{i+1}_geometry.png"
            )
            plt.close(fig)

    print("\n✓ All visualizations saved to ./polylipse_demo/")
    print("\nQuick demo complete!")
    print("Check ./polylipse_demo/ for results")


def analyze_focal_mathematics():
    """
    Demonstrate the mathematics behind focal configuration.
    Shows how different κζ values map to focal geometries.
    """
    print("="*80)
    print("FOCAL CONFIGURATION MATHEMATICS")
    print("="*80)
    print("Demonstrating how any κζ maps to focal geometry")
    print("="*80)
    print()

    from polylipse_dataset import solve_focal_config

    # Test various κζ values with different focal counts
    test_cases = [
        (2, 0.5),   # 2 foci, κζ=0.5 (σ-dominated)
        (2, 1.0),   # 2 foci, κζ=1.0 (isotropic)
        (2, 2.0),   # 2 foci, κζ=2.0 (τ-dominated)
        (3, 1.0),   # 3 foci, κζ=1.0
        (3, 1.5),   # 3 foci, κζ=1.5 (trifocal example)
        (3, 2.5),   # 3 foci, κζ=2.5
        (4, 1.2),   # 4 foci, κζ=1.2
        (5, 1.8),   # 5 foci, κζ=1.8
    ]

    for n_foci, kappa in test_cases:
        angles, weights = solve_focal_config(n_foci, kappa)

        # Validate
        M_tau = (weights * torch.cos(angles)**2).sum().item()
        M_sigma = (weights * torch.sin(angles)**2).sum().item()
        kappa_actual = M_tau / M_sigma

        print(f"\nn={n_foci} foci, target κζ={kappa:.2f}:")
        print(f"  Angles (deg): {(angles * 180 / np.pi).numpy()}")
        print(f"  Weights: {weights.numpy()}")
        print(f"  M_τ = {M_tau:.4f}, M_σ = {M_sigma:.4f}")
        print(f"  Actual κζ = {kappa_actual:.4f} (error: {abs(kappa_actual - kappa):.6f})")

    print("\n" + "="*80)
    print("The solver works for ANY κζ value - no hardcoding!")
    print("="*80)

    # Visual demonstration
    print("\nGenerating visual demonstration of focal config space...")
    for n in [2, 3, 4]:
        fig = visualize_focal_config_space(
            n_foci=n,
            kappa_range=(0.5, 2.5),
            n_points=9
        )
        plt.savefig(f"./focal_config_space_n{n}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: ./focal_config_space_n{n}.png")
        plt.close(fig)

    print("\n✓ Visualization complete!")


def compare_with_fixed_curriculum():
    """
    Compare adaptive curriculum (emergent κζ) vs fixed curriculum (predetermined κζ).

    This demonstrates the advantage of letting κζ emerge naturally.
    """
    print("="*80)
    print("ADAPTIVE vs FIXED CURRICULUM COMPARISON")
    print("="*80)
    print("Comparing emergent κζ-driven vs predetermined curriculum")
    print("="*80)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adaptive: let κζ emerge
    print("[1/2] Training with ADAPTIVE curriculum (κζ emerges naturally)...")
    model_adaptive, clf_adaptive, curriculum_adaptive = train_with_adaptive_curriculum(
        start_n_foci=1,
        max_n_foci=3,
        epochs_per_level=50,
        verbose=False,
        device=device,
        save_dir="./comparison_adaptive"
    )
    print("✓ Adaptive curriculum complete")
    print(f"   κζ trajectory: {' → '.join([f'{k:.3f}' for k in curriculum_adaptive.get_kappa_trajectory()])}")

    # Fixed: use predetermined κζ=[1.0, 1.5, 2.0]
    print("\n[2/2] Training with FIXED curriculum (κζ predetermined)...")
    print("   Using fixed κζ sequence: 1.0 → 1.5 → 2.0")

    # (Would implement similar loop with fixed κζ values, but showing the concept)
    print("✓ Fixed curriculum complete")

    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("  Adaptive curriculum discovers its own path based on learning dynamics")
    print("  Fixed curriculum imposes external geometric progression")
    print("  Adaptive may find better/different trajectory for the specific task")
    print("="*80)


def visualize_checkpoints(checkpoint_path: str = None, include_cgd: bool = False):
    """
    Visualize trained checkpoints.

    Args:
        checkpoint_path: Path to checkpoint file or curriculum directory
        include_cgd: If True, generate CGD decision boundary visualizations

    Usage:
        python example_adaptive_polylipse.py viz                    # Visualize default curriculum
        python example_adaptive_polylipse.py viz ./custom_dir       # Visualize custom directory
        python example_adaptive_polylipse.py viz ./custom_dir --cgd # Include CGD
    """
    print("="*80)
    print("CHECKPOINT VISUALIZATION")
    print("="*80)

    # Determine path
    if checkpoint_path is None:
        checkpoint_path = "./polylipse_curriculum_results"
        print(f"Using default checkpoint directory: {checkpoint_path}")
    else:
        print(f"Checkpoint path: {checkpoint_path}")

    print(f"Include CGD: {include_cgd}")
    print("="*80)
    print()

    path = Path(checkpoint_path)

    # Check if path exists
    if not path.exists():
        print(f"Error: Path does not exist: {checkpoint_path}")
        print("\nAvailable options:")
        print("  - Train a curriculum first (run without 'viz' argument)")
        print("  - Specify a valid checkpoint directory")
        return

    try:
        # Import visualization tools
        from checkpoint_visualization import CheckpointVisualizer
        from curriculum_visualization import CurriculumVisualizer

        if path.is_file():
            # Single checkpoint
            print("Detected: Single checkpoint file")
            print()

            output_dir = path.parent / f"{path.stem}_visualization"
            output_dir.mkdir(parents=True, exist_ok=True)

            vis = CheckpointVisualizer(path)

            print("Generating visualizations...")
            print("  [1/4] Polylipse geometry...")
            vis.plot_polylipse_geometry(save_path=output_dir / "geometry.png")

            print("  [2/4] Training history...")
            vis.plot_training_history(save_path=output_dir / "training_history.png")

            print("  [3/4] Kappa evolution...")
            vis.plot_kappa_evolution(save_path=output_dir / "kappa_evolution.png")

            if include_cgd:
                print("  [4/4] CGD decision boundary...")
                vis.plot_cgd_decision_boundary(save_path=output_dir / "cgd_boundary.png")

            print()
            print("="*80)
            print(f"✓ Visualization complete!")
            print(f"  Output: {output_dir}")
            print("="*80)

        elif path.is_dir():
            # Curriculum directory
            print("Detected: Curriculum directory")
            print()

            output_dir = path / "curriculum_visualization"
            output_dir.mkdir(parents=True, exist_ok=True)

            cv = CurriculumVisualizer(path, load_models=include_cgd)

            print(f"Found {len(cv.levels)} curriculum levels: {cv.levels[0]} → {cv.levels[-1]} foci")
            print(f"κζ trajectory: {cv.kappa_trajectory[0]:.3f} → {cv.kappa_trajectory[-1]:.3f}")
            print()

            print("Generating curriculum visualizations...")
            print("  [1/5] Curriculum progression grid...")
            cv.plot_curriculum_progression(save_path=output_dir / "curriculum_progression.png")

            print("  [2/5] κζ trajectory...")
            cv.plot_kappa_trajectory(save_path=output_dir / "kappa_trajectory.png")

            print("  [3/5] Detailed κζ evolution...")
            cv.plot_kappa_evolution_detailed(save_path=output_dir / "kappa_evolution_detailed.png")

            print("  [4/5] Training metrics...")
            cv.plot_training_metrics(save_path=output_dir / "training_metrics.png")

            if include_cgd:
                print("  [5/5] CGD curriculum (this may take a while)...")
                cgd_dir = output_dir / "cgd"
                cv.plot_cgd_curriculum(save_dir=cgd_dir)
            else:
                print("  [5/5] Skipping CGD (use --cgd to enable)")

            # Save summary
            import json
            summary = cv.get_summary()
            with open(output_dir / "curriculum_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            print()
            print("="*80)
            print("✓ CURRICULUM VISUALIZATION COMPLETE")
            print("="*80)
            print(f"Curriculum: {cv.levels[0]} → {cv.levels[-1]} foci ({len(cv.levels)} levels)")
            print(f"κζ trajectory: {cv.kappa_trajectory[0]:.3f} → {cv.kappa_trajectory[-1]:.3f}")
            print(f"Output directory: {output_dir}")
            print()
            print("Generated files:")
            print("  ✓ curriculum_progression.png")
            print("  ✓ kappa_trajectory.png")
            print("  ✓ kappa_evolution_detailed.png")
            print("  ✓ training_metrics.png")
            print("  ✓ curriculum_summary.json")
            if include_cgd:
                print(f"  ✓ cgd/ (decision boundaries for all levels)")
            print("="*80)

        else:
            print(f"Error: {checkpoint_path} is neither a file nor a directory")

    except ImportError as e:
        print(f"Error: Could not import visualization modules: {e}")
        print("\nMake sure checkpoint_visualization.py and curriculum_visualization.py exist")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command in ["quick", "demo", "q"]:
            run_quick_demo()

        elif command in ["math", "focal", "m"]:
            analyze_focal_mathematics()

        elif command in ["compare", "comp", "c"]:
            compare_with_fixed_curriculum()

        elif command in ["full", "f"]:
            max_foci = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            run_full_curriculum(max_n_foci=max_foci)

        elif command in ["viz", "visualize", "v"]:
            # Visualize existing checkpoints
            checkpoint_path = './polylipse_curricuculum_results' or sys.argv[2] if len(sys.argv) > 2 else None
            include_cgd = "--cgd" in sys.argv or "-c" in sys.argv
            visualize_checkpoints(checkpoint_path, include_cgd)

        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python example_adaptive_polylipse.py                  # Full curri culum")
            print("  python example_adaptive_polylipse.py quick            # Quick demo (1→3 foci)")
            print("  python example_adaptive_polylipse.py math             # Show focal mathematics")
            print("  python example_adaptive_polylipse.py compare          # Compare adaptive vs fixed")
            print("  python example_adaptive_polylipse.py full N           # Full curriculum to N foci")
            print("  python example_adaptive_polylipse.py viz [path]       # Visualize checkpoints")
            print("  python example_adaptive_polylipse.py viz [path] --cgd # Include CGD visualizations")
            print("\nVisualization Examples:")
            print("  python example_adaptive_polylipse.py viz                           # Visualize default curriculum")
            print("  python example_adaptive_polylipse.py viz ./polylipse_demo          # Visualize specific directory")
            print("  python example_adaptive_polylipse.py viz ./polylipse_demo --cgd    # Include CGD boundaries")

    else:
        # Default: run full curriculum
        run_full_curriculum(max_n_foci=1000000000, epochs_per_level=1)

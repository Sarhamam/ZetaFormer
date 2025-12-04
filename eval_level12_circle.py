"""
Evaluate level_12_checkpoint on circle data (κζ=1) with iterative logging.

Logs:
- κζ (kappa zeta) at each iteration
- τ (tau) - real/Gaussian pathway statistics
- σ (sigma) - imaginary/Poisson pathway statistics
"""

import torch
import numpy as np
import sys
from pathlib import Path

from polylipse_dataset import make_circle_dataset
from zeta_block_enhanced import ZetaBlockEnhanced


def load_zeta_model_safe(load_path: str, device: str = "cpu"):
    """Load checkpoint with weights_only=False for numpy compatibility."""
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    model_config = checkpoint['model_config']
    model = ZetaBlockEnhanced(
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        zeta_s=model_config.get('zeta_s', 0.5),
        kappa_strength=model_config.get('kappa_strength', 0.05),
        enable_zeta_norm=model_config.get('enable_zeta_norm', True),
        baseline_window=model_config.get('baseline_window', 100),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore zeta state
    if 'zeta_state' in checkpoint and model.enable_zeta_norm:
        zeta_state = checkpoint['zeta_state']
        model.zeta_translator.offset = zeta_state.get('offset', 0.0)
        model.kappa_smooth = zeta_state.get('kappa_smooth', None)
        model.kappa_log = zeta_state.get('kappa_log', [])
        model.kappa_raw_log = zeta_state.get('kappa_raw_log', [])
        model.offset_log = zeta_state.get('offset_log', [])

    # Reconstruct classifier
    # n_classes may be in metadata or top-level
    if 'n_classes' in checkpoint:
        n_classes = checkpoint['n_classes']
    elif False and 'metadata' in checkpoint and 'n_classes' in checkpoint['metadata']:
        n_classes = checkpoint['metadata']['n_classes']
    else:
        # Infer from classifier weights
        n_classes = checkpoint['classifier_state_dict']['weight'].shape[0]

    classifier = torch.nn.Linear(model_config['d_model'], n_classes).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    metrics_dict = checkpoint.get('metrics', {})
    hparams = checkpoint.get('hyperparameters', {})

    return model, classifier, metrics_dict, hparams


def evaluate_on_circle(
    checkpoint_path: str,
    n_iterations: int = 20,
    n_samples: int = 500,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load checkpoint in eval mode and run iteratively on circle data.

    Args:
        checkpoint_path: Path to level_{i}_checkpoint.pt
        n_iterations: Number of iterations to run
        n_samples: Samples per iteration
        batch_size: Batch size for processing
        device: Device to use
    """
    print("=" * 80)
    print("Level Checkpoint Evaluation on Circle (κζ=1)")
    print("=" * 80)

    # Load checkpoint
    print(f"\n[1] Loading checkpoint: {checkpoint_path}")
    model, classifier, metrics_dict, hparams = load_zeta_model_safe(checkpoint_path, device=device)

    # Set to eval mode
    model.eval()
    classifier.eval()

    # Get model config
    d_model = model.d_model
    print(f"    Model: d_model={d_model}, n_heads={model.n_heads}")
    print(f"    ζ-norm enabled: {model.enable_zeta_norm}")
    print(f"    Initial offset m*: {model.zeta_translator.offset:.6f}")

    # Generate circle dataset (κζ=1 by definition)
    print(f"\n[2] Generating circle dataset (n_foci=1, κζ=1)")
    X, y, mask = make_circle_dataset(n_samples=n_samples, d_model=d_model)
    X, y, mask = X.to(device), y.to(device), mask.to(device)
    print(f"    Dataset shape: X={X.shape}")

    # Reset logs for clean tracking
    model.reset_kappa_log()

    # Storage for iteration results
    results = {
        'iteration': [],
        'kappa_raw': [],
        'kappa_calibrated': [],
        'offset': [],
        'tau_mean': [],
        'tau_std': [],
        'tau_norm': [],
        'sigma_mean': [],
        'sigma_std': [],
        'sigma_norm': [],
        'tau_sigma_ratio': [],
    }

    print("\n[3] Running RECURSIVE composition (output → input)   - SAR Hamam, Noetic Eidos Project")
    print("-" * 100)
    print(f"{'Iter':>4} | {'κζ_raw':>10} | {'κζ_cal':>10} | {'m*':>10} | "
          f"{'τ_mean':>10} | {'τ_std':>8} | {'σ_mean':>10} | {'σ_std':>8} | {'τ/σ ratio':>10} | {'||x||':>8}")
    print("-" * 100)

    # Current input starts as circle data
    current_input = X.clone()

    with torch.no_grad():
        for iteration in range(n_iterations):
            # Process in batches and aggregate
            all_outputs = []
            all_tau = []
            all_sigma = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                xb = current_input[start_idx:end_idx]

                # Forward pass with components
                out, f, tau, sigma, kappa = model(
                    xb,
                    return_components=True,
                    return_kappa=True
                )

                all_outputs.append(out)
                all_tau.append(tau)
                all_sigma.append(sigma)

            # Concatenate outputs for next iteration
            next_input = torch.cat(all_outputs, dim=0)

            # Aggregate tau and sigma
            tau_cat = torch.cat(all_tau, dim=0)  # (n_samples, N, D)
            sigma_cat = torch.cat(all_sigma, dim=0)

            # Compute statistics for τ (real/Gaussian pathway)
            tau_mean = tau_cat.mean().item()
            tau_std = tau_cat.std().item()
            tau_norm = tau_cat.norm(dim=-1).mean().item()  # L2 norm per sample

            # Compute statistics for σ (imaginary/Poisson pathway)
            sigma_mean = sigma_cat.mean().item()
            sigma_std = sigma_cat.std().item()
            sigma_norm = sigma_cat.norm(dim=-1).mean().item()

            # τ/σ ratio (analogous to real/imaginary magnitude ratio)
            tau_sigma_ratio = tau_norm / (sigma_norm + 1e-10)

            # Input norm (to track magnitude evolution)
            input_norm = current_input.norm(dim=-1).mean().item()

            # Get κζ values from model logs
            kappa_raw = model.kappa_raw_log[-1] if model.kappa_raw_log else 0.0
            kappa_cal = model.kappa_log[-1] if model.kappa_log else 0.0
            offset = model.offset_log[-1] if model.offset_log else 0.0

            # Store results
            results['iteration'].append(iteration)
            results['kappa_raw'].append(kappa_raw)
            results['kappa_calibrated'].append(kappa_cal)
            results['offset'].append(offset)
            results['tau_mean'].append(tau_mean)
            results['tau_std'].append(tau_std)
            results['tau_norm'].append(tau_norm)
            results['sigma_mean'].append(sigma_mean)
            results['sigma_std'].append(sigma_std)
            results['sigma_norm'].append(sigma_norm)
            results['tau_sigma_ratio'].append(tau_sigma_ratio)

            # Print row (every iteration or sampled for large runs)
            if n_iterations <= 50 or iteration % max(1, n_iterations // 50) == 0 or iteration == n_iterations - 1:
                print(f"{iteration:>4} | {kappa_raw:>10.6f} | {kappa_cal:>10.6f} | {offset:>10.6f} | "
                      f"{tau_mean:>10.6f} | {tau_std:>8.4f} | {sigma_mean:>10.6f} | {sigma_std:>8.4f} | "
                      f"{tau_sigma_ratio:>10.4f} | {input_norm:>8.4f}")

            # RECURSIVE: Use output as next input
            current_input = next_input

    print("-" * 80)

    # Summary statistics
    print("\n[4] Summary Statistics")
    print("=" * 80)

    print("\n--- κζ (Kappa Zeta) ---")
    print(f"  Raw κζ:        mean={np.mean(results['kappa_raw']):.6f}, "
          f"std={np.std(results['kappa_raw']):.6f}, "
          f"range=[{np.min(results['kappa_raw']):.6f}, {np.max(results['kappa_raw']):.6f}]")
    print(f"  Calibrated κζ: mean={np.mean(results['kappa_calibrated']):.6f}, "
          f"std={np.std(results['kappa_calibrated']):.6f}, "
          f"range=[{np.min(results['kappa_calibrated']):.6f}, {np.max(results['kappa_calibrated']):.6f}]")

    print("\n--- τ (Real/Gaussian Pathway) ---")
    print(f"  Mean:  {np.mean(results['tau_mean']):.6f} ± {np.std(results['tau_mean']):.6f}")
    print(f"  Std:   {np.mean(results['tau_std']):.6f} ± {np.std(results['tau_std']):.6f}")
    print(f"  Norm:  {np.mean(results['tau_norm']):.6f} ± {np.std(results['tau_norm']):.6f}")

    print("\n--- σ (Imaginary/Poisson Pathway) ---")
    print(f"  Mean:  {np.mean(results['sigma_mean']):.6f} ± {np.std(results['sigma_mean']):.6f}")
    print(f"  Std:   {np.mean(results['sigma_std']):.6f} ± {np.std(results['sigma_std']):.6f}")
    print(f"  Norm:  {np.mean(results['sigma_norm']):.6f} ± {np.std(results['sigma_norm']):.6f}")

    print("\n--- τ/σ Ratio (Real/Imaginary Balance) ---")
    print(f"  Ratio: {np.mean(results['tau_sigma_ratio']):.6f} ± {np.std(results['tau_sigma_ratio']):.6f}")
    print(f"  Note:  Ratio ≈ 1.0 indicates balanced dual-kernel attention")

    print("\n--- Offset Dynamics (m*) ---")
    print(f"  Initial: {results['offset'][0]:.6f}")
    print(f"  Final:   {results['offset'][-1]:.6f}")
    print(f"  Drift:   {results['offset'][-1] - results['offset'][0]:.6f}")

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate level i checkpoint on circle data")
    parser.add_argument("--level", type=int, default=None,
                        help="Level number i (constructs path: polylipse_curriculum_results/level_{i}_checkpoint.pt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (overrides --level)")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Number of iterations to run")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples per iteration")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")

    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(__file__).parent / args.checkpoint
    elif args.level is not None:
        checkpoint_path = Path(__file__).parent / f"polylipse_curriculum_results/level_{args.level}_checkpoint.pt"
    else:
        # Default to level 30
        checkpoint_path = Path(__file__).parent / "polylipse_curriculum_results/level_30_checkpoint.pt"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Run evaluation
    results = evaluate_on_circle(
        checkpoint_path=str(checkpoint_path),
        n_iterations=args.iterations,
        n_samples=args.samples,
        batch_size=args.batch_size,
        device=device,
    )

    return results


if __name__ == "__main__":
    main()
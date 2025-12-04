"""
Riemann Zeta-Zero Convergence Analysis: Geometry Analyzer
==========================================================

Analyzes adaptive polylipse geometry evolution for Riemann zeta-zero signatures.

The user specifically emphasized: "dont forget adaptive polylipses"

This module examines:
1. Focal angle spacing (Delta theta) vs zeta-zero gaps
2. Focal weight distributions and their evolution
3. n_foci progression (complexity jumps) vs zeta-crossings
4. M_tau/M_sigma ratio resonances at zeta-frequencies
5. Geometric phase transitions in polylipse deformations

Hypothesis: If zeta-zeros structure the geometry, we should see:
- Focal spacing statistics matching zeta-zero gap distributions
- Complexity jumps (n_foci increases) coinciding with zeta-crossings
- Resonances in M_tau/M_sigma at zeta-frequencies

Author: Noetic Eidos Project
License: MIT
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json


# Riemann zeta-zeros (same as spectral analyzer)
ZETA_ZEROS_TAU = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
])


@dataclass
class GeometryAnalysisResult:
    """Results from polylipse geometry analysis."""
    # Focal angle analysis
    focal_angle_spacings: np.ndarray  # Delta theta across levels
    angle_spacing_mean: float
    angle_spacing_std: float

    # Focal weight analysis
    focal_weight_entropy: np.ndarray  # Entropy of weight distribution per level
    weight_concentration: np.ndarray  # Gini coefficient per level

    # Complexity evolution
    n_foci_trajectory: np.ndarray
    complexity_jumps: List[int]  # Levels where n_foci increases
    jump_kappa_values: List[float]  # kappa_zeta at jumps

    # M_tau/M_sigma analysis
    m_ratio: np.ndarray
    m_ratio_peaks: np.ndarray
    m_ratio_matching_score: float

    # Statistical tests
    angle_spacing_ks_statistic: float
    angle_spacing_ks_pvalue: float


class ZetaGeometryAnalyzer:
    """
    Analyzes adaptive polylipse geometry for Riemann zeta-zero signatures.
    """

    def __init__(
        self,
        data_path: str = 'results/zeta_convergence/extracted_data.h5',
        output_dir: str = 'results/zeta_convergence/geometry',
        verbose: bool = True
    ):
        """
        Initialize geometry analyzer.

        Args:
            data_path: Path to extracted HDF5 data
            output_dir: Where to save plots and results
            verbose: Print progress
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Load data
        self.global_data = None
        self.geometry_data = None
        self._load_data()

    def _load_data(self):
        """Load global and geometry data from HDF5."""
        if self.verbose:
            print(f"Loading data from {self.data_path}...")

        with h5py.File(self.data_path, 'r') as f:
            # Global data
            self.global_data = {
                'levels': f['global']['levels'][:],
                'stabilized_kappa': f['global']['stabilized_kappa'][:],
                'n_foci': f['global']['n_foci'][:],
                'M_tau': f['global']['M_tau'][:],
                'M_sigma': f['global']['M_sigma'][:]
            }

            # Geometry data (load all levels)
            self.geometry_data = {}
            for level_key in f['geometry'].keys():
                level = int(level_key.split('_')[1])
                self.geometry_data[level] = {
                    'focal_angles': f['geometry'][level_key]['focal_angles'][:],
                    'n_foci': f['geometry'][level_key].attrs.get('n_foci', 0)
                }
                if 'focal_weights' in f['geometry'][level_key]:
                    self.geometry_data[level]['focal_weights'] = f['geometry'][level_key]['focal_weights'][:]
                if 'focal_centers' in f['geometry'][level_key]:
                    self.geometry_data[level]['focal_centers'] = f['geometry'][level_key]['focal_centers'][:]

        if self.verbose:
            print(f"  Loaded {len(self.global_data['levels'])} levels")
            print(f"  Geometry data for {len(self.geometry_data)} levels")

    def analyze_focal_angle_spacing(self) -> Dict:
        """
        Analyze focal angle spacing evolution and compare to zeta-zero gaps.

        Returns:
            Dictionary with spacing statistics
        """
        if self.verbose:
            print("\n[1] Analyzing focal angle spacing...")

        all_spacings = []
        spacings_per_level = []
        geometry_levels = []  # Track which levels have geometry data

        for level in sorted(self.geometry_data.keys()):
            angles = self.geometry_data[level]['focal_angles']
            geometry_levels.append(level)

            if len(angles) > 1:
                # Sort angles
                angles_sorted = np.sort(angles)

                # Compute spacing (Delta theta)
                spacings = np.diff(angles_sorted)

                # Also include wrap-around spacing
                if len(angles_sorted) > 2:
                    wrap_spacing = (2*np.pi) - (angles_sorted[-1] - angles_sorted[0])
                    spacings = np.append(spacings, wrap_spacing)

                all_spacings.extend(spacings)
                spacings_per_level.append(np.mean(spacings))
            else:
                spacings_per_level.append(np.nan)

        all_spacings = np.array(all_spacings)
        spacings_per_level = np.array(spacings_per_level)
        geometry_levels = np.array(geometry_levels)

        # Statistics
        mean_spacing = np.nanmean(spacings_per_level)
        std_spacing = np.nanstd(spacings_per_level)

        # Compare to zeta-zero gap distribution
        # Normalize both to same scale for comparison
        zeta_gaps = np.diff(ZETA_ZEROS_TAU)

        # Normalize to [0, 1]
        if len(all_spacings) > 0 and len(zeta_gaps) > 0:
            angles_norm = (all_spacings - np.min(all_spacings)) / (np.ptp(all_spacings) + 1e-10)
            zeta_norm = (zeta_gaps - np.min(zeta_gaps)) / (np.ptp(zeta_gaps) + 1e-10)

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(angles_norm, zeta_norm)
        else:
            ks_stat, ks_pval = np.nan, np.nan

        if self.verbose:
            print(f"  Mean angular spacing: {mean_spacing:.4f} rad")
            print(f"  Std angular spacing: {std_spacing:.4f} rad")
            print(f"  KS test vs zeta-gaps: stat={ks_stat:.4f}, p={ks_pval:.4f}")

        return {
            'all_spacings': all_spacings,
            'spacings_per_level': spacings_per_level,
            'geometry_levels': geometry_levels,
            'mean': mean_spacing,
            'std': std_spacing,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval
        }

    def analyze_focal_weights(self) -> Dict:
        """
        Analyze focal weight distributions and their evolution.

        Returns:
            Dictionary with weight statistics
        """
        if self.verbose:
            print("\n[2] Analyzing focal weight distributions...")

        weight_entropy = []
        weight_gini = []
        weight_levels = []

        for level in sorted(self.geometry_data.keys()):
            weight_levels.append(level)

            if 'focal_weights' in self.geometry_data[level]:
                weights = self.geometry_data[level]['focal_weights']

                # Entropy (higher = more uniform)
                weights_norm = weights / (np.sum(weights) + 1e-10)
                entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-10))
                weight_entropy.append(entropy)

                # Gini coefficient (higher = more concentrated)
                weights_sorted = np.sort(weights)
                n = len(weights)
                index = np.arange(1, n+1)
                gini = (2 * np.sum(index * weights_sorted)) / (n * np.sum(weights_sorted)) - (n + 1) / n
                weight_gini.append(gini)
            else:
                weight_entropy.append(np.nan)
                weight_gini.append(np.nan)

        weight_entropy = np.array(weight_entropy)
        weight_gini = np.array(weight_gini)
        weight_levels = np.array(weight_levels)

        if self.verbose:
            print(f"  Mean weight entropy: {np.nanmean(weight_entropy):.4f}")
            print(f"  Mean Gini coefficient: {np.nanmean(weight_gini):.4f}")

        return {
            'entropy': weight_entropy,
            'gini': weight_gini,
            'weight_levels': weight_levels
        }

    def analyze_complexity_evolution(self) -> Dict:
        """
        Analyze n_foci progression and identify complexity jumps.

        Returns:
            Dictionary with complexity analysis
        """
        if self.verbose:
            print("\n[3] Analyzing complexity evolution (n_foci)...")

        levels = self.global_data['levels']
        n_foci = self.global_data['n_foci']
        kappa = self.global_data['stabilized_kappa']

        # Find jumps (where n_foci increases)
        jumps = []
        jump_kappas = []

        for i in range(1, len(n_foci)):
            if n_foci[i] > n_foci[i-1]:
                jumps.append(levels[i])
                jump_kappas.append(kappa[i])

        if self.verbose:
            print(f"  n_foci range: {np.min(n_foci)} to {np.max(n_foci)}")
            print(f"  Number of complexity jumps: {len(jumps)}")
            if len(jumps) > 0:
                print(f"  Jump kappa values: {np.min(jump_kappas):.4f} to {np.max(jump_kappas):.4f}")

        return {
            'n_foci': n_foci,
            'jumps': jumps,
            'jump_kappas': jump_kappas
        }

    def analyze_m_ratio_resonances(self) -> Dict:
        """
        Analyze M_tau/M_sigma ratio for resonances at zeta-frequencies.

        Returns:
            Dictionary with M ratio analysis
        """
        if self.verbose:
            print("\n[4] Analyzing M_tau/M_sigma resonances...")

        M_tau = self.global_data['M_tau']
        M_sigma = self.global_data['M_sigma']

        # Compute ratio
        m_ratio = M_tau / (M_sigma + 1e-10)

        # Remove NaN/Inf
        valid_mask = np.isfinite(m_ratio)
        m_ratio_clean = m_ratio[valid_mask]

        if len(m_ratio_clean) > 10:
            # FFT to look for resonances
            freqs, power = self._compute_simple_fft(m_ratio_clean)

            # Detect peaks
            peak_indices, _ = signal.find_peaks(power, prominence=0.01*np.max(power))
            peak_freqs = freqs[peak_indices]

            # Match to zeta-zeros (very lenient tolerance since scale is different)
            matches = []
            for pf in peak_freqs:
                diffs = np.abs(ZETA_ZEROS_TAU - pf)
                if np.min(diffs) < 1.0:  # 1 Hz tolerance
                    matches.append(pf)

            matching_score = len(matches) / len(peak_freqs) if len(peak_freqs) > 0 else 0.0
        else:
            peak_freqs = np.array([])
            matching_score = 0.0

        if self.verbose:
            print(f"  M_tau/M_sigma range: {np.nanmin(m_ratio):.4f} to {np.nanmax(m_ratio):.4f}")
            print(f"  Detected {len(peak_freqs)} peaks in FFT")
            print(f"  Zeta-zero matching score: {matching_score:.1%}")

        return {
            'm_ratio': m_ratio,
            'peak_frequencies': peak_freqs,
            'matching_score': matching_score
        }

    def _compute_simple_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple FFT helper."""
        signal_clean = signal - np.mean(signal)  # Remove DC
        n = len(signal_clean)
        fft_vals = fft(signal_clean)
        freqs = fftfreq(n, d=1.0)

        # Positive frequencies only
        mask = freqs >= 0
        freqs = freqs[mask]
        power = np.abs(fft_vals[mask])**2
        power = power / np.sum(power)

        return freqs, power

    def plot_geometry_evolution(self, results: Dict):
        """
        Plot polylipse geometry evolution.

        Args:
            results: Dictionary with all analysis results
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        levels = self.global_data['levels']
        kappa = self.global_data['stabilized_kappa']

        # Plot 1: n_foci evolution
        ax = axes[0, 0]
        ax.plot(levels, results['complexity']['n_foci'], linewidth=1.5)
        ax.scatter(results['complexity']['jumps'],
                   [results['complexity']['n_foci'][np.where(levels == j)[0][0]]
                    for j in results['complexity']['jumps']],
                   color='red', s=50, zorder=5, label='Complexity jumps')
        ax.set_xlabel("Curriculum level")
        ax.set_ylabel("n_foci (complexity)")
        ax.set_title(f"Polylipse Complexity Evolution ({len(results['complexity']['jumps'])} jumps)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: kappa_zeta vs n_foci
        ax = axes[0, 1]
        ax.scatter(kappa, results['complexity']['n_foci'], alpha=0.5, s=10)
        ax.scatter(results['complexity']['jump_kappas'],
                   [results['complexity']['n_foci'][np.where(levels == j)[0][0]]
                    for j in results['complexity']['jumps']],
                   color='red', s=50, zorder=5, label='Jumps')
        ax.set_xlabel("kappa_zeta")
        ax.set_ylabel("n_foci")
        ax.set_title("Complexity vs kappa_zeta")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Focal angle spacing
        ax = axes[1, 0]
        geom_levels = results['angle_spacing']['geometry_levels']
        spacings = results['angle_spacing']['spacings_per_level']
        valid_mask = ~np.isnan(spacings)
        ax.plot(geom_levels[valid_mask], spacings[valid_mask],
                linewidth=1, alpha=0.7)
        ax.set_xlabel("Curriculum level")
        ax.set_ylabel("Mean angular spacing (rad)")
        ax.set_title("Focal Angle Spacing Evolution")
        ax.grid(True, alpha=0.3)

        # Plot 4: Angle spacing distribution vs zeta-gaps
        ax = axes[1, 1]
        if len(results['angle_spacing']['all_spacings']) > 0:
            ax.hist(results['angle_spacing']['all_spacings'], bins=50, alpha=0.5,
                   label='Focal spacings', density=True)
            zeta_gaps = np.diff(ZETA_ZEROS_TAU)
            # Scale zeta-gaps to similar range
            zeta_gaps_scaled = zeta_gaps * (np.mean(results['angle_spacing']['all_spacings']) / np.mean(zeta_gaps))
            ax.hist(zeta_gaps_scaled, bins=20, alpha=0.5,
                   label='Zeta-zero gaps (scaled)', density=True)
            ax.set_xlabel("Spacing")
            ax.set_ylabel("Density")
            ax.set_title(f"Spacing Distributions (KS p={results['angle_spacing']['ks_pvalue']:.3f})")
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: M_tau/M_sigma ratio
        ax = axes[2, 0]
        valid_mask = np.isfinite(results['m_ratio']['m_ratio'])
        ax.plot(levels[valid_mask], results['m_ratio']['m_ratio'][valid_mask],
                linewidth=1, alpha=0.7)
        ax.set_xlabel("Curriculum level")
        ax.set_ylabel("M_tau / M_sigma")
        ax.set_title("Moment Ratio Evolution")
        ax.grid(True, alpha=0.3)

        # Plot 6: Weight entropy evolution
        ax = axes[2, 1]
        if len(results['weights']['entropy']) > 0:
            weight_levels = results['weights']['weight_levels']
            entropy = results['weights']['entropy']
            valid_mask = ~np.isnan(entropy)
            ax.plot(weight_levels[valid_mask], entropy[valid_mask],
                    linewidth=1, alpha=0.7, label='Entropy')
            ax.set_xlabel("Curriculum level")
            ax.set_ylabel("Weight distribution entropy")
            ax.set_title("Focal Weight Uniformity")
            ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / 'geometry_evolution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if self.verbose:
            print(f"  Plot saved: {save_path}")
        plt.close()

    def run_full_analysis(self) -> Dict:
        """
        Run complete geometry analysis pipeline.

        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("Riemann Zeta-Zero Convergence: Polylipse Geometry Analysis")
        print("="*80)

        results = {}

        # 1. Focal angle spacing
        results['angle_spacing'] = self.analyze_focal_angle_spacing()

        # 2. Focal weights
        results['weights'] = self.analyze_focal_weights()

        # 3. Complexity evolution
        results['complexity'] = self.analyze_complexity_evolution()

        # 4. M_tau/M_sigma resonances
        results['m_ratio'] = self.analyze_m_ratio_resonances()

        # Generate plots
        self.plot_geometry_evolution(results)

        # Save summary
        summary = {
            'angle_spacing': {
                'mean': float(results['angle_spacing']['mean']),
                'std': float(results['angle_spacing']['std']),
                'ks_statistic': float(results['angle_spacing']['ks_statistic']),
                'ks_pvalue': float(results['angle_spacing']['ks_pvalue'])
            },
            'complexity': {
                'n_foci_min': int(np.min(results['complexity']['n_foci'])),
                'n_foci_max': int(np.max(results['complexity']['n_foci'])),
                'n_jumps': len(results['complexity']['jumps'])
            },
            'm_ratio': {
                'matching_score': float(results['m_ratio']['matching_score']),
                'n_peaks': len(results['m_ratio']['peak_frequencies'])
            }
        }

        summary_path = self.output_dir / 'geometry_analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("Geometry Analysis Complete")
        print("="*80)
        print(f"\nKey findings:")
        print(f"  Complexity range: {summary['complexity']['n_foci_min']} to {summary['complexity']['n_foci_max']} foci")
        print(f"  Complexity jumps: {summary['complexity']['n_jumps']}")
        print(f"  Angular spacing KS p-value: {summary['angle_spacing']['ks_pvalue']:.4f}")
        print(f"  M-ratio matching score: {summary['m_ratio']['matching_score']:.1%}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*80)

        return results


if __name__ == "__main__":
    # Run geometry analysis
    analyzer = ZetaGeometryAnalyzer(
        data_path='results/zeta_convergence/extracted_data.h5',
        output_dir='results/zeta_convergence/geometry',
        verbose=True
    )

    results = analyzer.run_full_analysis()

    print("\nNext steps:")
    print("  1. Run zeta_convergence_tests.py for statistical validation")
    print("  2. Run zeta_convergence_report.py for comprehensive report")

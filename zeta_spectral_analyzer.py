"""
Riemann Zeta-Zero Convergence Analysis: Spectral Analyzer
==========================================================

Performs FFT analysis on kappa_zeta time series and compares observed
frequency peaks to Riemann zeta-function zero imaginary parts (tau_k).

Tests:
1. Global trajectory: FFT of stabilized_kappa across curriculum levels
2. Fine-grained dynamics: FFT of epoch_kappa_raw within levels
3. Offset dynamics: FFT of epoch_offset (dual-kernel m* parameter)

Riemann zeta zeros: rho_k = 1/2 + i*tau_k where zeta(rho_k) = 0
First few tau_k values: [14.13, 21.02, 25.01, 30.42, 32.93, ...]

Author: Noetic Eidos Project
License: MIT
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json


# First 100 Riemann zeta-zero imaginary parts (tau_k)
# From: https://www.lmfdb.org/zeros/zeta/ and Odlyzko's tables
ZETA_ZEROS_TAU = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736209, 141.123707, 143.111846,
    146.000982, 147.422765, 150.053183, 150.925257, 153.024693,
    156.112909, 157.597591, 158.849988, 161.188964, 163.030709,
    165.537069, 167.184440, 169.094515, 169.911976, 173.411536,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026656,
    193.079726, 195.265396, 196.876481, 198.015330, 201.264731,
    202.493594, 204.189671, 205.394697, 207.906258, 209.576509,
    211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714918, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250188, 231.987235, 233.693404, 236.524229
])


@dataclass
class SpectralAnalysisResult:
    """Results from spectral analysis of a time series."""
    # Input data
    signal_name: str
    time_series: np.ndarray
    sampling_rate: float

    # FFT results
    frequencies: np.ndarray
    power_spectrum: np.ndarray

    # Peak detection
    peak_frequencies: np.ndarray
    peak_powers: np.ndarray
    peak_indices: np.ndarray

    # Zeta-zero matching
    matched_zeta_zeros: List[float]
    matching_distances: List[float]
    matching_score: float  # Fraction of top peaks that match zeta-zeros

    # Statistics
    dominant_frequency: float
    spectral_entropy: float
    total_power: float


class ZetaSpectralAnalyzer:
    """
    Performs spectral analysis on extracted curriculum data to detect
    Riemann zeta-zero frequency signatures.
    """

    def __init__(
        self,
        data_path: str = 'results/zeta_convergence/extracted_data.h5',
        output_dir: str = 'results/zeta_convergence/spectral',
        zeta_zeros: Optional[np.ndarray] = None,
        matching_tolerance: float = 0.5,  # Hz tolerance for frequency matching
        min_peak_prominence: float = 0.1,  # Relative to max power
        verbose: bool = True
    ):
        """
        Initialize spectral analyzer.

        Args:
            data_path: Path to extracted HDF5 data
            output_dir: Where to save plots and results
            zeta_zeros: Array of zeta-zero tau values (default: use ZETA_ZEROS_TAU)
            matching_tolerance: Tolerance for matching frequencies to zeta-zeros
            min_peak_prominence: Minimum prominence for peak detection (0-1)
            verbose: Print progress
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.zeta_zeros = zeta_zeros if zeta_zeros is not None else ZETA_ZEROS_TAU
        self.matching_tolerance = matching_tolerance
        self.min_peak_prominence = min_peak_prominence
        self.verbose = verbose

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Load extracted data from HDF5."""
        if self.verbose:
            print(f"Loading data from {self.data_path}...")

        result = {'global': {}, 'fine_grained': {}, 'metadata': {}}

        with h5py.File(self.data_path, 'r') as f:
            # Global data
            for key in f['global'].keys():
                result['global'][key] = f['global'][key][:]

            # Fine-grained (just store level list, load on demand)
            result['fine_grained_levels'] = list(f['fine_grained'].keys())

            # Metadata
            for key in f['metadata'].attrs.keys():
                result['metadata'][key] = f['metadata'].attrs[key]

        if self.verbose:
            n_levels = len(result['global']['levels'])
            n_fine = len(result['fine_grained_levels'])
            print(f"  Loaded {n_levels} levels")
            print(f"  Fine-grained data available for {n_fine} levels")

        return result

    def compute_fft(
        self,
        time_series: np.ndarray,
        sampling_rate: float = 1.0,
        detrend: bool = True,
        window: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of time series.

        Args:
            time_series: Input signal
            sampling_rate: Sampling rate (Hz)
            detrend: Remove linear trend
            window: Window function ('hann', 'hamming', 'blackman', None)

        Returns:
            frequencies: Frequency bins (Hz)
            power_spectrum: Power spectral density
        """
        # Preprocessing
        signal_proc = np.copy(time_series)

        if detrend:
            signal_proc = signal.detrend(signal_proc)

        if window is not None:
            if window == 'hann':
                win = signal.windows.hann(len(signal_proc))
            elif window == 'hamming':
                win = signal.windows.hamming(len(signal_proc))
            elif window == 'blackman':
                win = signal.windows.blackman(len(signal_proc))
            else:
                win = np.ones(len(signal_proc))
            signal_proc = signal_proc * win

        # FFT
        n = len(signal_proc)
        fft_vals = fft(signal_proc)
        freqs = fftfreq(n, d=1.0/sampling_rate)

        # Power spectrum (one-sided, positive frequencies only)
        power = np.abs(fft_vals)**2

        # Use only positive frequencies
        mask = freqs >= 0
        freqs = freqs[mask]
        power = power[mask]

        # Normalize
        power = power / np.sum(power)

        return freqs, power

    def detect_peaks(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        n_peaks: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect peaks in power spectrum.

        Args:
            frequencies: Frequency bins
            power_spectrum: Power values
            n_peaks: Number of top peaks to return

        Returns:
            peak_frequencies: Frequencies of detected peaks
            peak_powers: Powers of detected peaks
            peak_indices: Indices of peaks in original arrays
        """
        # Find peaks with prominence threshold
        prominence_threshold = self.min_peak_prominence * np.max(power_spectrum)
        peak_indices, properties = signal.find_peaks(
            power_spectrum,
            prominence=prominence_threshold,
            distance=2  # Minimum separation between peaks
        )

        if len(peak_indices) == 0:
            return np.array([]), np.array([]), np.array([])

        # Sort by power and take top n_peaks
        peak_powers = power_spectrum[peak_indices]
        sorted_idx = np.argsort(peak_powers)[::-1][:n_peaks]

        peak_indices = peak_indices[sorted_idx]
        peak_frequencies = frequencies[peak_indices]
        peak_powers = power_spectrum[peak_indices]

        return peak_frequencies, peak_powers, peak_indices

    def match_to_zeta_zeros(
        self,
        peak_frequencies: np.ndarray
    ) -> Tuple[List[float], List[float], float]:
        """
        Match detected peaks to Riemann zeta-zeros.

        Args:
            peak_frequencies: Detected frequency peaks

        Returns:
            matched_zeros: List of matched zeta-zero values
            distances: Distance from peak to matched zero
            score: Matching score (fraction of peaks matched)
        """
        matched_zeros = []
        distances = []

        for peak_freq in peak_frequencies:
            # Find closest zeta-zero
            diffs = np.abs(self.zeta_zeros - peak_freq)
            min_idx = np.argmin(diffs)
            min_dist = diffs[min_idx]

            if min_dist <= self.matching_tolerance:
                matched_zeros.append(self.zeta_zeros[min_idx])
                distances.append(min_dist)
            else:
                matched_zeros.append(np.nan)
                distances.append(np.nan)

        # Compute matching score
        n_matched = np.sum(~np.isnan(matched_zeros))
        score = n_matched / len(peak_frequencies) if len(peak_frequencies) > 0 else 0.0

        return matched_zeros, distances, score

    def analyze_time_series(
        self,
        time_series: np.ndarray,
        sampling_rate: float,
        signal_name: str
    ) -> SpectralAnalysisResult:
        """
        Complete spectral analysis of a time series.

        Args:
            time_series: Input signal
            sampling_rate: Sampling rate
            signal_name: Name for identification

        Returns:
            SpectralAnalysisResult object
        """
        # FFT
        frequencies, power_spectrum = self.compute_fft(time_series, sampling_rate)

        # Detect peaks
        peak_frequencies, peak_powers, peak_indices = self.detect_peaks(
            frequencies, power_spectrum
        )

        # Match to zeta-zeros
        matched_zeros, distances, score = self.match_to_zeta_zeros(peak_frequencies)

        # Statistics
        dominant_freq = frequencies[np.argmax(power_spectrum)]
        spectral_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        total_power = np.sum(power_spectrum)

        return SpectralAnalysisResult(
            signal_name=signal_name,
            time_series=time_series,
            sampling_rate=sampling_rate,
            frequencies=frequencies,
            power_spectrum=power_spectrum,
            peak_frequencies=peak_frequencies,
            peak_powers=peak_powers,
            peak_indices=peak_indices,
            matched_zeta_zeros=matched_zeros,
            matching_distances=distances,
            matching_score=score,
            dominant_frequency=dominant_freq,
            spectral_entropy=spectral_entropy,
            total_power=total_power
        )

    def analyze_global_trajectory(self) -> SpectralAnalysisResult:
        """
        Analyze global kappa_zeta trajectory across curriculum levels.
        """
        if self.verbose:
            print("\n[1] Analyzing global kappa_zeta trajectory...")

        levels = self.data['global']['levels']
        kappa_zeta = self.data['global']['stabilized_kappa']

        # Sampling rate: 1 Hz (1 sample per level)
        sampling_rate = 1.0

        result = self.analyze_time_series(
            kappa_zeta,
            sampling_rate,
            "Global kappa_zeta trajectory"
        )

        if self.verbose:
            print(f"  Dominant frequency: {result.dominant_frequency:.4f} Hz")
            print(f"  Zeta-zero matching score: {result.matching_score:.2%}")
            print(f"  Matched {np.sum(~np.isnan(result.matched_zeta_zeros))}/{len(result.peak_frequencies)} peaks")

        return result

    def analyze_fine_grained_ensemble(
        self,
        n_levels_sample: int = 100
    ) -> SpectralAnalysisResult:
        """
        Analyze fine-grained dynamics by concatenating epoch_kappa_raw
        from multiple levels.

        Args:
            n_levels_sample: Number of levels to sample
        """
        if self.verbose:
            print("\n[2] Analyzing fine-grained epoch dynamics (ensemble)...")

        # Sample levels evenly
        fine_levels = self.data['fine_grained_levels']
        n_available = len(fine_levels)
        sample_indices = np.linspace(0, n_available-1, min(n_levels_sample, n_available), dtype=int)

        # Concatenate time series
        concatenated = []
        with h5py.File(self.data_path, 'r') as f:
            for idx in sample_indices:
                level_key = fine_levels[idx]
                epoch_kappa = f['fine_grained'][level_key]['epoch_kappa_raw'][:]
                # Flatten if 2D (epochs × batches)
                if epoch_kappa.ndim > 1:
                    epoch_kappa = epoch_kappa.flatten()
                concatenated.append(epoch_kappa)

        time_series = np.concatenate(concatenated)

        # Sampling rate: assuming 1 Hz (1 sample per epoch)
        sampling_rate = 1.0

        result = self.analyze_time_series(
            time_series,
            sampling_rate,
            f"Fine-grained dynamics (ensemble of {len(sample_indices)} levels)"
        )

        if self.verbose:
            print(f"  Total samples: {len(time_series)}")
            print(f"  Dominant frequency: {result.dominant_frequency:.4f} Hz")
            print(f"  Zeta-zero matching score: {result.matching_score:.2%}")

        return result

    def plot_spectral_analysis(
        self,
        result: SpectralAnalysisResult,
        show_zeta_zeros: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot spectral analysis results.

        Args:
            result: Analysis result to plot
            show_zeta_zeros: Show zeta-zero reference lines
            save_path: Where to save (None = display only)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Time series
        ax = axes[0]
        ax.plot(result.time_series, linewidth=0.5, alpha=0.7)
        ax.set_title(f"{result.signal_name}\nSampling rate: {result.sampling_rate} Hz", fontsize=12)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("kappa_zeta")
        ax.grid(True, alpha=0.3)

        # Plot 2: Power spectrum
        ax = axes[1]
        ax.semilogy(result.frequencies, result.power_spectrum, linewidth=1, alpha=0.7, label='Power spectrum')

        # Mark detected peaks
        ax.plot(
            result.peak_frequencies,
            result.peak_powers,
            'ro',
            markersize=8,
            label=f'Detected peaks (n={len(result.peak_frequencies)})'
        )

        # Show zeta-zeros
        if show_zeta_zeros:
            zeta_in_range = self.zeta_zeros[self.zeta_zeros <= np.max(result.frequencies)]
            if len(zeta_in_range) > 0:
                for tau in zeta_in_range[:30]:  # Show first 30
                    ax.axvline(tau, color='green', alpha=0.2, linewidth=1, linestyle='--')
                ax.axvline(zeta_in_range[0], color='green', alpha=0.5, linewidth=1.5,
                          linestyle='--', label='Zeta-zero frequencies')

        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Power (normalized)", fontsize=11)
        ax.set_title(f"Power Spectrum - Matching Score: {result.matching_score:.1%}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Limit x-axis to reasonable range
        ax.set_xlim(0, min(200, np.max(result.frequencies)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"  Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def run_full_analysis(self):
        """
        Run complete spectral analysis pipeline.
        """
        print("="*80)
        print("Riemann Zeta-Zero Convergence: Spectral Analysis")
        print("="*80)

        results = {}

        # 1. Global trajectory
        results['global'] = self.analyze_global_trajectory()
        self.plot_spectral_analysis(
            results['global'],
            save_path=self.output_dir / 'spectral_global_trajectory.png'
        )

        # 2. Fine-grained ensemble
        results['fine_ensemble'] = self.analyze_fine_grained_ensemble(n_levels_sample=100)
        self.plot_spectral_analysis(
            results['fine_ensemble'],
            save_path=self.output_dir / 'spectral_fine_ensemble.png'
        )

        # Save summary
        summary = {
            'global_trajectory': {
                'matching_score': float(results['global'].matching_score),
                'dominant_frequency': float(results['global'].dominant_frequency),
                'n_peaks_matched': int(np.sum(~np.isnan(results['global'].matched_zeta_zeros))),
                'n_peaks_total': len(results['global'].peak_frequencies)
            },
            'fine_ensemble': {
                'matching_score': float(results['fine_ensemble'].matching_score),
                'dominant_frequency': float(results['fine_ensemble'].dominant_frequency),
                'n_peaks_matched': int(np.sum(~np.isnan(results['fine_ensemble'].matched_zeta_zeros))),
                'n_peaks_total': len(results['fine_ensemble'].peak_frequencies)
            }
        }

        summary_path = self.output_dir / 'spectral_analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("Spectral Analysis Complete")
        print("="*80)
        print(f"\nSummary:")
        print(f"  Global trajectory matching score: {results['global'].matching_score:.1%}")
        print(f"  Fine-grained ensemble matching score: {results['fine_ensemble'].matching_score:.1%}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*80)

        return results


if __name__ == "__main__":
    # Run spectral analysis
    analyzer = ZetaSpectralAnalyzer(
        data_path='results/zeta_convergence/extracted_data.h5',
        output_dir='results/zeta_convergence/spectral',
        matching_tolerance=0.5,
        min_peak_prominence=0.01,  # Lower threshold to detect weaker peaks
        verbose=True
    )

    results = analyzer.run_full_analysis()

    print("\nNext steps:")
    print("  1. Run zeta_geometry_analyzer.py for polylipse evolution analysis")
    print("  2. Run zeta_convergence_tests.py for statistical validation")

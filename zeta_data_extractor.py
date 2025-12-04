"""
Riemann Zeta-Zero Convergence Analysis: Data Extractor
=======================================================

Extracts kappa_zeta time series, polylipse geometry evolution, and training metrics
from 1378 pre-trained curriculum checkpoints.

Output: HDF5 file (~500MB) containing:
  - Global kappa_zeta trajectory across all curriculum levels
  - Fine-grained kappa_zeta oscillations within each level
  - Polylipse geometry evolution (focal angles, weights, centers, n_foci)
  - Dual-kernel parameters (beta, t)
  - Training metrics (losses, convergence rates)

Author: Noetic Eidos Project
License: MIT
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json


@dataclass
class CheckpointData:
    """Extracted data from a single checkpoint."""
    level: int

    # Global kappa_zeta value
    stabilized_kappa: float

    # Fine-grained time series (if available)
    epoch_kappa_raw: Optional[np.ndarray] = None
    epoch_kappa: Optional[np.ndarray] = None
    epoch_offset: Optional[np.ndarray] = None
    epoch_beta: Optional[np.ndarray] = None
    epoch_t: Optional[np.ndarray] = None

    # Polylipse geometry
    n_foci: Optional[int] = None
    focal_angles: Optional[np.ndarray] = None
    focal_weights: Optional[np.ndarray] = None
    focal_centers: Optional[np.ndarray] = None

    # Additional statistics
    M_tau: Optional[float] = None
    M_sigma: Optional[float] = None
    kappa_actual: Optional[float] = None
    observed_kappa: Optional[float] = None

    # Training metrics
    stability_variance: Optional[float] = None
    convergence_rate: Optional[float] = None

    # Loss curves
    epoch_loss: Optional[np.ndarray] = None
    epoch_task_loss: Optional[np.ndarray] = None
    epoch_zero_loss: Optional[np.ndarray] = None


class ZetaDataExtractor:
    """
    Extracts all relevant data from curriculum checkpoints for zeta-zero convergence analysis.

    Handles both early and later checkpoint formats robustly.
    """

    def __init__(
        self,
        checkpoint_dir: str = 'polylipse_curriculum_results',
        output_path: str = 'results/zeta_convergence/extracted_data.h5',
        verbose: bool = True
    ):
        """
        Initialize extractor.

        Args:
            checkpoint_dir: Directory containing level_*_checkpoint.pt files
            output_path: Where to save HDF5 output
            verbose: Print progress
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_path = Path(output_path)
        self.verbose = verbose

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.n_checkpoints_processed = 0
        self.n_checkpoints_failed = 0
        self.n_with_fine_grained = 0
        self.n_with_geometry = 0

    def extract_kappa_zeta(self, ckpt: Dict) -> Optional[float]:
        """
        Robustly extract kappa_zeta value from checkpoint (handles both formats).

        Args:
            ckpt: Loaded checkpoint dictionary

        Returns:
            kappa_zeta value or None if not found
        """
        # Method 1: Top-level (early checkpoints)
        if 'stabilized_kappa' in ckpt:
            return float(ckpt['stabilized_kappa'])

        # Method 2: In dataset_config (later checkpoints)
        if 'dataset_config' in ckpt:
            ds = ckpt['dataset_config']
            if 'stabilized_kappa' in ds:
                return float(ds['stabilized_kappa'])
            elif 'observed_kappa' in ds:
                return float(ds['observed_kappa'])
            elif 'kappa_actual' in ds:
                return float(ds['kappa_actual'])

        return None

    def extract_checkpoint(self, ckpt_path: Path) -> Optional[CheckpointData]:
        """
        Extract all data from a single checkpoint.

        Args:
            ckpt_path: Path to checkpoint file

        Returns:
            CheckpointData object or None if extraction fails
        """
        try:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location='cpu')

            # Extract level number from filename
            # Format: level_XXX_checkpoint.pt
            level_str = ckpt_path.stem.split('_')[1]
            level = int(level_str)

            # Extract global κζ
            stabilized_kappa = self.extract_kappa_zeta(ckpt)
            if stabilized_kappa is None:
                if self.verbose:
                    print(f"  Warning: Could not extract kappa_zeta from {ckpt_path.name}")
                return None

            # Initialize data object
            data = CheckpointData(
                level=level,
                stabilized_kappa=stabilized_kappa
            )

            # Extract fine-grained metrics (if available)
            if 'metrics' in ckpt:
                metrics = ckpt['metrics']

                # κζ time series
                if 'epoch_kappa_raw' in metrics:
                    data.epoch_kappa_raw = self._to_numpy(metrics['epoch_kappa_raw'])
                if 'epoch_kappa' in metrics:
                    data.epoch_kappa = self._to_numpy(metrics['epoch_kappa'])
                if 'epoch_offset' in metrics:
                    data.epoch_offset = self._to_numpy(metrics['epoch_offset'])

                # Dual-kernel parameters
                if 'epoch_beta' in metrics:
                    data.epoch_beta = self._to_numpy(metrics['epoch_beta'])
                if 'epoch_t' in metrics:
                    data.epoch_t = self._to_numpy(metrics['epoch_t'])

                # Loss curves
                if 'epoch_loss' in metrics:
                    data.epoch_loss = self._to_numpy(metrics['epoch_loss'])
                if 'epoch_task_loss' in metrics:
                    data.epoch_task_loss = self._to_numpy(metrics['epoch_task_loss'])
                if 'epoch_zero_loss' in metrics:
                    data.epoch_zero_loss = self._to_numpy(metrics['epoch_zero_loss'])

                if data.epoch_kappa_raw is not None:
                    self.n_with_fine_grained += 1

            # Extract polylipse geometry (CRITICAL - user emphasized!)
            dataset_config = ckpt.get('dataset_config', {})

            # n_foci (complexity measure)
            if 'n_foci' in dataset_config:
                data.n_foci = int(dataset_config['n_foci'])
            elif 'n_foci' in ckpt:
                data.n_foci = int(ckpt['n_foci'])

            # Focal configurations
            if 'focal_angles' in dataset_config:
                data.focal_angles = self._to_numpy(dataset_config['focal_angles'])
                self.n_with_geometry += 1
            if 'focal_weights' in dataset_config:
                data.focal_weights = self._to_numpy(dataset_config['focal_weights'])
            if 'focal_centers' in dataset_config:
                data.focal_centers = self._to_numpy(dataset_config['focal_centers'])

            # Additional statistics
            if 'M_tau' in dataset_config:
                data.M_tau = float(dataset_config['M_tau'])
            if 'M_sigma' in dataset_config:
                data.M_sigma = float(dataset_config['M_sigma'])
            if 'kappa_actual' in dataset_config:
                data.kappa_actual = float(dataset_config['kappa_actual'])
            if 'observed_kappa' in dataset_config:
                data.observed_kappa = float(dataset_config['observed_kappa'])

            # Curriculum info
            if 'curriculum_info' in ckpt:
                curr = ckpt['curriculum_info']
                if 'stability_variance' in curr:
                    data.stability_variance = float(curr['stability_variance'])
                if 'convergence_rate' in curr:
                    data.convergence_rate = float(curr['convergence_rate'])

            self.n_checkpoints_processed += 1
            return data

        except Exception as e:
            if self.verbose:
                print(f"  Error loading {ckpt_path.name}: {e}")
            self.n_checkpoints_failed += 1
            return None

    def _to_numpy(self, data) -> np.ndarray:
        """Convert various data types to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            # Try conversion
            return np.array(data)

    def extract_all(self) -> Dict[int, CheckpointData]:
        """
        Extract data from all checkpoints.

        Returns:
            Dictionary mapping level -> CheckpointData
        """
        # Find all checkpoint files
        checkpoint_files = sorted(
            self.checkpoint_dir.glob('level_*_checkpoint.pt'),
            key=lambda p: int(p.stem.split('_')[1])
        )

        if len(checkpoint_files) == 0:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        if self.verbose:
            print(f"Found {len(checkpoint_files)} checkpoints")
            print(f"Extracting data...")

        # Extract data
        extracted_data = {}

        iterator = tqdm(checkpoint_files) if self.verbose else checkpoint_files

        for ckpt_path in iterator:
            if self.verbose and hasattr(iterator, 'set_description'):
                iterator.set_description(f"Processing {ckpt_path.name}")

            data = self.extract_checkpoint(ckpt_path)
            if data is not None:
                extracted_data[data.level] = data

        if self.verbose:
            print(f"\nExtraction complete:")
            print(f"  Processed: {self.n_checkpoints_processed}/{len(checkpoint_files)}")
            print(f"  Failed: {self.n_checkpoints_failed}")
            print(f"  With fine-grained metrics: {self.n_with_fine_grained}")
            print(f"  With geometry data: {self.n_with_geometry}")

        return extracted_data

    def save_to_hdf5(self, data: Dict[int, CheckpointData]):
        """
        Save extracted data to HDF5 file.

        Structure:
          /global/
            - levels: [0, 1, 2, ..., 1377]
            - stabilized_kappa: κζ values
            - n_foci: complexity
            - M_tau, M_sigma, etc.
          /fine_grained/
            - level_XXX/
              - epoch_kappa_raw
              - epoch_kappa
              - epoch_offset
              - epoch_beta
              - epoch_t
              - epoch_loss
              - ...
          /geometry/
            - level_XXX/
              - focal_angles
              - focal_weights
              - focal_centers
          /metadata/
            - extraction_info

        Args:
            data: Dictionary of extracted checkpoint data
        """
        if self.verbose:
            print(f"\nSaving to {self.output_path}...")

        with h5py.File(self.output_path, 'w') as f:
            # Sort by level
            levels = sorted(data.keys())
            n_levels = len(levels)

            # Create groups
            global_grp = f.create_group('global')
            fine_grp = f.create_group('fine_grained')
            geom_grp = f.create_group('geometry')
            meta_grp = f.create_group('metadata')

            # Global trajectory
            global_grp.create_dataset('levels', data=np.array(levels))

            stabilized_kappa = np.array([data[lvl].stabilized_kappa for lvl in levels])
            global_grp.create_dataset('stabilized_kappa', data=stabilized_kappa)

            # Optional global fields (use NaN for missing)
            n_foci = np.array([
                data[lvl].n_foci if data[lvl].n_foci is not None else -1
                for lvl in levels
            ])
            global_grp.create_dataset('n_foci', data=n_foci)

            M_tau = np.array([
                data[lvl].M_tau if data[lvl].M_tau is not None else np.nan
                for lvl in levels
            ])
            global_grp.create_dataset('M_tau', data=M_tau)

            M_sigma = np.array([
                data[lvl].M_sigma if data[lvl].M_sigma is not None else np.nan
                for lvl in levels
            ])
            global_grp.create_dataset('M_sigma', data=M_sigma)

            stability_var = np.array([
                data[lvl].stability_variance if data[lvl].stability_variance is not None else np.nan
                for lvl in levels
            ])
            global_grp.create_dataset('stability_variance', data=stability_var)

            convergence = np.array([
                data[lvl].convergence_rate if data[lvl].convergence_rate is not None else np.nan
                for lvl in levels
            ])
            global_grp.create_dataset('convergence_rate', data=convergence)

            # Fine-grained time series
            n_fine_saved = 0
            for lvl in levels:
                d = data[lvl]
                if d.epoch_kappa_raw is not None:
                    lvl_grp = fine_grp.create_group(f'level_{lvl:04d}')

                    lvl_grp.create_dataset('epoch_kappa_raw', data=d.epoch_kappa_raw)
                    if d.epoch_kappa is not None:
                        lvl_grp.create_dataset('epoch_kappa', data=d.epoch_kappa)
                    if d.epoch_offset is not None:
                        lvl_grp.create_dataset('epoch_offset', data=d.epoch_offset)
                    if d.epoch_beta is not None:
                        lvl_grp.create_dataset('epoch_beta', data=d.epoch_beta)
                    if d.epoch_t is not None:
                        lvl_grp.create_dataset('epoch_t', data=d.epoch_t)
                    if d.epoch_loss is not None:
                        lvl_grp.create_dataset('epoch_loss', data=d.epoch_loss)
                    if d.epoch_task_loss is not None:
                        lvl_grp.create_dataset('epoch_task_loss', data=d.epoch_task_loss)
                    if d.epoch_zero_loss is not None:
                        lvl_grp.create_dataset('epoch_zero_loss', data=d.epoch_zero_loss)

                    n_fine_saved += 1

            # Polylipse geometry (CRITICAL!)
            n_geom_saved = 0
            for lvl in levels:
                d = data[lvl]
                if d.focal_angles is not None:
                    lvl_grp = geom_grp.create_group(f'level_{lvl:04d}')

                    lvl_grp.create_dataset('focal_angles', data=d.focal_angles)
                    if d.focal_weights is not None:
                        lvl_grp.create_dataset('focal_weights', data=d.focal_weights)
                    if d.focal_centers is not None:
                        lvl_grp.create_dataset('focal_centers', data=d.focal_centers)

                    # Store n_foci for this level
                    lvl_grp.attrs['n_foci'] = d.n_foci if d.n_foci is not None else 0

                    n_geom_saved += 1

            # Metadata
            meta_grp.attrs['n_checkpoints_total'] = n_levels
            meta_grp.attrs['n_with_fine_grained'] = n_fine_saved
            meta_grp.attrs['n_with_geometry'] = n_geom_saved
            meta_grp.attrs['checkpoint_dir'] = str(self.checkpoint_dir)
            meta_grp.attrs['extraction_date'] = str(np.datetime64('now'))

            # Compute statistics
            stats = {
                'kappa_mean': float(np.mean(stabilized_kappa)),
                'kappa_std': float(np.std(stabilized_kappa)),
                'kappa_min': float(np.min(stabilized_kappa)),
                'kappa_max': float(np.max(stabilized_kappa)),
                'kappa_range': float(np.ptp(stabilized_kappa)),
                'n_foci_mean': float(np.nanmean(n_foci[n_foci >= 0])),
                'n_foci_max': int(np.max(n_foci))
            }

            # Save as JSON string
            meta_grp.attrs['statistics'] = json.dumps(stats, indent=2)

        if self.verbose:
            print(f"  Saved {n_levels} levels")
            print(f"  Fine-grained data: {n_fine_saved} levels")
            print(f"  Geometry data: {n_geom_saved} levels")
            print(f"  File size: {self.output_path.stat().st_size / 1024 / 1024:.1f} MB")

    def run(self):
        """
        Complete extraction pipeline.
        """
        print("="*80)
        print("Riemann Zeta-Zero Convergence Analysis: Data Extraction")
        print("="*80)

        # Extract
        data = self.extract_all()

        # Save
        self.save_to_hdf5(data)

        print("\n" + "="*80)
        print("Extraction complete!")
        print(f"Output: {self.output_path}")
        print("="*80)

        return data


def load_extracted_data(h5_path: str = 'results/zeta_convergence/extracted_data.h5'):
    """
    Load extracted data from HDF5 file.

    Returns:
        Dictionary with keys: 'global', 'fine_grained', 'geometry', 'metadata'
    """
    result = {
        'global': {},
        'fine_grained': {},
        'geometry': {},
        'metadata': {}
    }

    with h5py.File(h5_path, 'r') as f:
        # Global data
        for key in f['global'].keys():
            result['global'][key] = f['global'][key][:]

        # Fine-grained (lazy loading - just store references)
        for lvl_key in f['fine_grained'].keys():
            level = int(lvl_key.split('_')[1])
            result['fine_grained'][level] = {}
            for metric_key in f['fine_grained'][lvl_key].keys():
                result['fine_grained'][level][metric_key] = f['fine_grained'][lvl_key][metric_key][:]

        # Geometry
        for lvl_key in f['geometry'].keys():
            level = int(lvl_key.split('_')[1])
            result['geometry'][level] = {}
            for geom_key in f['geometry'][lvl_key].keys():
                result['geometry'][level][geom_key] = f['geometry'][lvl_key][geom_key][:]
            # Get n_foci attribute
            result['geometry'][level]['n_foci'] = f['geometry'][lvl_key].attrs.get('n_foci', 0)

        # Metadata
        for key in f['metadata'].attrs.keys():
            result['metadata'][key] = f['metadata'].attrs[key]

    return result


if __name__ == "__main__":
    # Run extraction
    extractor = ZetaDataExtractor(
        checkpoint_dir='polylipse_curriculum_results',
        output_path='results/zeta_convergence/extracted_data.h5',
        verbose=True
    )

    data = extractor.run()

    # Quick preview
    print("\nQuick Preview:")
    print("-"*80)

    if data:
        levels = sorted(data.keys())
        print(f"Levels: {levels[0]} to {levels[-1]} ({len(levels)} total)")

        kappas = [data[lvl].stabilized_kappa for lvl in levels]
        print(f"kappa_zeta range: {min(kappas):.4f} to {max(kappas):.4f}")

        n_foci_list = [data[lvl].n_foci for lvl in levels if data[lvl].n_foci is not None]
        if n_foci_list:
            print(f"n_foci range: {min(n_foci_list)} to {max(n_foci_list)}")

        # Count fine-grained data points
        total_fine_points = sum(
            len(data[lvl].epoch_kappa_raw)
            for lvl in levels
            if data[lvl].epoch_kappa_raw is not None
        )
        print(f"Total fine-grained kappa_zeta measurements: {total_fine_points:,}")

    print("-"*80)
    print("\nNext steps:")
    print("  1. Run zeta_spectral_analyzer.py for FFT analysis")
    print("  2. Run zeta_geometry_analyzer.py for polylipse evolution")
    print("  3. Run zeta_convergence_tests.py for statistical validation")

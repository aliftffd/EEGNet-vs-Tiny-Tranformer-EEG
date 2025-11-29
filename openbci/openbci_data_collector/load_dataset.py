#!/usr/bin/env python3
"""
Motor Imagery Dataset Loader
Loads OpenBCI motor imagery data for deep learning
"""

import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class MotorImageryDataset(Dataset):
    """PyTorch Dataset for motor imagery EEG data"""

    def __init__(self, data_dir: str, subject_id: str = None, session_id: str = None):
        """
        Args:
            data_dir: Root directory containing motor_imagery_data
            subject_id: Specific subject (e.g., 'S01') or None for all
            session_id: Specific session (e.g., 'session_01') or None for all
        """
        self.data = []
        self.labels = []
        self.metadata = []

        # Build search pattern
        pattern = str(Path(data_dir))
        if subject_id:
            pattern = str(Path(pattern) / subject_id)
        else:
            pattern = str(Path(pattern) / "*")

        if session_id:
            pattern = str(Path(pattern) / session_id)
        else:
            pattern = str(Path(pattern) / "*")

        pattern = str(Path(pattern) / "*_class_*.csv")

        # Load all matching CSV files
        csv_files = glob.glob(pattern)
        print(f"Found {len(csv_files)} trial files")

        for csv_file in sorted(csv_files):
            df = pd.read_csv(csv_file)

            # Extract EEG channels (exclude timestamp, sample_id, class_id)
            channel_cols = [col for col in df.columns
                          if col not in ['timestamp', 'sample_id', 'class_id']]
            eeg_data = df[channel_cols].values

            # Get label
            label = df['class_id'].iloc[0]

            # Load metadata
            meta_file = csv_file.replace('.csv', '_metadata.json').replace(
                f"_class_{label}_", f"_class_{label}_"
            )
            # Try to find metadata file with same base name
            meta_pattern = csv_file.rsplit('_', 1)[0] + '_metadata.json'
            meta_files = glob.glob(meta_pattern)

            metadata = {}
            if meta_files:
                with open(meta_files[0], 'r') as f:
                    metadata = json.load(f)

            self.data.append(torch.FloatTensor(eeg_data))
            self.labels.append(int(label))
            self.metadata.append(metadata)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_class_distribution(self) -> Dict[int, int]:
        """Returns count of samples per class"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def load_data_numpy(data_dir: str, subject_id: str = None, session_id: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all data as numpy arrays

    Returns:
        X: shape (n_trials, n_samples, n_channels) - EEG data
        y: shape (n_trials,) - class labels
    """
    dataset = MotorImageryDataset(data_dir, subject_id, session_id)

    X = np.array([trial.numpy() for trial in dataset.data])
    y = np.array(dataset.labels)

    return X, y


def load_data_pandas(data_dir: str, subject_id: str = None) -> pd.DataFrame:
    """
    Load all data as a single pandas DataFrame

    Returns:
        DataFrame with columns: timestamp, sample_id, class_id, C3_left_motor, C4_right_motor, ...
    """
    pattern = str(Path(data_dir))
    if subject_id:
        pattern = str(Path(pattern) / subject_id / "*")
    else:
        pattern = str(Path(pattern) / "*" / "*")

    pattern = str(Path(pattern) / "*_class_*.csv")

    csv_files = glob.glob(pattern)
    print(f"Loading {len(csv_files)} files...")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Add trial identifier
        df['trial_file'] = Path(csv_file).name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(combined_df)}")

    return combined_df


def print_dataset_info(data_dir: str):
    """Print summary statistics about the dataset"""
    dataset = MotorImageryDataset(data_dir)

    print("=" * 50)
    print("Motor Imagery Dataset Info")
    print("=" * 50)
    print(f"Total trials: {len(dataset)}")
    print(f"Samples per trial: {dataset.data[0].shape[0]} (avg)")
    print(f"Channels: {dataset.data[0].shape[1]}")

    print("\nClass distribution:")
    class_names = {0: 'left_hand', 1: 'right_hand', 2: 'both_hands', 3: 'rest'}
    for class_id, count in sorted(dataset.get_class_distribution().items()):
        class_name = class_names.get(class_id, 'unknown')
        print(f"  Class {class_id} ({class_name}): {count} trials")

    if dataset.metadata:
        meta = dataset.metadata[0]
        print("\nRecording parameters:")
        print(f"  Sample rate: {meta.get('sample_rate', 'N/A')} Hz")
        print(f"  Duration: {meta.get('duration_seconds', 'N/A')} seconds")
        print(f"  Channels: {meta.get('electrode_config', {}).get('channels', 'N/A')}")

    print("=" * 50)


# Example usage
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "motor_imagery_data"

    print("Loading motor imagery dataset...\n")

    # Print dataset info
    print_dataset_info(data_dir)

    # Example 1: Load as PyTorch Dataset
    print("\n--- Example 1: PyTorch DataLoader ---")
    dataset = MotorImageryDataset(data_dir, subject_id="S01")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_idx, (data, labels) in enumerate(loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels: {labels}")
        if batch_idx >= 2:  # Show first 3 batches
            break

    # Example 2: Load as numpy arrays
    print("\n--- Example 2: NumPy Arrays ---")
    X, y = load_data_numpy(data_dir, subject_id="S01")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")

    # Example 3: Load as pandas DataFrame
    print("\n--- Example 3: Pandas DataFrame ---")
    df = load_data_pandas(data_dir, subject_id="S01")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Class distribution:\n{df['class_id'].value_counts().sort_index()}")

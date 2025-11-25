"""
BCI Competition IV 2a Dataset Loader
Handles EEG data loading, preprocessing, and epoching for Motor Imagery tasks
"""

import numpy as np
import mne
from scipy import signal
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os
import urllib.request
import zipfile


class BCICompetition2aDataset(Dataset):
    """
    PyTorch Dataset for BCI Competition IV 2a
    Motor Imagery Classification: Left Hand, Right Hand, Feet, Tongue
    """
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array of shape (n_trials, n_channels, n_timepoints)
            labels: numpy array of shape (n_trials,)
            transform: optional transform to apply
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label


class BCIDataProcessor:
    """
    Handles downloading, preprocessing, and epoching of BCI Competition IV 2a data
    """
    def __init__(self, 
                 data_path='./data',
                 subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 selected_channels=['EEG-C3', 'EEG-Cz', 'EEG-C4'],
                 freq_band=(8, 30),
                 epoch_duration=2.0,
                 epoch_offset=0.5,
                 sfreq=250,
                 model_type='eegnet'):
        """
        Args:
            data_path: where to store/load data
            subjects: list of subject IDs (1-9)
            selected_channels: EEG channels to use
            freq_band: (low, high) frequency band in Hz
            epoch_duration: length of epoch in seconds
            epoch_offset: offset after cue in seconds
            sfreq: sampling frequency (250 Hz for BCI Competition IV 2a)
            model_type: 'eegnet' or 'transformer'. Controls data shape.
        """
        self.data_path = data_path
        self.subjects = subjects
        self.selected_channels = selected_channels
        self.freq_band = freq_band
        self.epoch_duration = epoch_duration
        self.epoch_offset = epoch_offset
        self.sfreq = sfreq
        self.model_type = model_type
        
        os.makedirs(data_path, exist_ok=True)
        
    def download_data(self):
        """
        Download and extract BCI Competition IV 2a dataset if not already present.
        """
        gdf_files_present = all(os.path.exists(os.path.join(self.data_path, f"A0{subj}T.gdf")) for subj in self.subjects)
        
        if gdf_files_present:
            print("Dataset already downloaded and extracted.")
            return

        zip_url = "http://www.bbci.de/competition/iv/dads/BCICIV_2a_gdf.zip"
        zip_path = os.path.join(self.data_path, "BCICIV_2a_gdf.zip")

        if not os.path.exists(zip_path):
            print("="*60)
            print("BCI COMPETITION IV 2a DATASET")
            print("="*60)
            print(f"Downloading dataset from {zip_url}...")
            
            try:
                urllib.request.urlretrieve(zip_url, zip_path, self._report_hook)
                print("\nDownload complete.")
            except Exception as e:
                print(f"\nError downloading dataset: {e}")
                print("Please manually download 'Dataset 2a' from http://www.bbci.de/competition/iv/ and place the .gdf files in the 'data' directory.")
                return

        print(f"Extracting dataset to {self.data_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return
        finally:
            # Clean up the zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)
    
    def _report_hook(self, count, block_size, total_size):
        """
        A hook to show download progress.
        """
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading... {percent}%", end="")
        
    def load_raw_gdf(self, filepath):
        """
        Load raw .gdf file using MNE
        """
        try:
            raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
            return raw
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def apply_bandpass_filter(self, raw):
        """
        Apply bandpass filter to isolate motor imagery frequencies
        """
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=self.freq_band[0], 
            h_freq=self.freq_band[1],
            method='iir',
            verbose=False
        )
        return raw_filtered
    
    def select_channels(self, raw):
        """
        Select specific motor cortex channels
        """
        try:
            raw_selected = raw.copy().pick_channels(self.selected_channels)
            return raw_selected
        except Exception as e:
            print(f"Channel selection error: {e}")
            print(f"Available channels: {raw.ch_names}")
            return raw
    
    def create_epochs(self, raw):
        """
        Create epochs based on event markers
        BCI Competition IV 2a event IDs:
        - 769: Left Hand
        - 770: Right Hand
        - 771: Feet
        - 772: Tongue
        """
        # Find events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Create a mapping from the integer event code to the desired label
        event_mapping = {}
        if np.str_('769') in event_id:
            event_mapping[event_id[np.str_('769')]] = 0 # Left Hand
        if np.str_('770') in event_id:
            event_mapping[event_id[np.str_('770')]] = 1 # Right Hand
        if np.str_('771') in event_id:
            event_mapping[event_id[np.str_('771')]] = 2 # Feet
        if np.str_('772') in event_id:
            event_mapping[event_id[np.str_('772')]] = 3 # Tongue
        
        # Filter events
        valid_events = []
        for event in events:
            event_code = event[2]
            if event_code in event_mapping:
                valid_events.append([event[0], event[1], event_mapping[event_code]])
        
        if len(valid_events) == 0:
            print("No valid events found!")
            return None
        
        valid_events = np.array(valid_events)
        
        # Create epochs
        tmin = self.epoch_offset
        tmax = self.epoch_offset + self.epoch_duration
        
        epochs = mne.Epochs(
            raw,
            valid_events,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        return epochs
    
    def process_single_subject(self, subject_id, session='T'):
        """
        Process single subject data
        Args:
            subject_id: 1-9
            session: 'T' (training) or 'E' (evaluation)
        """
        filename = f"A0{subject_id}{session}.gdf"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None, None
        
        print(f"Processing {filename}...")
        
        # Load raw data
        raw = self.load_raw_gdf(filepath)
        if raw is None:
            return None, None
        
        # Apply bandpass filter
        raw = self.apply_bandpass_filter(raw)
        
        # Select channels
        raw = self.select_channels(raw)
        
        # Create epochs
        epochs = self.create_epochs(raw)
        if epochs is None:
            return None, None
        
        # Extract data and labels
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        labels = epochs.events[:, -1]  # Event labels
        
        print(f"  → Extracted {len(labels)} epochs")
        print(f"  → Shape: {data.shape}")
        
        return data, labels
    
    def load_and_preprocess(self, num_classes=2):
        """
        Load and preprocess all subjects
        Args:
            num_classes: 2 (Left vs Right) or 4 (all classes)
        Returns:
            X_train, y_train, X_val, y_val
        """
        all_data = []
        all_labels = []

        for subject_id in self.subjects:
            # Load training session
            data, labels = self.process_single_subject(subject_id, session='T')

            if data is not None:
                # For 2-class: keep only Left (0) and Right (1)
                if num_classes == 2:
                    mask = (labels == 0) | (labels == 1)
                    data = data[mask]
                    labels = labels[mask]

                all_data.append(data)
                all_labels.append(labels)

        if len(all_data) == 0:
            raise ValueError("No data loaded! Please download dataset first.")

        # Concatenate all subjects
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # CRITICAL FIX: Standardize the data
        # EEG data after filtering has very small values, need to normalize
        # Compute mean and std across all samples for each channel
        X_mean = X.mean(axis=(0, 2), keepdims=True)  # Mean across trials and time
        X_std = X.std(axis=(0, 2), keepdims=True)    # Std across trials and time
        X = (X - X_mean) / (X_std + 1e-8)  # Standardize

        print(f"\nData normalization applied:")
        print(f"  Mean per channel: {X_mean.squeeze()}")
        print(f"  Std per channel: {X_std.squeeze()}")
        print(f"  Normalized data - Mean: {X.mean():.6f}, Std: {X.std():.6f}")

        # Shape data based on model type
        if self.model_type == 'eegnet':
            # Add channel dimension for EEGNet (expects 1 "kernel" dimension)
            # Shape: (trials, 1, channels, timepoints)
            X = np.expand_dims(X, axis=1)
        # For 'transformer', shape is already (trials, channels, timepoints)

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total trials: {len(y)}")
        print(f"Data shape: {X.shape} (for model_type='{self.model_type}')")
        print(f"Class distribution: {np.bincount(y)}")
        print("=" * 60)

        # Train/validation split (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, y_train, X_val, y_val
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, batch_size=64):
        """
        Create PyTorch DataLoaders
        """
        train_dataset = BCICompetition2aDataset(X_train, y_train)
        val_dataset = BCICompetition2aDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader


def get_single_subject_data(subject_id, data_path='./data', model_type='eegnet'):
    """
    Helper function to get data for a single subject (for fine-tuning)
    """
    processor = BCIDataProcessor(
        data_path=data_path,
        subjects=[subject_id],
        model_type=model_type
    )
    
    X_train, y_train, X_val, y_val = processor.load_and_preprocess(num_classes=2)
    
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    # Example usage
    processor = BCIDataProcessor(
        data_path='./data',
        subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9]  # All 9 subjects
    )
    
    # First time: download instructions
    processor.download_data()
    
    # After downloading, load and preprocess
    # X_train, y_train, X_val, y_val = processor.load_and_preprocess()
    
    # Create dataloaders
    # train_loader, val_loader = processor.create_dataloaders(
    #     X_train, y_train, X_val, y_val, batch_size=64
    # )

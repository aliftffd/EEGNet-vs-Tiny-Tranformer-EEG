"""
Modified training script to include OpenBCI transfer learning
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import sys
import os

# Import from your working train.py
from train import BCITrainer, EEGTransformerTransfer, count_parameters
from dataset import BCIDataProcessor, BCICompetition2aDataset
from models import EEGTransformer


class OpenBCIDataLoader:
    """Load and preprocess your OpenBCI S01 data"""
    
    def __init__(self, data_dir, num_classes=2):
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        
    def load_data(self):
        """Load OpenBCI data with same preprocessing as competition"""
        from scipy import signal as sp_signal
        import pandas as pd
        
        session_dir = self.data_dir / "S01" / "training"
        metadata_file = session_dir / "S01_training_metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        all_data = []
        all_labels = []
        
        print(f"Loading OpenBCI data from {session_dir}...")
        
        for trial_meta in metadata['trials']:
            csv_file = session_dir / trial_meta['filename']
            if not csv_file.exists():
                continue
            
            df = pd.read_csv(csv_file)
            
            # Extract C3, C4, Cz (match competition channels)
            data = df[['C3', 'C4', 'Cz']].values.T  # (3, n_samples)
            
            # Extract 0.5-2.5s window (500 samples @ 250Hz)
            start_idx = int(0.5 * 250)
            end_idx = int(2.5 * 250)
            
            if end_idx > data.shape[1]:
                padding = np.zeros((3, end_idx - data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
            
            data = data[:, start_idx:end_idx]  # (3, 500)

            # Pad to 512 to be compatible with transformer
            target_len = 512
            pad_len = target_len - data.shape[1]
            if pad_len > 0:
                pad_width = ((0, 0), (0, pad_len))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
            
            # Apply 8-30 Hz bandpass
            nyq = 250 / 2.0
            b, a = sp_signal.butter(5, [8.0/nyq, 30.0/nyq], btype='band')
            for ch in range(3):
                data[ch, :] = sp_signal.filtfilt(b, a, data[ch, :])
            
            all_data.append(data)
            all_labels.append(trial_meta['class_id'])
        
        X = np.array(all_data)
        y = np.array(all_labels)
        
        print(f"✓ Loaded {len(y)} trials")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {np.bincount(y)}")
        
        # Filter to 2 classes (Left vs Right)
        if self.num_classes == 2:
            mask = (y == 0) | (y == 1)
            X = X[mask]
            y = y[mask]
            print(f"✓ Filtered to 2 classes: {len(y)} trials")
        
        # Global normalization (CRITICAL - same as competition!)
        X_mean = X.mean(axis=(0, 2), keepdims=True)
        X_std = X.std(axis=(0, 2), keepdims=True)
        X = (X - X_mean) / (X_std + 1e-8)
        
        print(f"  After normalization: mean={X.mean():.6f}, std={X.std():.6f}")
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, y_train, X_val, y_val


def experiment_1_pretrain():
    """Experiment 1: Pre-train on competition data"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: PRE-TRAIN ON COMPETITION DATA (9 SUBJECTS)")
    print("="*70)
    
    processor = BCIDataProcessor(
        data_path='./data',
        subjects=list(range(1, 10)),
        model_type='transformer'
    )
    
    processor.download_data()
    X_train, y_train, X_val, y_val = processor.load_and_preprocess(num_classes=2)
    train_loader, val_loader = processor.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGTransformer(
        num_classes=2,
        in_channels=3,
        seq_length=512,
        d_model=128,
        n_head=8,
        n_layers=6,
        ffn_hidden=256,
        drop_prob=0.1,
        device=device
    )
    
    trainer = BCITrainer(model, device=device, learning_rate=0.001, weight_decay=0.01)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=100,
        save_dir='./pretrained_competition',
        early_stopping_patience=20
    )
    
    trainer.evaluate_and_plot(val_loader, save_dir='./pretrained_competition', 
                             class_names=['Left', 'Right'])
    
    print(f"\n✓ Experiment 1 Complete! Best accuracy: {best_acc:.2f}%")
    return './pretrained_competition/best_model.pth'


def experiment_2_transfer(pretrained_path):
    """Experiment 2: Transfer learning to your OpenBCI data"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: TRANSFER LEARNING TO YOUR OPENBCI DATA (S01)")
    print("="*70)
    
    # Load your OpenBCI data
    loader = OpenBCIDataLoader(
        data_dir='./openbci/pythonbci/bci_competition_dataset',
        num_classes=2
    )
    X_train, y_train, X_val, y_val = loader.load_data()
    
    # Create dataloaders
    train_dataset = BCICompetition2aDataset(X_train, y_train)
    val_dataset = BCICompetition2aDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load pre-trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model = EEGTransformer(
        num_classes=2, in_channels=3, seq_length=512,
        d_model=128, n_head=8, n_layers=6,
        ffn_hidden=256, drop_prob=0.1, device=device
    )
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create transfer model
    transfer_model = EEGTransformerTransfer(pretrained_model, num_classes=2)
    transfer_model.unfreeze_features()
    
    # Fine-tune
    trainer = BCITrainer(transfer_model, device=device, learning_rate=0.0001, weight_decay=0.01)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=50,
        save_dir='./finetuned_openbci',
        early_stopping_patience=15
    )
    
    trainer.evaluate_and_plot(val_loader, save_dir='./finetuned_openbci',
                             class_names=['Left', 'Right'])
    
    print(f"\n✓ Experiment 2 Complete! Best accuracy: {best_acc:.2f}%")
    return best_acc


def experiment_3_scratch():
    """Experiment 3: Train on your data from scratch (no transfer)"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: TRAIN ON YOUR DATA FROM SCRATCH (BASELINE)")
    print("="*70)
    
    # Load your data
    loader = OpenBCIDataLoader(
        data_dir='./openbci/pythonbci/bci_competition_dataset',
        num_classes=2
    )
    X_train, y_train, X_val, y_val = loader.load_data()
    
    train_dataset = BCICompetition2aDataset(X_train, y_train)
    val_dataset = BCICompetition2aDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create fresh model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGTransformer(
        num_classes=2, in_channels=3, seq_length=512,
        d_model=128, n_head=8, n_layers=6,
        ffn_hidden=256, drop_prob=0.1, device=device
    )
    
    trainer = BCITrainer(model, device=device, learning_rate=0.001, weight_decay=0.01)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=100,
        save_dir='./scratch_openbci',
        early_stopping_patience=20
    )
    
    trainer.evaluate_and_plot(val_loader, save_dir='./scratch_openbci',
                             class_names=['Left', 'Right'])
    
    print(f"\n✓ Experiment 3 Complete! Best accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    print("="*70)
    print("CAPSTONE PROJECT: TRANSFER LEARNING FOR MOTOR IMAGERY BCI")
    print("="*70)
    
    # Run all 3 experiments
    pretrained_path = experiment_1_pretrain()
    transfer_acc = experiment_2_transfer(pretrained_path)
    scratch_acc = experiment_3_scratch()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"Competition baseline:     60-70% (from Exp 1)")
    print(f"Your data (transfer):     {transfer_acc:.2f}%")
    print(f"Your data (from scratch): {scratch_acc:.2f}%")
    print(f"Improvement:              {transfer_acc - scratch_acc:+.2f}%")
    print("="*70)

"""
Capstone Experiment: Competition Pre-training + Transfer to Your OpenBCI Data
"""

import sys
sys.path.append('./models')

from train import pretrain_model, BCITrainer, EEGTransformerTransfer
from dataset import BCIDataProcessor
from models.eeg_transformer import EEGTransformer
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

class OpenBCIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_your_openbci_data(data_dir, num_classes=2):
    """
    Load your S01 OpenBCI data with same preprocessing as competition
    """
    import mne
    from scipy import signal as sp_signal
    
    session_dir = Path(data_dir) / "S01" / "training"
    metadata_file = session_dir / "S01_training_metadata.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    all_data = []
    all_labels = []
    
    print(f"Loading your OpenBCI data from {session_dir}...")
    
    for trial_meta in metadata['trials']:
        csv_file = session_dir / trial_meta['filename']
        if not csv_file.exists():
            continue
        
        # Load CSV
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Extract C3, C4, Cz
        data = df[['C3', 'C4', 'Cz']].values.T  # (3, n_samples)
        
        # Extract 0.5-2.5s window (motor imagery period)
        # Your data has 250 Hz sampling rate
        start_idx = int(0.5 * 250)  # 125 samples
        end_idx = int(2.5 * 250)    # 625 samples
        
        if end_idx > data.shape[1]:
            # Pad if needed
            padding = np.zeros((3, end_idx - data.shape[1]))
            data = np.concatenate([data, padding], axis=1)
        
        data = data[:, start_idx:end_idx]  # (3, 500)
        
        # Apply 8-30 Hz bandpass (same as competition)
        nyq = 250 / 2.0
        b, a = sp_signal.butter(5, [8.0/nyq, 30.0/nyq], btype='band')
        for ch in range(3):
            data[ch, :] = sp_signal.filtfilt(b, a, data[ch, :])
        
        all_data.append(data)
        all_labels.append(trial_meta['class_id'])
    
    X = np.array(all_data)  # (n_trials, 3, 500)
    y = np.array(all_labels)
    
    # Filter to 2 classes if needed
    if num_classes == 2:
        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
    
    # Global normalization (CRITICAL!)
    X_mean = X.mean(axis=(0, 2), keepdims=True)
    X_std = X.std(axis=(0, 2), keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)
    
    print(f"✓ Loaded {len(y)} trials from OpenBCI")
    print(f"  Left: {np.sum(y==0)}, Right: {np.sum(y==1)}")
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_val, y_val


def main():
    print("="*70)
    print("CAPSTONE EXPERIMENT: TRANSFER LEARNING")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================================================================
    # EXPERIMENT 1: Pre-train on Competition Data
    # ==================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: PRE-TRAIN ON COMPETITION DATA")
    print("="*70)
    
    pretrained_model = pretrain_model(
        data_path='./data',
        save_dir='./pretrained_transformer',
        epochs=100,
        model_type='transformer'
    )
    
    # ==================================================================
    # EXPERIMENT 2: Fine-tune on Your OpenBCI Data
    # ==================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: FINE-TUNE ON YOUR OPENBCI DATA")
    print("="*70)
    
    # Load your data
    X_train, y_train, X_val, y_val = load_your_openbci_data(
        './openbci/pythonbci/bci_competition_dataset',
        num_classes=2
    )
    
    # Create dataloaders
    train_dataset = OpenBCIDataset(X_train, y_train)
    val_dataset = OpenBCIDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load pre-trained model
    checkpoint = torch.load('./pretrained_transformer/best_model.pth', map_location=device)
    
    pretrained_model = EEGTransformer(
        num_classes=2, in_channels=3, seq_length=500,
        d_model=128, n_head=8, n_layers=6,
        ffn_hidden=256, drop_prob=0.1, device=device
    )
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create transfer model
    transfer_model = EEGTransformerTransfer(pretrained_model, num_classes=2)
    transfer_model.unfreeze_features()
    
    # Fine-tune
    trainer = BCITrainer(transfer_model, device=device, learning_rate=0.0001)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=50,
        save_dir='./finetuned_your_data',
        early_stopping_patience=15
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Competition pre-training: ~60-70% (from your earlier runs)")
    print(f"Fine-tuned on your data:  {best_acc:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()

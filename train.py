"""
Training Pipeline for EEG-based Motor Imagery BCI
Supports both pre-training and transfer learning workflows
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
import json
import optuna

from model import EEGNet, EEGNetTransfer, count_parameters
from dataset import BCIDataProcessor, get_single_subject_data


class BCITrainer:
    """
    Trainer class for EEG-based BCI with transfer learning support
    """
    def __init__(self,
                 model,
                 device='cuda',
                 learning_rate=0.001,
                 weight_decay=0.01,
                 use_amp=True):
        """
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            learning_rate: initial learning rate
            weight_decay: L2 regularization
            use_amp: use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = accuracy_score(all_labels, all_preds) * 100
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(all_labels, all_preds) * 100
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc='Validation'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = accuracy_score(all_labels, all_preds) * 100
        
        return avg_loss, avg_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=100, save_dir='./checkpoints', early_stopping_patience=20, trial=None):
        """
        Full training loop with early stopping
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Epochs: {epochs}")
        print("=" * 60 + "\n")
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_acc)
                print(f"  ✓ New best model saved! (Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1

            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered (patience: {early_stopping_patience})")
                break
            
            # Save latest checkpoint
            latest_path = os.path.join(save_dir, 'latest_model.pth')
            self.save_checkpoint(latest_path, epoch, val_acc)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print("=" * 60)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        return self.best_val_acc
    
    def save_checkpoint(self, path, epoch, val_acc):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%")
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
        plt.close()
        
        print(f"Training curves saved to {save_dir}/training_curves.png")
    
    def evaluate_and_plot(self, val_loader, save_dir='./results', class_names=['Left', 'Right']):
        """
        Evaluate model and generate confusion matrix
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        _, val_acc, val_preds, val_labels = self.validate(val_loader)
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (Accuracy: {val_acc:.2f}%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Classification report
        report = classification_report(val_labels, val_preds, target_names=class_names)
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60)
        
        # Save report to file
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        return val_acc, cm


def pretrain_model(data_path='./data', save_dir='./pretrained', epochs=100, use_best_params=False):
    """
    Pre-train EEGNet on all subjects (subject-independent)
    """
    print("\n" + "=" * 70)
    print("STAGE 1: PRE-TRAINING ON MULTI-SUBJECT DATA")
    print("=" * 70)
    
    # Hyperparameters
    if use_best_params:
        best_params_path = './best_params.json'
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                params = json.load(f)
            print("✓ Using best hyperparameters from Optuna.")
        else:
            print("⚠ Best parameters file not found. Using default hyperparameters.")
            params = {'learning_rate': 0.001, 'weight_decay': 0.01, 'dropout_rate': 0.5}
    else:
        params = {'learning_rate': 0.001, 'weight_decay': 0.01, 'dropout_rate': 0.5}

    # Load multi-subject data
    processor = BCIDataProcessor(
        data_path=data_path,
        subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    processor.download_data()
    
    X_train, y_train, X_val, y_val = processor.load_and_preprocess(num_classes=2)
    train_loader, val_loader = processor.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGNet(
        num_classes=2,
        channels=3,
        samples=500,
        dropout_rate=params['dropout_rate']
    )
    
    # Train
    trainer = BCITrainer(model, device=device, learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])
    best_acc = trainer.train(
        train_loader, val_loader, 
        epochs=epochs, 
        save_dir=save_dir,
        early_stopping_patience=20
    )
    
    # Evaluate
    trainer.evaluate_and_plot(val_loader, save_dir=save_dir)
    
    print(f"\n✓ Pre-training completed! Best accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to: {save_dir}/best_model.pth")
    
    return trainer.model


def finetune_model(pretrained_path, subject_id, data_path='./data', save_dir='./finetuned', epochs=50):
    """
    Fine-tune pre-trained model on single subject data (transfer learning)
    """
    print("\n" + "=" * 70)
    print(f"STAGE 2: FINE-TUNING ON SUBJECT {subject_id}")
    print("=" * 70)
    
    # Load single subject data
    X_train, y_train, X_val, y_val = get_single_subject_data(subject_id, data_path)
    
    processor = BCIDataProcessor(data_path=data_path)
    train_loader, val_loader = processor.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Load pre-trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model = EEGNet(num_classes=2, channels=3, samples=500)
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded pre-trained model from: {pretrained_path}")
    
    # Create transfer learning model
    transfer_model = EEGNetTransfer(pretrained_model, num_classes=2)
    
    # Strategy 1: Freeze features, train only classifier (fast adaptation)
    print("\n--- Phase 1: Training classifier only ---")
    transfer_model.freeze_features()
    trainer = BCITrainer(transfer_model, device=device, learning_rate=0.01)
    trainer.train(
        train_loader, val_loader,
        epochs=20,
        save_dir=os.path.join(save_dir, 'phase1'),
        early_stopping_patience=10
    )
    
    # Strategy 2: Unfreeze last layer, fine-tune (better performance)
    print("\n--- Phase 2: Fine-tuning last layer ---")
    transfer_model.unfreeze_last_n_layers(n=1)
    trainer.optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=30,
        save_dir=os.path.join(save_dir, 'phase2'),
        early_stopping_patience=15
    )
    
    # Evaluate final model
    trainer.evaluate_and_plot(val_loader, save_dir=save_dir)
    
    print(f"\n✓ Fine-tuning completed! Best accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to: {save_dir}/phase2/best_model.pth")
    
    return best_acc


if __name__ == "__main__":
    import argparse
    from tabulate import tabulate
    
    parser = argparse.ArgumentParser(description='Train EEGNet for BCI')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'], 
                       default='both', help='Training mode')
    parser.add_argument('--data_path', type=str, default='./data', 
                       help='Path to BCI Competition IV 2a data')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs for pre-training')
    parser.add_argument('--subject', type=int, default=1,
                        help='Subject ID for fine-tuning (1-9)')
    parser.add_argument('--all-subjects', action='store_true',
                        help='Fine-tune on all subjects sequentially')
    parser.add_argument('--use-best-params', action='store_true',
                        help='Use best hyperparameters from Optuna tuning')
    
    args = parser.parse_args()
    
    if args.mode == 'pretrain' or args.mode == 'both':
        # Pre-train on all subjects
        pretrained_model = pretrain_model(
            data_path=args.data_path,
            save_dir='./pretrained',
            epochs=args.epochs,
            use_best_params=args.use_best_params
        )
    
    if args.mode == 'finetune' or args.mode == 'both':
        pretrained_path = './pretrained/best_model.pth'
        if not os.path.exists(pretrained_path):
            print(f"Pretrained model not found at {pretrained_path}. Please run in 'pretrain' or 'both' mode first.")
        else:
            results = []
            if args.all_subjects:
                for subject_id in range(1, 10):
                    best_acc = finetune_model(
                        pretrained_path=pretrained_path,
                        subject_id=subject_id,
                        data_path=args.data_path,
                        save_dir=f'./finetuned_models/finetuned_subject{subject_id}',
                        epochs=50
                    )
                    results.append({'subject': subject_id, 'accuracy': best_acc})
            else:
                best_acc = finetune_model(
                    pretrained_path=pretrained_path,
                    subject_id=args.subject,
                    data_path=args.data_path,
                    save_dir=f'./finetuned_models/finetuned_subject{args.subject}',
                    epochs=50
                )
                results.append({'subject': args.subject, 'accuracy': best_acc})
            
            if results:
                print("\n" + "=" * 70)
                print("FINE-TUNING RESULTS SUMMARY")
                print("=" * 70)
                headers = ["Subject", "Best Accuracy"]
                rows = [[res['subject'], f"{res['accuracy']:.2f}%"] for res in results]
                print(tabulate(rows, headers=headers, tablefmt="grid"))

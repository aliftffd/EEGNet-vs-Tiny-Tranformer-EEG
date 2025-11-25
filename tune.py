
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json

from model import EEGNet
from dataset import BCIDataProcessor
from train import BCITrainer

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.8)

    # Model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGNet(
        num_classes=2,
        channels=3,
        samples=500,
        dropout_rate=dropout_rate
    )
    
    processor = BCIDataProcessor(
        data_path='./data',
        subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    
    X_train, y_train, X_val, y_val = processor.load_and_preprocess(num_classes=2)
    train_loader, val_loader = processor.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Trainer
    trainer = BCITrainer(
        model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Train and validate
    best_acc = trainer.train(
        train_loader, val_loader, 
        epochs=300, 
        save_dir=f'./tuning_trials/trial_{trial.number}',
        early_stopping_patience=10,
        trial=trial
    )
    
    return best_acc

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for EEGNet with Optuna')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()

    # Create a study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    # Run the optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Get the best trial
    best_trial = study.best_trial
    
    print("\n" + "=" * 70)
    print("BEST TRIAL")
    print("=" * 70)
    print(f"  Value: {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    # Save the best parameters
    best_params_path = './best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
        
    print(f"\n✓ Best parameters saved to: {best_params_path}")

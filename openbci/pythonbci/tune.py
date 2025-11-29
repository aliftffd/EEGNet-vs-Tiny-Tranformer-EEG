
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
from functools import partial

from model import EEGNet
from models.eeg_transformer import EEGTransformer
from dataset import BCIDataProcessor
from train import BCITrainer

def objective(trial, model_type):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == 'eegnet':
        # Hyperparameter search space for EEGNet
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)

        # Model and data
        model = EEGNet(
            num_classes=2,
            channels=3,
            samples=501,  # Actual data sequence length is 501
            dropout_rate=dropout_rate
        )
    elif model_type == 'transformer':
        # Hyperparameter search space for EEGTransformer
        # Reduced model sizes to fit in ~4GB GPU memory
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        drop_prob = trial.suggest_float('drop_prob', 0.1, 0.5)
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])  # Reduced from [64, 128, 256]
        n_head = trial.suggest_categorical('n_head', [4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 4)  # Reduced from 2-6 to 2-4
        ffn_hidden = trial.suggest_int('ffn_hidden', d_model, d_model * 3)  # Reduced multiplier from 4 to 3

        model = EEGTransformer(
            num_classes=2,
            in_channels=3,
            seq_length=501,  # Actual data sequence length is 501
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            device=device,
            embedding_type='conv1d'  # Use conv1d to avoid segment size divisibility constraint
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    processor = BCIDataProcessor(
        data_path='./data',
        subjects=list(range(1, 10)),
        model_type=model_type
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
        epochs=100, 
        save_dir=f'./tuning_trials/trial_{trial.number}_{model_type}',
        early_stopping_patience=15,
        trial=trial
    )
    
    return best_acc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter tuning for EEG Models with Optuna')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--model_type', type=str, choices=['eegnet', 'transformer'],
                        default='eegnet', help='Model to tune')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: {model_type}_study)')
    parser.add_argument('--dashboard-port', type=int, default=8080,
                        help='Port for Optuna dashboard (default: 8080)')
    args = parser.parse_args()

    # Set study name
    study_name = args.study_name if args.study_name else f'{args.model_type}_study'

    # Create SQLite database for persistent storage
    storage_name = f'sqlite:///optuna_{args.model_type}.db'

    # Create a study with persistent storage
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=pruner,
        storage=storage_name,
        load_if_exists=True  # Resume existing study if available
    )

    print("\n" + "=" * 70)
    print("OPTUNA DASHBOARD")
    print("=" * 70)
    print(f"Study name: {study_name}")
    print(f"Database: {storage_name}")
    print(f"\nTo view the dashboard, run in another terminal:")
    print(f"  optuna-dashboard {storage_name} --port {args.dashboard_port}")
    print(f"\nThen open: http://localhost:{args.dashboard_port}")
    print("=" * 70 + "\n")
    
    # Run the optimization
    objective_func = partial(objective, model_type=args.model_type)
    study.optimize(objective_func, n_trials=args.n_trials)
    
    # Get the best trial
    best_trial = study.best_trial
    
    print("\n" + "=" * 70)
    print(f"BEST TRIAL FOR {args.model_type.upper()}")
    print("=" * 70)
    print(f"  Value: {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    # Save the best parameters
    best_params_path = f'./best_params_{args.model_type}.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)

    print(f"\n✓ Best parameters saved to: {best_params_path}")

    # Generate and save visualization plots
    try:
        import optuna.visualization as vis
        import plotly.io as pio

        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATION PLOTS")
        print("=" * 70)

        viz_dir = f'./optuna_viz_{args.model_type}'
        os.makedirs(viz_dir, exist_ok=True)

        # Optimization history
        fig = vis.plot_optimization_history(study)
        pio.write_html(fig, f'{viz_dir}/optimization_history.html')
        print(f"✓ Optimization history: {viz_dir}/optimization_history.html")

        # Parameter importances
        fig = vis.plot_param_importances(study)
        pio.write_html(fig, f'{viz_dir}/param_importances.html')
        print(f"✓ Parameter importances: {viz_dir}/param_importances.html")

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        pio.write_html(fig, f'{viz_dir}/parallel_coordinate.html')
        print(f"✓ Parallel coordinate: {viz_dir}/parallel_coordinate.html")

        # Slice plot
        fig = vis.plot_slice(study)
        pio.write_html(fig, f'{viz_dir}/slice.html')
        print(f"✓ Slice plot: {viz_dir}/slice.html")

        print(f"\n✓ All visualizations saved to: {viz_dir}/")
        print("=" * 70)
    except Exception as e:
        print(f"\n⚠ Could not generate visualizations: {e}")

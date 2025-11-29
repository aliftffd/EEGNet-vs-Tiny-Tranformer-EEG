"""
Motor Imagery Dataset Analysis - Standalone Script
Generates all plots at 600 DPI for publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import signal as sp_signal
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "bci_competition_dataset"  # Change this to your dataset folder
OUTPUT_FOLDER = "bci_competition_dataset/outputs/plots_600dpi"
SAMPLE_RATE = 250  # Hz

# Set high-quality plot parameters
plt.rcParams['figure.dpi'] = 150  # Screen display
plt.rcParams['savefig.dpi'] = 600  # File export (HIGH QUALITY)
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

sns.set_palette("husl")

# ============================================================================
# LOAD DATA
# ============================================================================

def load_dataset(dataset_path):
    """Load all CSV files from dataset"""
    dataset_path = Path(dataset_path)
    csv_files = sorted(dataset_path.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    trials = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Extract trial info
        if 'label' in df.columns:
            label = df['label'].iloc[0]
        else:
            label = csv_file.stem.split('_')[1] + '_' + csv_file.stem.split('_')[2]
        
        class_id = int(df['class_id'].iloc[0]) if 'class_id' in df.columns else -1
        
        trials[csv_file.stem] = {
            'filename': csv_file.name,
            'label': label,
            'class_id': class_id,
            'data': df,
            'num_samples': len(df),
            'duration': len(df) / SAMPLE_RATE
        }
    
    return trials

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def bandpass_filter(data, lowcut, highcut, fs=250, order=5):
    """Apply bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp_signal.butter(order, [low, high], btype='band')
    return sp_signal.filtfilt(b, a, data)

def calculate_band_power(data, lowcut, highcut, fs=250):
    """Calculate average power in frequency band"""
    filtered = bandpass_filter(data, lowcut, highcut, fs)
    power = np.mean(filtered ** 2)
    return power

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_raw_signals(trials, output_folder):
    """Plot 1: Raw EEG signals"""
    channels = ['C3_left_motor', 'C4_right_motor', 'Cz_reference']
    colors_map = {
        'left_hand': '#FF6B6B',
        'right_hand': '#4ECDC4',
        'both_hands': '#45B7D1',
        'both_feet': '#96CEB4',
        'feet': '#96CEB4',
        'tongue': '#FFEAA7',
        'rest': '#DFE6E9',
    }
    
    plot_duration = 5
    plot_samples = SAMPLE_RATE * plot_duration
    
    print("\nGenerating Plot 1: Raw EEG Signals...")
    for trial_name, trial in trials.items():
        df = trial['data']
        label = trial['label']
        color = colors_map.get(label, '#999999')
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Raw EEG Signals: {label.replace('_', ' ').title()}", 
                     fontsize=14, fontweight='bold')
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].values[:plot_samples]
                time_axis = np.arange(len(data)) / SAMPLE_RATE
                
                axes[i].plot(time_axis, data, color=color, linewidth=0.8, alpha=0.8)
                axes[i].set_ylabel(f"{channel.replace('_', ' ').title()}\\n(µV)", 
                                  fontsize=10, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        output_file = Path(output_folder) / f"01_raw_signals_{trial_name}.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
        plt.close()

def plot_psd(trials, output_folder):
    """Plot 2: Power Spectral Density"""
    channels = ['C3_left_motor', 'C4_right_motor', 'Cz_reference']
    colors_map = {
        'left_hand': '#FF6B6B',
        'right_hand': '#4ECDC4',
        'both_hands': '#45B7D1',
        'both_feet': '#96CEB4',
        'feet': '#96CEB4',
        'tongue': '#FFEAA7',
        'rest': '#DFE6E9',
    }
    
    print("\nGenerating Plot 2: Power Spectral Density...")
    for trial_name, trial in trials.items():
        df = trial['data']
        label = trial['label']
        color = colors_map.get(label, '#999999')
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Power Spectral Density: {label.replace('_', ' ').title()}", 
                     fontsize=14, fontweight='bold')
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].values
                freqs, psd = sp_signal.welch(data, fs=SAMPLE_RATE, nperseg=min(512, len(data)))
                
                axes[i].semilogy(freqs, psd, color=color, linewidth=2, alpha=0.8)
                axes[i].axvspan(8, 12, alpha=0.2, color='green', label='Mu (8-12 Hz)')
                axes[i].axvspan(12, 30, alpha=0.2, color='blue', label='Beta (12-30 Hz)')
                
                axes[i].set_xlim([0, 50])
                axes[i].set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
                axes[i].set_ylabel('Power (µV²/Hz)', fontsize=10, fontweight='bold')
                axes[i].set_title(channel.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=9)
        
        plt.tight_layout()
        
        output_file = Path(output_folder) / f"02_psd_{trial_name}.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
        plt.close()

def plot_spectrograms(trials, output_folder):
    """Plot 3: Spectrograms"""
    channels = ['C3_left_motor', 'C4_right_motor', 'Cz_reference']
    
    print("\nGenerating Plot 3: Spectrograms...")
    for trial_name, trial in trials.items():
        df = trial['data']
        label = trial['label']
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f"Time-Frequency Analysis: {label.replace('_', ' ').title()}", 
                     fontsize=14, fontweight='bold')
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].values
                f, t, Sxx = sp_signal.spectrogram(data, fs=SAMPLE_RATE, 
                                                  nperseg=256, noverlap=128)
                
                im = axes[i].pcolormesh(t, f, 10 * np.log10(Sxx), 
                                       shading='gouraud', cmap='viridis')
                axes[i].set_ylim([0, 50])
                axes[i].set_ylabel(f"{channel.replace('_', ' ').title()}\\nFrequency (Hz)", 
                                  fontsize=10, fontweight='bold')
                
                axes[i].axhline(8, color='green', linestyle='--', alpha=0.5, linewidth=1)
                axes[i].axhline(12, color='green', linestyle='--', alpha=0.5, linewidth=1)
                axes[i].axhline(30, color='blue', linestyle='--', alpha=0.5, linewidth=1)
                
                plt.colorbar(im, ax=axes[i], label='Power (dB)')
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        output_file = Path(output_folder) / f"03_spectrogram_{trial_name}.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
        plt.close()

def plot_c3_vs_c4(trials, output_folder):
    """Plot 4: C3 vs C4 comparison"""
    plot_samples = SAMPLE_RATE * 5
    
    print("\nGenerating Plot 4: C3 vs C4 Comparison...")
    for trial_name, trial in trials.items():
        df = trial['data']
        label = trial['label']
        
        if 'C3_left_motor' in df.columns and 'C4_right_motor' in df.columns:
            c3 = df['C3_left_motor'].values[:plot_samples]
            c4 = df['C4_right_motor'].values[:plot_samples]
            time_axis = np.arange(len(c3)) / SAMPLE_RATE
            
            c3_norm = (c3 - np.mean(c3)) / (np.std(c3) + 1e-8)
            c4_norm = (c4 - np.mean(c4)) / (np.std(c4) + 1e-8)
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            fig.suptitle(f"Motor Cortex Lateralization: {label.replace('_', ' ').title()}", 
                         fontsize=14, fontweight='bold')
            
            ax.plot(time_axis, c3_norm, color='#FF6B6B', linewidth=1.5, 
                   label='C3 (Left Motor Cortex)', alpha=0.8)
            ax.plot(time_axis, c4_norm, color='#4ECDC4', linewidth=1.5, 
                   label='C4 (Right Motor Cortex)', alpha=0.8)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_file = Path(output_folder) / f"04_c3_vs_c4_{trial_name}.png"
            plt.savefig(output_file, dpi=600, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file.name}")
            plt.close()

def plot_band_power(trials, output_folder):
    """Plot 5: Frequency band power"""
    channels = ['C3_left_motor', 'C4_right_motor', 'Cz_reference']
    colors_map = {
        'left_hand': '#FF6B6B',
        'right_hand': '#4ECDC4',
        'both_hands': '#45B7D1',
        'both_feet': '#96CEB4',
        'feet': '#96CEB4',
        'tongue': '#FFEAA7',
        'rest': '#DFE6E9',
    }
    
    bands = {
        'Mu (8-12 Hz)': (8, 12),
        'Beta (12-30 Hz)': (12, 30),
    }
    
    print("\nGenerating Plot 5: Frequency Band Power...")
    for trial_name, trial in trials.items():
        df = trial['data']
        label = trial['label']
        color = colors_map.get(label, '#999999')
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Frequency Band Power: {label.replace('_', ' ').title()}", 
                     fontsize=14, fontweight='bold')
        
        for ch_idx, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].values
                
                band_names = list(bands.keys())
                powers = [calculate_band_power(data, low, high) for low, high in bands.values()]
                
                x = np.arange(len(band_names))
                axes[ch_idx].bar(x, powers, color=color, alpha=0.7, edgecolor='black')
                
                axes[ch_idx].set_xticks(x)
                axes[ch_idx].set_xticklabels(band_names, rotation=0, fontsize=9)
                axes[ch_idx].set_ylabel('Average Power (µV²)', fontsize=10, fontweight='bold')
                axes[ch_idx].set_title(channel.replace('_', ' ').title(), 
                                      fontsize=11, fontweight='bold')
                axes[ch_idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = Path(output_folder) / f"05_band_power_{trial_name}.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
        plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MOTOR IMAGERY DATASET ANALYSIS")
    print("Generating 600 DPI plots for publication")
    print("=" * 70)
    
    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    trials = load_dataset(DATASET_PATH)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    for trial_name, trial in trials.items():
        print(f"\n{trial['label'].upper()}:")
        print(f"  Samples: {trial['num_samples']}")
        print(f"  Duration: {trial['duration']:.2f} seconds")
        print(f"  Class ID: {trial['class_id']}")
    
    # Generate all plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_raw_signals(trials, OUTPUT_FOLDER)
    plot_psd(trials, OUTPUT_FOLDER)
    plot_spectrograms(trials, OUTPUT_FOLDER)
    plot_c3_vs_c4(trials, OUTPUT_FOLDER)
    plot_band_power(trials, OUTPUT_FOLDER)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    output_path = Path(OUTPUT_FOLDER)
    png_files = list(output_path.glob("*.png"))
    print(f"\nTotal plots generated: {len(png_files)}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"\n✅ All plots saved at 600 DPI for publication quality!")

if __name__ == "__main__":
    main()

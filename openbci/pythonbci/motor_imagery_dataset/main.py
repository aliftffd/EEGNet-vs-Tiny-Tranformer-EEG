"""
Quick visualization for S01 Motor Imagery trials
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from pathlib import Path

# Load the three trials
trials = {
    'Left Hand': pd.read_csv('S01/session_01/S01_left_hand_trial_001_20251129_124613.csv'),
    'Both Hands': pd.read_csv('S01/session_01/S01_both_hands_trial_002_20251129_124642.csv'),
    'Feet': pd.read_csv('S01/session_01/S01_feet_trial_003_20251129_124731.csv'),
}

print("=== Dataset Summary ===")
for label, df in trials.items():
    print(f"\n{label}:")
    print(f"  Samples: {len(df)}")
    print(f"  Duration: {len(df)/250:.2f} seconds")
    print(f"  Quality Score: {df['quality_score'].iloc[0]:.2f}")
    print(f"  Class ID: {df['class_id'].iloc[0]}")

# ============================================================================
# PLOT 1: Raw EEG Signals Comparison
# ============================================================================

fig1, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
fig1.suptitle('Motor Imagery EEG Signals - S01', fontsize=16, fontweight='bold')

channels = ['C3_left_motor', 'C4_right_motor', 'Cz_reference']
colors = {'Left Hand': '#FF6B6B', 'Both Hands': '#45B7D1', 'Feet': '#96CEB4'}

sample_rate = 250
plot_duration = 5  # Show first 5 seconds
plot_samples = sample_rate * plot_duration

for col_idx, channel in enumerate(channels):
    for row_idx, (label, df) in enumerate(trials.items()):
        ax = axes[row_idx, col_idx]
        
        # Get data (first 5 seconds)
        data = df[channel].values[:plot_samples]
        time_axis = np.arange(len(data)) / sample_rate
        
        # Plot
        ax.plot(time_axis, data, color=colors[label], linewidth=0.8, alpha=0.8)
        ax.grid(True, alpha=0.3)
        
        # Labels
        if col_idx == 0:
            ax.set_ylabel(f'{label}\n(µV)', fontsize=10, fontweight='bold')
        if row_idx == 0:
            ax.set_title(channel.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        if row_idx == 2:
            ax.set_xlabel('Time (seconds)', fontsize=10)

plt.tight_layout()
plt.savefig('eeg_signals_comparison.png', dpi=600, bbox_inches='tight')
print("\n✓ Saved: eeg_signals_comparison.png")

# ============================================================================
# PLOT 2: Power Spectral Density
# ============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Power Spectral Density - Motor Imagery', fontsize=16, fontweight='bold')

for col_idx, channel in enumerate(channels):
    ax = axes[col_idx]
    
    for label, df in trials.items():
        # Compute PSD
        data = df[channel].values
        freqs, psd = sp_signal.welch(data, fs=sample_rate, nperseg=min(512, len(data)))
        
        # Plot
        ax.semilogy(freqs, psd, color=colors[label], linewidth=2, label=label, alpha=0.8)
    
    # Highlight mu and beta bands
    ax.axvspan(8, 12, alpha=0.15, color='green', label='Mu (8-12 Hz)' if col_idx == 0 else '')
    ax.axvspan(12, 30, alpha=0.15, color='blue', label='Beta (12-30 Hz)' if col_idx == 0 else '')
    
    ax.set_xlim([0, 50])
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (µV²/Hz)', fontsize=10)
    ax.set_title(channel.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('power_spectrum_comparison.png', dpi=600, bbox_inches='tight')
print("✓ Saved: power_spectrum_comparison.png")

# ============================================================================
# PLOT 3: C3 vs C4 Comparison (Motor Cortex Activity)
# ============================================================================

fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('C3 vs C4 Activity Comparison', fontsize=16, fontweight='bold')

for idx, (label, df) in enumerate(trials.items()):
    ax = axes[idx]
    
    # Get first 5 seconds
    c3 = df['C3_left_motor'].values[:plot_samples]
    c4 = df['C4_right_motor'].values[:plot_samples]
    time_axis = np.arange(len(c3)) / sample_rate
    
    # Normalize for better comparison
    c3_norm = (c3 - np.mean(c3)) / (np.std(c3) + 1e-8)
    c4_norm = (c4 - np.mean(c4)) / (np.std(c4) + 1e-8)
    
    # Plot
    ax.plot(time_axis, c3_norm, color='#FF6B6B', linewidth=1.5, label='C3 (Left Motor)', alpha=0.8)
    ax.plot(time_axis, c4_norm, color='#4ECDC4', linewidth=1.5, label='C4 (Right Motor)', alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Normalized Amplitude', fontsize=10)
    ax.set_title(f'{label} - Motor Cortex Activity', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('motor_cortex_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_cortex_comparison.png")

# ============================================================================
# PLOT 4: Band Power Analysis
# ============================================================================

fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('Frequency Band Power Analysis', fontsize=16, fontweight='bold')

bands = {
    'Mu (8-12 Hz)': (8, 12),
    'Beta (12-30 Hz)': (12, 30),
}

def bandpass_filter(data, lowcut, highcut, fs=250, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp_signal.butter(order, [low, high], btype='band')
    return sp_signal.filtfilt(b, a, data)

def calculate_band_power(data, lowcut, highcut, fs=250):
    filtered = bandpass_filter(data, lowcut, highcut, fs)
    power = np.mean(filtered ** 2)
    return power

# Calculate band powers
for channel_idx, channel in enumerate(['C3_left_motor', 'C4_right_motor']):
    ax = axes[channel_idx]
    
    band_names = list(bands.keys())
    x = np.arange(len(band_names))
    width = 0.25
    
    for trial_idx, (label, df) in enumerate(trials.items()):
        data = df[channel].values
        powers = [calculate_band_power(data, low, high) for low, high in bands.values()]
        
        offset = (trial_idx - 1) * width
        bars = ax.bar(x + offset, powers, width, label=label, color=colors[label], alpha=0.8)
    
    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Average Power (µV²)', fontsize=10)
    ax.set_title(channel.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('band_power_analysis.png', dpi=600, bbox_inches='tight')
print("✓ Saved: band_power_analysis.png")

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================

print("\n=== Signal Statistics ===")
for label, df in trials.items():
    print(f"\n{label}:")
    for channel in ['C3_left_motor', 'C4_right_motor', 'Cz_reference']:
        data = df[channel].values
        print(f"  {channel}:")
        print(f"    Mean: {np.mean(data):.2f} µV")
        print(f"    Std:  {np.std(data):.2f} µV")
        print(f"    Range: [{np.min(data):.2f}, {np.max(data):.2f}] µV")

print("\n=== All visualizations saved to /mnt/user-data/outputs/ ===")
print("Files created:")
print("  - eeg_signals_comparison.png")
print("  - power_spectrum_comparison.png")
print("  - motor_cortex_comparison.png")
print("  - band_power_analysis.png")

"""
BCI Competition IV 2a Compatible Data Collection Protocol
Matches the structure and timing of the competition dataset for model fine-tuning
"""

import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import json

# ============================================================================
# BCI COMPETITION IV 2A PROTOCOL SPECIFICATION
# ============================================================================

"""
Original BCI Competition IV 2a Protocol:
- 4 classes: Left Hand, Right Hand, Both Feet, Tongue
- 2 sessions per subject (training and evaluation)
- 6 runs per session
- 48 trials per run (12 per class)
- Total: 288 trials per session (72 per class)

Timing Protocol (Graz BCI paradigm):
- t=0s:     Fixation cross appears (preparation)
- t=2s:     Beep sound (cue warning)
- t=3s:     Cue appears (arrow: left/right/down/up)
- t=3-7s:   Motor imagery (4 seconds)
- t=7s:     Short break
- t=8s:     Next trial begins

We'll adapt this for OpenBCI with C3, C4, Cz channels
"""

class BCICompetitionConfig:
    """Configuration matching BCI Competition IV 2a"""
    
    # Hardware
    BOARD_ID = BoardIds.CYTON_WIFI_BOARD.value
    IP_ADDRESS = "192.168.4.1"
    IP_PORT = 3000
    SAMPLE_RATE = 250  # Hz (Competition used 250 Hz)
    
    # Protocol timing (in seconds)
    FIXATION_DURATION = 2.0      # Fixation cross
    CUE_WARNING_DURATION = 1.0   # Beep before cue
    MOTOR_IMAGERY_DURATION = 4.0 # Actual imagery task
    BREAK_DURATION = 1.0         # Short break
    
    # Total trial duration
    TRIAL_DURATION = FIXATION_DURATION + CUE_WARNING_DURATION + MOTOR_IMAGERY_DURATION + BREAK_DURATION
    
    # Dataset structure
    CLASSES = {
        'L': {'id': 0, 'name': 'left_hand', 'description': 'Left Hand', 'cue': '←'},
        'R': {'id': 1, 'name': 'right_hand', 'description': 'Right Hand', 'cue': '→'},
        'F': {'id': 2, 'name': 'both_feet', 'description': 'Both Feet', 'cue': '↓'},
        'T': {'id': 3, 'name': 'tongue', 'description': 'Tongue', 'cue': '↑'},
    }
    
    # Session structure
    TRIALS_PER_CLASS_PER_RUN = 12
    CLASSES_COUNT = 4
    TRIALS_PER_RUN = TRIALS_PER_CLASS_PER_RUN * CLASSES_COUNT  # 48 trials
    RUNS_PER_SESSION = 6
    TOTAL_TRIALS_PER_SESSION = TRIALS_PER_RUN * RUNS_PER_SESSION  # 288 trials
    
    # Subject and session naming
    SUBJECT_ID = "S01"
    SESSION_TYPE = "training"  # 'training' or 'evaluation'
    
    # Output
    SAVE_FOLDER = "bci_competition_dataset"
    
    # Electrode mapping (adapt to your OpenBCI setup)
    CHANNEL_NAMES = ['C3', 'C4', 'Cz']  # Primary motor channels

class TrialData:
    """Store trial information"""
    def __init__(self, trial_num, run_num, class_id, class_name):
        self.trial_num = trial_num
        self.run_num = run_num
        self.class_id = class_id
        self.class_name = class_name
        self.start_time = None
        self.end_time = None
        self.fixation_start = None
        self.cue_start = None
        self.imagery_start = None
        self.imagery_end = None
        self.data = None
        self.num_samples = 0

class BCICompetitionRecorder:
    """Record data following BCI Competition IV 2a protocol"""
    
    def __init__(self):
        self.config = BCICompetitionConfig()
        self.board = None
        self.session_folder = None
        self.run_trials = []
        self.all_trials_metadata = []
        
    def setup_board(self):
        """Initialize OpenBCI board"""
        print("="*70)
        print("BCI Competition IV 2a Compatible Data Collection")
        print("="*70)
        
        BoardShim.enable_dev_board_logger()
        
        params = BrainFlowInputParams()
        params.ip_address = self.config.IP_ADDRESS
        params.ip_port = self.config.IP_PORT
        
        self.board = BoardShim(self.config.BOARD_ID, params)
        
        print(f"\n[SETUP] Connecting to OpenBCI at {self.config.IP_ADDRESS}...")
        self.board.prepare_session()
        self.board.start_stream()
        
        # Verify streaming
        time.sleep(1)
        test_data = self.board.get_current_board_data(10)
        if test_data.shape[1] > 0:
            print(f"[SETUP] ✓ Board connected and streaming ({test_data.shape[1]} samples/sec)")
        else:
            print("[SETUP] ⚠ Warning: No data received in test")
        
    def create_session_folder(self):
        """Create folder structure matching competition format"""
        # Structure: bci_competition_dataset/S01/training/
        self.session_folder = Path(self.config.SAVE_FOLDER) / \
                             self.config.SUBJECT_ID / \
                             self.config.SESSION_TYPE
        self.session_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"[SETUP] Save folder: {self.session_folder}")
    
    def generate_run_sequence(self, run_number):
        """Generate pseudo-randomized trial sequence for a run"""
        # Create balanced sequence: 12 trials per class
        trials = []
        for class_key, class_info in self.config.CLASSES.items():
            for _ in range(self.config.TRIALS_PER_CLASS_PER_RUN):
                trials.append((class_info['id'], class_info['name']))
        
        # Shuffle (with seed for reproducibility)
        np.random.seed(run_number * 42)  # Different seed per run
        np.random.shuffle(trials)
        
        return trials
    
    def display_protocol_info(self):
        """Display protocol information before starting"""
        print("\n" + "="*70)
        print("PROTOCOL INFORMATION")
        print("="*70)
        print(f"Following BCI Competition IV 2a timing:")
        print(f"  0-2s:   Fixation cross (+)")
        print(f"  2-3s:   Beep (warning)")
        print(f"  3-7s:   Cue + Motor Imagery (4 seconds)")
        print(f"  7-8s:   Break")
        print(f"  Total:  {self.config.TRIAL_DURATION}s per trial")
        print(f"\nClasses:")
        for key, info in self.config.CLASSES.items():
            print(f"  [{key}] {info['description']:15} {info['cue']} (ID: {info['id']})")
        print(f"\nSession structure:")
        print(f"  {self.config.RUNS_PER_SESSION} runs × {self.config.TRIALS_PER_RUN} trials = {self.config.TOTAL_TRIALS_PER_SESSION} total trials")
        print(f"  ~{self.config.TOTAL_TRIALS_PER_SESSION * self.config.TRIAL_DURATION / 60:.0f} minutes total recording time")
        print("="*70)
    
    def record_trial(self, trial_num, run_num, class_id, class_name):
        """Record single trial following competition protocol"""
        trial = TrialData(trial_num, run_num, class_id, class_name)
        class_info = [v for v in self.config.CLASSES.values() if v['id'] == class_id][0]
        
        print(f"\n{'='*70}")
        print(f"Trial {trial_num}/48 in Run {run_num}")
        print(f"Class: {class_info['description']} {class_info['cue']}")
        print(f"{'='*70}")
        
        # Phase 1: Fixation (0-2s)
        print(f"[{0:.1f}s] Phase 1: Fixation cross (+)")
        print("         → Clear your mind, relax")
        trial.fixation_start = time.time()
        time.sleep(self.config.FIXATION_DURATION)
        
        # Phase 2: Cue warning (2-3s)
        print(f"[{self.config.FIXATION_DURATION:.1f}s] Phase 2: *BEEP* (Get ready!)")
        trial.cue_start = time.time()
        time.sleep(self.config.CUE_WARNING_DURATION)
        
        # Phase 3: Motor Imagery (3-7s)
        print(f"[{self.config.FIXATION_DURATION + self.config.CUE_WARNING_DURATION:.1f}s] Phase 3: CUE {class_info['cue']} → IMAGINE {class_info['description'].upper()}!")
        print(f"         ⚡ MOTOR IMAGERY NOW! (4 seconds)")
        
        # Clear buffer and start recording
        self.board.get_board_data()
        trial.imagery_start = time.time()
        trial.start_time = trial.imagery_start
        
        time.sleep(self.config.MOTOR_IMAGERY_DURATION)
        
        trial.imagery_end = time.time()
        
        # Get recorded data
        data = self.board.get_board_data()
        trial.data = data
        trial.num_samples = data.shape[1]
        
        print(f"[{self.config.FIXATION_DURATION + self.config.CUE_WARNING_DURATION + self.config.MOTOR_IMAGERY_DURATION:.1f}s] Phase 3: Complete")
        print(f"         Recorded {trial.num_samples} samples ({trial.num_samples/self.config.SAMPLE_RATE:.2f}s)")
        
        # Phase 4: Break (7-8s)
        print(f"[{self.config.FIXATION_DURATION + self.config.CUE_WARNING_DURATION + self.config.MOTOR_IMAGERY_DURATION:.1f}s] Phase 4: Break")
        time.sleep(self.config.BREAK_DURATION)
        
        trial.end_time = time.time()
        
        # Save trial
        self._save_trial(trial)
        
        return trial
    
    def _save_trial(self, trial):
        """Save trial data in competition-compatible format"""
        eeg_channels = BoardShim.get_eeg_channels(self.config.BOARD_ID)
        timestamp_channel = BoardShim.get_timestamp_channel(self.config.BOARD_ID)
        
        data = trial.data
        
        # Build dataframe
        df_data = {
            'timestamp': data[timestamp_channel],
            'sample_id': np.arange(trial.num_samples),
        }
        
        # Add EEG channels (map to C3, C4, Cz)
        for i, channel_name in enumerate(self.config.CHANNEL_NAMES):
            if i < len(eeg_channels):
                df_data[channel_name] = data[eeg_channels[i]]
        
        # Add metadata
        df_data['class_id'] = trial.class_id
        df_data['class_name'] = trial.class_name
        df_data['trial_num'] = trial.trial_num
        df_data['run_num'] = trial.run_num
        
        df = pd.DataFrame(df_data)
        
        # Filename: S01_training_run01_trial_001_left_hand.csv
        filename = f"{self.config.SUBJECT_ID}_{self.config.SESSION_TYPE}_" \
                  f"run{trial.run_num:02d}_trial_{trial.trial_num:03d}_{trial.class_name}.csv"
        filepath = self.session_folder / filename
        
        df.to_csv(filepath, index=False)
        print(f"         ✓ Saved: {filename}")
        
        # Store metadata
        trial_meta = {
            'trial_num': trial.trial_num,
            'run_num': trial.run_num,
            'class_id': trial.class_id,
            'class_name': trial.class_name,
            'num_samples': trial.num_samples,
            'duration': trial.num_samples / self.config.SAMPLE_RATE,
            'filename': filename,
            'imagery_start': trial.imagery_start,
            'imagery_end': trial.imagery_end,
        }
        self.all_trials_metadata.append(trial_meta)
    
    def record_run(self, run_number):
        """Record complete run (48 trials)"""
        print("\n" + "="*70)
        print(f"STARTING RUN {run_number}/{self.config.RUNS_PER_SESSION}")
        print("="*70)
        
        # Generate trial sequence
        sequence = self.generate_run_sequence(run_number)
        
        print(f"\nRun {run_number} sequence generated:")
        class_counts = {}
        for class_id, class_name in sequence:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} trials")
        
        input(f"\nPress ENTER to start Run {run_number}...")
        
        # Record all trials in sequence
        for trial_idx, (class_id, class_name) in enumerate(sequence, 1):
            self.record_trial(trial_idx, run_number, class_id, class_name)
        
        print(f"\n{'='*70}")
        print(f"✓ Run {run_number} complete! ({len(sequence)} trials)")
        print(f"{'='*70}")
        
        # Break between runs
        if run_number < self.config.RUNS_PER_SESSION:
            print(f"\n⏸  Take a 2-3 minute break before Run {run_number + 1}")
            print("   Relax, stretch, drink water...")
            input("   Press ENTER when ready to continue...")
    
    def record_session(self):
        """Record complete session (6 runs)"""
        self.display_protocol_info()
        
        print("\n" + "="*70)
        print(f"SESSION: {self.config.SUBJECT_ID} - {self.config.SESSION_TYPE.upper()}")
        print("="*70)
        print(f"You will record {self.config.RUNS_PER_SESSION} runs")
        print(f"Each run has {self.config.TRIALS_PER_RUN} trials")
        print(f"Total: {self.config.TOTAL_TRIALS_PER_SESSION} trials")
        print(f"\n⚠ Important:")
        print(f"  - Minimize movement during trials")
        print(f"  - Focus during motor imagery")
        print(f"  - Take breaks between runs")
        print(f"  - Session will take ~{self.config.TOTAL_TRIALS_PER_SESSION * self.config.TRIAL_DURATION / 60:.0f} minutes")
        print("="*70)
        
        input("\nPress ENTER to begin session...")
        
        # Record all runs
        for run_num in range(1, self.config.RUNS_PER_SESSION + 1):
            self.record_run(run_num)
        
        # Save session metadata
        self._save_session_metadata()
        
        print("\n" + "="*70)
        print("✓✓✓ SESSION COMPLETE! ✓✓✓")
        print("="*70)
        print(f"Total trials recorded: {len(self.all_trials_metadata)}")
        print(f"Data saved to: {self.session_folder}")
        print("="*70)
    
    def _save_session_metadata(self):
        """Save session metadata in competition format"""
        metadata = {
            'subject_id': self.config.SUBJECT_ID,
            'session_type': self.config.SESSION_TYPE,
            'protocol': 'BCI Competition IV 2a compatible',
            'sample_rate': self.config.SAMPLE_RATE,
            'channels': self.config.CHANNEL_NAMES,
            'classes': {k: {'id': v['id'], 'name': v['name'], 'description': v['description']} 
                       for k, v in self.config.CLASSES.items()},
            'timing': {
                'fixation': self.config.FIXATION_DURATION,
                'cue_warning': self.config.CUE_WARNING_DURATION,
                'motor_imagery': self.config.MOTOR_IMAGERY_DURATION,
                'break': self.config.BREAK_DURATION,
                'total_trial': self.config.TRIAL_DURATION,
            },
            'structure': {
                'runs': self.config.RUNS_PER_SESSION,
                'trials_per_run': self.config.TRIALS_PER_RUN,
                'total_trials': self.config.TOTAL_TRIALS_PER_SESSION,
            },
            'trials': self.all_trials_metadata,
            'class_distribution': self._get_class_distribution(),
        }
        
        metadata_file = self.session_folder / f"{self.config.SUBJECT_ID}_{self.config.SESSION_TYPE}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadata saved: {metadata_file}")
    
    def _get_class_distribution(self):
        """Count trials per class"""
        dist = {}
        for trial in self.all_trials_metadata:
            name = trial['class_name']
            dist[name] = dist.get(name, 0) + 1
        return dist
    
    def cleanup(self):
        """Clean shutdown"""
        if self.board and self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
            print("\n[CLEANUP] Board disconnected")

def main():
    """Entry point"""
    recorder = BCICompetitionRecorder()
    
    try:
        recorder.setup_board()
        recorder.create_session_folder()
        recorder.record_session()
    except KeyboardInterrupt:
        print("\n\n⚠ Session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main()

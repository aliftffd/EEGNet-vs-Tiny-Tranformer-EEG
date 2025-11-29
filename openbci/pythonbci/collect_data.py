"""
Robust OpenBCI EEG Dataset Collection Script
Optimized for Motor Imagery BCI with C3, Cz, C4 electrode configuration
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    # Hardware
    BOARD_ID = BoardIds.CYTON_WIFI_BOARD.value
    IP_ADDRESS = "192.168.4.1"
    IP_PORT = 3000  # Changed from 12345 to match WiFi Shield
    
    # Recording parameters
    DURATION_BASELINE = 3     # Baseline recording before trial (seconds)
    DURATION_CUE = 2          # Visual/audio cue duration
    DURATION_IMAGERY = 5      # Motor imagery duration
    DURATION_REST = 3         # Rest between trials
    
    # Data quality
    MIN_SAMPLES_REQUIRED = 100  # Minimum samples for valid trial
    EXPECTED_SAMPLE_RATE = 250  # Hz
    
    # Dataset
    SUBJECT_NAME = "S01"
    SESSION_ID = "session_01"
    SAVE_FOLDER = "motor_imagery_dataset"
    
    # Electrode mapping (adjust based on your physical setup)
    ELECTRODE_MAP = {
        0: 'C3_left_motor',    # Pin 1 (N1P) -> C3
        1: 'C4_right_motor',   # Pin 2 (N2P) -> C4
        2: 'Cz_reference',     # Pin 3 (N3P) -> Cz (if used as active)
        3: 'F3_frontal',
        4: 'F4_frontal',
        5: 'P3_parietal',
        6: 'P4_parietal',
        7: 'O1_occipital',
    }
    
    # Class labels
    CLASSES = {
        'l': {'name': 'left_hand', 'id': 0, 'description': 'Imagine moving LEFT hand'},
        'r': {'name': 'right_hand', 'id': 1, 'description': 'Imagine moving RIGHT hand'},
        'b': {'name': 'both_hands', 'id': 2, 'description': 'Imagine moving BOTH hands'},
        'f': {'name': 'feet', 'id': 3, 'description': 'Imagine moving FEET'},
        'x': {'name': 'rest', 'id': 4, 'description': 'Rest - no imagery'},
    }

# ============================================================================
# UTILITIES
# ============================================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_info(text):
    """Print info message"""
    print(f"[INFO] {text}")

def print_warning(text):
    """Print warning message"""
    print(f"[WARN] {text}")

def print_error(text):
    """Print error message"""
    print(f"[ERROR] {text}")

def countdown(seconds, message="Starting in"):
    """Visual countdown"""
    print(f"\n{message}:")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end='\r')
        time.sleep(1)
    print("  GO!   ")

# ============================================================================
# SESSION METADATA
# ============================================================================

class SessionMetadata:
    """Track and save session information"""
    
    def __init__(self, subject_name, session_id, save_folder):
        self.subject_name = subject_name
        self.session_id = session_id
        self.save_folder = save_folder
        self.start_time = datetime.now()
        self.trials = []
        self.total_samples = 0
        
    def add_trial(self, label, num_samples, duration, quality_score, filename):
        """Record trial information"""
        self.trials.append({
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'class_id': Config.CLASSES.get(label[0], {}).get('id', -1),
            'num_samples': num_samples,
            'duration': duration,
            'quality_score': quality_score,
            'filename': filename
        })
        self.total_samples += num_samples
    
    def save(self, session_folder):
        """Save session metadata to JSON"""
        metadata = {
            'subject_name': self.subject_name,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_trials': len(self.trials),
            'total_samples': self.total_samples,
            'sample_rate': Config.EXPECTED_SAMPLE_RATE,
            'electrode_config': {
                'channels': list(Config.ELECTRODE_MAP.values()),
                'reference': 'Cz (SRB pin)',
                'ground': 'Fpz (BIAS pin)',
            },
            'class_distribution': self._get_class_distribution(),
            'trials': self.trials
        }
        
        metadata_path = session_folder / f"{self.subject_name}_{self.session_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print_info(f"Metadata saved: {metadata_path}")
    
    def _get_class_distribution(self):
        """Count trials per class"""
        distribution = {}
        for trial in self.trials:
            label = trial['label']
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def print_summary(self):
        """Print session summary"""
        print_header("SESSION SUMMARY")
        print(f"Subject: {self.subject_name}")
        print(f"Session: {self.session_id}")
        print(f"Total Trials: {len(self.trials)}")
        print(f"Total Samples: {self.total_samples}")
        print(f"\nClass Distribution:")
        for label, count in self._get_class_distribution().items():
            print(f"  {label}: {count} trials")

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

class DataQualityChecker:
    """Check EEG data quality"""
    
    @staticmethod
    def check_sample_count(num_samples, expected_samples):
        """Verify sufficient samples collected"""
        if num_samples < Config.MIN_SAMPLES_REQUIRED:
            print_warning(f"Low sample count: {num_samples} (min: {Config.MIN_SAMPLES_REQUIRED})")
            return False
        
        expected = Config.EXPECTED_SAMPLE_RATE * Config.DURATION_IMAGERY
        if num_samples < expected * 0.8:  # Allow 20% tolerance
            print_warning(f"Sample count lower than expected: {num_samples}/{expected}")
        
        return True
    
    @staticmethod
    def check_signal_quality(data, eeg_channels):
        """Basic signal quality metrics"""
        quality_scores = []
        
        for ch_idx in eeg_channels[:3]:  # Check C3, C4, Cz
            channel_data = data[ch_idx]
            
            # Check for flat signal
            if np.std(channel_data) < 0.1:
                print_warning(f"Channel {ch_idx} appears flat (std: {np.std(channel_data):.3f})")
                quality_scores.append(0.0)
                continue
            
            # Check for saturation
            if np.max(np.abs(channel_data)) > 180000:  # OpenBCI range is ±187500 µV
                print_warning(f"Channel {ch_idx} may be saturated")
                quality_scores.append(0.5)
                continue
            
            # Check for excessive noise (simple heuristic)
            high_freq_noise = np.sum(np.abs(np.diff(channel_data)) > 1000) / len(channel_data)
            if high_freq_noise > 0.1:
                print_warning(f"Channel {ch_idx} has high noise: {high_freq_noise:.2%}")
                quality_scores.append(0.7)
            else:
                quality_scores.append(1.0)
        
        avg_quality = np.mean(quality_scores)
        print_info(f"Signal quality score: {avg_quality:.2f}/1.00")
        
        return avg_quality
    
    @staticmethod
    def remove_artifacts(data, eeg_channels, sample_rate):
        """Apply basic artifact removal"""
        cleaned_data = data.copy()
        
        for ch_idx in eeg_channels:
            # Detrend (remove DC offset and linear trend)
            DataFilter.detrend(cleaned_data[ch_idx], DetrendOperations.LINEAR.value)
            
            # Optional: Notch filter for 50/60 Hz powerline noise
            # DataFilter.perform_bandstop(cleaned_data[ch_idx], sample_rate, 50.0, 4.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        
        return cleaned_data

# ============================================================================
# TRIAL RECORDER
# ============================================================================

class TrialRecorder:
    """Handle individual trial recording"""
    
    def __init__(self, board, session_folder, metadata):
        self.board = board
        self.session_folder = session_folder
        self.metadata = metadata
        self.board_id = Config.BOARD_ID
        
    def record_trial(self, label_key):
        """Record a single trial with full protocol"""
        class_info = Config.CLASSES[label_key]
        label = class_info['name']
        
        print_header(f"TRIAL: {class_info['description'].upper()}")
        
        # Phase 1: Baseline
        print_info("Phase 1: Baseline recording")
        countdown(Config.DURATION_BASELINE, "Relax and clear your mind")
        self.board.get_board_data()  # Clear buffer
        time.sleep(Config.DURATION_BASELINE)
        
        # Phase 2: Cue
        print_info(f"Phase 2: Cue - {class_info['description']}")
        countdown(Config.DURATION_CUE, "Prepare for imagery")
        
        # Phase 3: Motor Imagery
        print_info("Phase 3: MOTOR IMAGERY - Start imagining NOW!")
        self.board.get_board_data()  # Clear buffer before recording
        
        start_time = time.time()
        time.sleep(Config.DURATION_IMAGERY)
        actual_duration = time.time() - start_time
        
        # Fetch recorded data
        data = self.board.get_board_data()  # shape: [channels, samples]
        
        print_info(f"Phase 3 complete - Duration: {actual_duration:.2f}s")
        
        # Phase 4: Rest
        print_info("Phase 4: Rest")
        time.sleep(Config.DURATION_REST)
        
        # Save trial
        if data.shape[1] > 0:
            self._save_trial(data, label, actual_duration)
        else:
            print_error("No data collected! Check board connection.")
            return False
        
        return True
    
    def _save_trial(self, data, label, duration):
        """Save trial data to CSV"""
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        
        num_samples = data.shape[1]
        
        # Quality checks
        if not DataQualityChecker.check_sample_count(num_samples, Config.EXPECTED_SAMPLE_RATE * duration):
            response = input("Low sample count. Save anyway? (y/n): ").lower()
            if response != 'y':
                print_info("Trial discarded")
                return
        
        # Clean data
        cleaned_data = DataQualityChecker.remove_artifacts(
            data, eeg_channels, Config.EXPECTED_SAMPLE_RATE
        )
        
        # Check quality
        quality_score = DataQualityChecker.check_signal_quality(cleaned_data, eeg_channels)
        
        # Build DataFrame
        df_data = {'timestamp': cleaned_data[timestamp_channel]}
        
        for ch_idx, ch_name in Config.ELECTRODE_MAP.items():
            if ch_idx < len(eeg_channels):
                df_data[ch_name] = cleaned_data[eeg_channels[ch_idx]]
        
        # Add metadata columns
        df_data['label'] = label
        df_data['class_id'] = Config.CLASSES.get(label[0], {}).get('id', -1)
        df_data['trial_duration'] = duration
        df_data['quality_score'] = quality_score
        
        df = pd.DataFrame(df_data)
        
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_number = len(self.metadata.trials) + 1
        filename = f"{Config.SUBJECT_NAME}_{label}_trial_{trial_number:03d}_{timestamp_str}.csv"
        filepath = self.session_folder / filename
        
        # Save
        df.to_csv(filepath, index=False)
        print_info(f"Saved: {filepath}")
        print_info(f"  Samples: {num_samples}, Rate: {num_samples/duration:.1f} Hz, Quality: {quality_score:.2f}")
        
        # Update metadata
        self.metadata.add_trial(label, num_samples, duration, quality_score, filename)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class OpenBCIRecorder:
    """Main application class"""
    
    def __init__(self):
        self.board = None
        self.session_folder = None
        self.metadata = None
        
    def setup(self):
        """Initialize board and session"""
        print_header("OpenBCI Motor Imagery Dataset Recorder")
        print_info(f"Subject: {Config.SUBJECT_NAME}")
        print_info(f"Session: {Config.SESSION_ID}")
        print_info(f"Board IP: {Config.IP_ADDRESS}")
        
        # Create folder structure
        self.session_folder = Path(Config.SAVE_FOLDER) / Config.SUBJECT_NAME / Config.SESSION_ID
        self.session_folder.mkdir(parents=True, exist_ok=True)
        print_info(f"Save folder: {self.session_folder}")
        
        # Initialize metadata
        self.metadata = SessionMetadata(Config.SUBJECT_NAME, Config.SESSION_ID, Config.SAVE_FOLDER)
        
        # Setup board
        BoardShim.enable_dev_board_logger()
        
        params = BrainFlowInputParams()
        params.ip_address = Config.IP_ADDRESS
        params.ip_port = Config.IP_PORT
        
        self.board = BoardShim(Config.BOARD_ID, params)
        
        print_info("Connecting to OpenBCI...")
        try:
            self.board.prepare_session()
            self.board.start_stream()
            print_info("Connected successfully!")
            
            # Verify data is flowing
            time.sleep(1)
            test_data = self.board.get_current_board_data(10)
            if test_data.shape[1] > 0:
                print_info(f"Data streaming OK (sample rate check: {test_data.shape[1]} samples/sec)")
            else:
                print_warning("No data received in test")
                
        except Exception as e:
            print_error(f"Failed to connect: {e}")
            raise
    
    def run(self):
        """Main recording loop"""
        recorder = TrialRecorder(self.board, self.session_folder, self.metadata)
        
        print_header("ELECTRODE CONFIGURATION")
        print("Pin configuration (verify your setup):")
        for pin, name in Config.ELECTRODE_MAP.items():
            print(f"  Pin {pin+1}: {name}")
        print("\nReference: Cz (SRB pin)")
        print("Ground: Fpz (BIAS pin)")
        
        input("\nPress ENTER when electrodes are ready...")
        
        try:
            while True:
                print_header("TRIAL MENU")
                print(f"Subject: {Config.SUBJECT_NAME} | Session: {Config.SESSION_ID}")
                print(f"Trials completed: {len(self.metadata.trials)}\n")
                
                print("Motor Imagery Classes:")
                for key, info in Config.CLASSES.items():
                    print(f"  [{key.upper()}] {info['description']}")
                print("\n  [S] Show session summary")
                print("  [Q] Quit and save")
                
                command = input("\nEnter command: ").lower().strip()
                
                if command == 'q':
                    break
                elif command == 's':
                    self.metadata.print_summary()
                    continue
                elif command in Config.CLASSES:
                    recorder.record_trial(command)
                else:
                    print_warning("Invalid command")
                    
        except KeyboardInterrupt:
            print_info("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        print_header("SHUTTING DOWN")
        
        if self.board and self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
            print_info("Board disconnected")
        
        if self.metadata:
            self.metadata.save(self.session_folder)
            self.metadata.print_summary()
        
        print_info("Session complete!")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Entry point"""
    try:
        app = OpenBCIRecorder()
        app.setup()
        app.run()
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

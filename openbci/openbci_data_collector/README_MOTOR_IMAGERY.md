# OpenBCI Motor Imagery Data Collector

Deep learning-optimized data collector for motor imagery BCI experiments.

## Features

- **Structured Output**: Organized by subject → session → trials
- **Class Labels**: Automatic numeric encoding for deep learning
- **Trial-based Collection**: Easy repetition tracking
- **CSV Format**: Ready for PyTorch/TensorFlow DataLoaders

## Motor Imagery Classes

| Class Label  | Class ID | Description                    |
|--------------|----------|--------------------------------|
| `left_hand`  | 0        | Imagine moving left hand       |
| `right_hand` | 1        | Imagine moving right hand      |
| `both_hands` | 2        | Imagine moving both hands      |
| `rest`       | 3        | Resting state / no imagination |

## Quick Start

### Single Trial Collection

```bash
./collect_trial.sh left_hand 1
```

This will record 5 seconds of EEG while you imagine moving your left hand.

### Full Session Collection

Collect multiple trials for all classes:

```bash
./collect_session.sh S01 session_01 10
```

This collects 10 trials per class (30 trials total) for subject S01.

## Manual Collection

For custom configurations:

```bash
cargo run --release -- \
  --class left_hand \
  --trial 1 \
  --subject-id S01 \
  --session-id session_01 \
  --duration 5 \
  --channels 2
```

### Options

- `--class`: Motor imagery class (left_hand, right_hand, both_hands, rest)
- `--trial`: Trial number (for organizing repetitions)
- `--subject-id`: Subject identifier (default: S01)
- `--session-id`: Session identifier (default: session_01)
- `--duration`: Recording duration in seconds (default: 5)
- `--channels`: Number of EEG channels (default: 2)

## Output Structure

```
motor_imagery_data/
└── S01/
    └── session_01/
        ├── S01_left_hand_session_01_trial_01_class_0_20250128_143022.csv
        ├── S01_left_hand_session_01_trial_01_class_0_metadata.json
        ├── S01_right_hand_session_01_trial_02_class_1_20250128_143035.csv
        ├── S01_right_hand_session_01_trial_02_class_1_metadata.json
        └── ...
```

## CSV Format

Each CSV file contains:

```csv
timestamp,sample_id,class_id,C3_left_motor,C4_right_motor
1234567890.123,0,0,12.5,15.3
1234567890.127,1,0,12.6,15.4
...
```

- `timestamp`: Unix timestamp with milliseconds
- `sample_id`: Sequential sample number
- `class_id`: Numeric class label (0-3)
- Channel columns: EEG data in microvolts

## Metadata JSON

Each trial includes a metadata file:

```json
{
  "subject_id": "S01",
  "session_id": "session_01",
  "trial_number": 1,
  "class_label": "left_hand",
  "class_id": 0,
  "start_time": "2025-01-28T14:30:22Z",
  "end_time": "2025-01-28T14:30:27Z",
  "sample_rate": 250,
  "num_channels": 2,
  "total_samples": 1250,
  "duration_seconds": 5,
  "electrode_config": {
    "channels": ["C3_left_motor", "C4_right_motor"],
    "reference": "Cz",
    "ground": "Fpz"
  }
}
```

## Loading Data in Python

### Using Pandas

```python
import pandas as pd
import glob

# Load all trials for a subject
files = glob.glob("motor_imagery_data/S01/session_01/*_class_*.csv")
df = pd.concat([pd.read_csv(f) for f in files])

# Separate features and labels
X = df[['C3_left_motor', 'C4_right_motor']].values
y = df['class_id'].values
```

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MotorImageryDataset(Dataset):
    def __init__(self, csv_files):
        self.data = []
        self.labels = []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Extract EEG channels
            channels = df[['C3_left_motor', 'C4_right_motor']].values
            label = df['class_id'].iloc[0]

            self.data.append(torch.FloatTensor(channels))
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create DataLoader
files = glob.glob("motor_imagery_data/S01/session_01/*.csv")
dataset = MotorImageryDataset(files)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Best Practices

### Data Collection Protocol

1. **Environment Setup**
   - Quiet room
   - Comfortable seating
   - Minimize movement during trials

2. **Electrode Placement**
   - C3: Left motor cortex (controls right hand)
   - C4: Right motor cortex (controls left hand)
   - Cz: Reference
   - Fpz: Ground

3. **Trial Structure**
   - 5 seconds per trial
   - Clear mental imagery
   - 3-5 second rest between trials
   - Breaks between classes

4. **Data Quality**
   - Collect 10-20 trials per class minimum
   - Multiple sessions on different days
   - Validate signal quality before starting

### Recommended Dataset Size

For good deep learning performance:
- **Minimum**: 10 trials/class × 3 classes = 30 trials
- **Good**: 20 trials/class × 4 classes = 80 trials
- **Excellent**: 50+ trials/class across multiple sessions

## Troubleshooting

### Connection Issues

If you see "504 Gateway Timeout":
1. Check WiFi connection to OpenBCI shield
2. Verify shield IP: `curl http://192.168.4.1/board`
3. Ensure local IP is correct: `ip addr show wlan1`

### Data Quality

- Check impedance before recording
- Verify electrode placement
- Minimize EMG artifacts (relax muscles)
- Practice mental imagery before recording

## Next Steps

1. Collect initial dataset (30+ trials)
2. Preprocess data (filtering, normalization)
3. Train deep learning model (CNN, LSTM, EEGNet)
4. Validate with cross-validation
5. Test real-time classification

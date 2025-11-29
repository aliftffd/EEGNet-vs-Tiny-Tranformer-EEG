# Quick Start Guide - Motor Imagery Data Collection

## Setup (One-time)

1. **Connect to OpenBCI WiFi Shield**
   ```bash
   # Connect to OpenBCI-XXXX WiFi network
   # Password: (default OpenBCI password)
   ```

2. **Verify Connection**
   ```bash
   curl http://192.168.4.1/board
   # Should show: {"board_connected":true,...}
   ```

3. **Build the Collector**
   ```bash
   cargo build --release
   ```

## Collecting Data

### Option 1: Single Trial (Quick Test)

```bash
./collect_trial.sh left_hand 1
```

Start imagining left hand movement when prompted. Records 5 seconds.

### Option 2: Full Session (Recommended)

```bash
./collect_session.sh S01 session_01 10
```

This collects 10 trials each for:
- Left hand imagery
- Right hand imagery
- Rest/baseline

Total: 30 trials in one session

### Option 3: Manual Control

```bash
cargo run --release -- \
  --class left_hand \
  --trial 1 \
  --subject-id S01 \
  --session-id session_01 \
  --duration 5
```

## After Collection

### View Dataset Summary

```bash
./dataset_summary.sh
```

Shows:
- Number of subjects
- Trials per class
- Total dataset size

### Check Output

```bash
ls motor_imagery_data/S01/session_01/
```

You'll see:
- CSV files with EEG data
- JSON metadata for each trial

## Recommended Workflow

### Day 1: Setup & Practice
1. Connect hardware
2. Test with 3-5 trials: `./collect_trial.sh left_hand 1`
3. Verify data quality
4. Practice mental imagery

### Day 2-4: Data Collection
1. Run full session: `./collect_session.sh S01 session_01 20`
2. Take breaks between classes
3. Collect 20+ trials per class
4. Multiple sessions = better model

### Day 5: Dataset Review
1. Run: `./dataset_summary.sh`
2. Check class balance
3. Verify file integrity
4. Ready for training!

## Tips

**Mental Imagery:**
- **Left hand**: Imagine clenching/opening LEFT fist
- **Right hand**: Imagine clenching/opening RIGHT fist
- **Rest**: Clear mind, relax

**Signal Quality:**
- Sit still during trials
- Relax facial muscles
- Consistent imagery each trial
- Same electrode placement each session

**Dataset Size:**
- Minimum: 10 trials/class (30 total)
- Good: 20 trials/class (60 total)
- Great: 50+ trials/class (150+ total)

## Troubleshooting

**"504 Gateway Timeout"**
```bash
# Check connection
ping 192.168.4.1

# Check IP
ip addr show wlan1
```

**Port already in use**
```bash
# Kill existing process
pkill openbci_data_collector
```

**Need help**
```bash
cargo run --release -- --help
```

## Example Session

```bash
# Session 1
./collect_session.sh S01 session_01 15

# Break (different day)

# Session 2 (more data for same subject)
./collect_session.sh S01 session_02 15

# Check total
./dataset_summary.sh
# Output: S01 has 90 trials (30 trials × 3 classes × 2 sessions)
```

## Next Steps

See `README_MOTOR_IMAGERY.md` for:
- Data format details
- Python loading examples
- Deep learning integration
- Best practices

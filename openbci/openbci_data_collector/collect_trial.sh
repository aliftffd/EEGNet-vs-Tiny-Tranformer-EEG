#!/bin/bash
# Motor Imagery Data Collection Script
# Usage: ./collect_trial.sh <class> <trial_number> [subject_id] [session_id]

CLASS=$1
TRIAL=$2
SUBJECT=${3:-S01}
SESSION=${4:-session_01}

if [ -z "$CLASS" ] || [ -z "$TRIAL" ]; then
    echo "Usage: $0 <class> <trial_number> [subject_id] [session_id]"
    echo ""
    echo "Classes:"
    echo "  left_hand  - Imagine moving left hand (Class 0)"
    echo "  right_hand - Imagine moving right hand (Class 1)"
    echo "  both_hands - Imagine moving both hands (Class 2)"
    echo "  rest       - Resting state / no imagery (Class 3)"
    echo ""
    echo "Example: $0 left_hand 1"
    echo "Example: $0 right_hand 5 S02 session_02"
    exit 1
fi

echo "========================================="
echo "Motor Imagery Trial Collection"
echo "========================================="
echo "Subject:  $SUBJECT"
echo "Session:  $SESSION"
echo "Class:    $CLASS"
echo "Trial:    $TRIAL"
echo "========================================="
echo ""
echo "Starting in 3 seconds..."
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1
echo "GO! Start imagining the movement..."
echo ""

cargo run --release -- \
    --class "$CLASS" \
    --trial "$TRIAL" \
    --subject-id "$SUBJECT" \
    --session-id "$SESSION" \
    --duration 5 \
    --channels 2

echo ""
echo "Trial complete!"

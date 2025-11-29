#!/bin/bash
# Batch Motor Imagery Data Collection
# Collects multiple trials for each class with rest periods

SUBJECT=${1:-S01}
SESSION=${2:-session_01}
TRIALS_PER_CLASS=${3:-10}
REST_BETWEEN_TRIALS=3

echo "========================================="
echo "Motor Imagery Session Collection"
echo "========================================="
echo "Subject:          $SUBJECT"
echo "Session:          $SESSION"
echo "Trials per class: $TRIALS_PER_CLASS"
echo "========================================="
echo ""
read -p "Press Enter to start data collection..."

# Classes for motor imagery
CLASSES=("left_hand" "right_hand" "rest")

for class in "${CLASSES[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting trials for: $class"
    echo "========================================"
    sleep 2

    for trial in $(seq 1 $TRIALS_PER_CLASS); do
        echo ""
        echo "--- Class: $class | Trial: $trial/$TRIALS_PER_CLASS ---"

        ./collect_trial.sh "$class" "$trial" "$SUBJECT" "$SESSION"

        if [ $trial -lt $TRIALS_PER_CLASS ]; then
            echo ""
            echo "Rest for $REST_BETWEEN_TRIALS seconds..."
            sleep $REST_BETWEEN_TRIALS
        fi
    done

    echo ""
    echo "Completed all trials for $class"
    echo "Take a break before next class..."
    read -p "Press Enter to continue to next class..."
done

echo ""
echo "========================================="
echo "Session Complete!"
echo "========================================="
echo "Collected data saved in:"
echo "  motor_imagery_data/$SUBJECT/$SESSION/"
echo ""
echo "Total trials collected: $((${#CLASSES[@]} * TRIALS_PER_CLASS))"

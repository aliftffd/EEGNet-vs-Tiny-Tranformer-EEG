#!/bin/bash
# Dataset Summary Tool
# Shows statistics about collected motor imagery data

DATA_DIR=${1:-motor_imagery_data}

echo "========================================="
echo "Motor Imagery Dataset Summary"
echo "========================================="
echo ""

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR not found"
    exit 1
fi

echo "Directory: $DATA_DIR"
echo ""

# Count subjects
SUBJECTS=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Subjects: $SUBJECTS"
echo ""

# For each subject
for subject_dir in "$DATA_DIR"/*; do
    if [ -d "$subject_dir" ]; then
        subject=$(basename "$subject_dir")
        echo "--- $subject ---"

        # Count sessions
        sessions=$(find "$subject_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  Sessions: $sessions"

        # For each session
        for session_dir in "$subject_dir"/*; do
            if [ -d "$session_dir" ]; then
                session=$(basename "$session_dir")
                echo "  └── $session"

                # Count trials per class
                for class_id in 0 1 2 3; do
                    class_name=""
                    case $class_id in
                        0) class_name="left_hand" ;;
                        1) class_name="right_hand" ;;
                        2) class_name="both_hands" ;;
                        3) class_name="rest" ;;
                    esac

                    count=$(find "$session_dir" -name "*_class_${class_id}_*.csv" | wc -l)
                    if [ $count -gt 0 ]; then
                        echo "      Class $class_id ($class_name): $count trials"
                    fi
                done

                # Total trials
                total=$(find "$session_dir" -name "*_class_*.csv" | wc -l)
                echo "      Total trials: $total"
                echo ""
            fi
        done
    fi
done

# Overall statistics
echo "========================================="
echo "Overall Dataset Statistics"
echo "========================================="
total_trials=$(find "$DATA_DIR" -name "*_class_*.csv" | wc -l)
total_metadata=$(find "$DATA_DIR" -name "*_metadata.json" | wc -l)
echo "Total trials: $total_trials"
echo "Total metadata files: $total_metadata"

# Class distribution
echo ""
echo "Class Distribution:"
for class_id in 0 1 2 3; do
    class_name=""
    case $class_id in
        0) class_name="left_hand" ;;
        1) class_name="right_hand" ;;
        2) class_name="both_hands" ;;
        3) class_name="rest" ;;
    esac

    count=$(find "$DATA_DIR" -name "*_class_${class_id}_*.csv" | wc -l)
    echo "  Class $class_id ($class_name): $count trials"
done

echo ""
echo "========================================="

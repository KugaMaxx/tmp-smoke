#!/bin/bash
# Simple batch FDS runner
# Usage: ./batch_run.sh <base_directory>

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

# Process train cases
if [ -d "$BASE_DIR/train" ]; then
    echo "Processing train cases..."
    for case_dir in "$BASE_DIR/train"/*; do
        if [ -d "$case_dir" ]; then
            echo "Running: $(basename "$case_dir")"
            bash run_fds.sh "$case_dir"
        fi
    done
fi

# Process validation cases
if [ -d "$BASE_DIR/validation" ]; then
    echo "Processing validation cases..."
    for case_dir in "$BASE_DIR/validation"/*; do
        if [ -d "$case_dir" ]; then
            echo "Running: $(basename "$case_dir")"
            bash run_fds.sh "$case_dir"
        fi
    done
fi

echo "All cases submitted!"
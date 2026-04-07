#!/bin/bash
# Post-round hook: update state and log activity after each optimization round
# Usage: post-round.sh <state_json_path> <worker_id> <round_num> <status>

set -e

STATE_PATH="$1"
WORKER_ID="$2"
ROUND_NUM="$3"
STATUS="$4"

if [ -z "$STATE_PATH" ] || [ -z "$WORKER_ID" ]; then
    echo "Usage: post-round.sh <state_path> <worker_id> <round_num> <status>"
    exit 1
fi

echo "Post-round hook: worker=$WORKER_ID round=$ROUND_NUM status=$STATUS"

# Verify round output files exist
ROUND_DIR="$(dirname "$STATE_PATH")/${WORKER_ID}/round-${ROUND_NUM}"
for f in report.md accuracy.log; do
    if [ ! -f "$ROUND_DIR/$f" ]; then
        echo "WARNING: Missing $ROUND_DIR/$f"
    fi
done

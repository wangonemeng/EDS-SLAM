#!/bin/bash

LOG_DUR=500
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
OUTPUT_DIR="${SCRIPT_DIR}/../Log/viral_eval"
fullfilename=$(basename "$1")
IFS='.' read -ra parts <<<"$fullfilename"
filename="${parts[0]}"

cleanup() {
    echo "Interrupt received. Cleaning up..."
    kill "$PID1" 2>/dev/null
    kill "$PID2" 2>/dev/null
    wait "$PID1" 2>/dev/null
    wait "$PID2" 2>/dev/null
    exit 1
}

# Trap SIGINT (Ctrl+C) to call cleanup function
trap cleanup SIGINT

rm -r "${OUTPUT_DIR}/result_${filename}/predict_odom.csv"
rm -r "${OUTPUT_DIR}/result_${filename}/leica_pose.csv"

timeout $LOG_DUR rostopic echo -p --nostr --noarr /Odometry >"${OUTPUT_DIR}/result_${filename}/predict_odom.csv" &
PID1=$!
timeout $LOG_DUR rostopic echo -p --nostr --noarr /leica/pose/relative >"${OUTPUT_DIR}/result_${filename}/leica_pose.csv" &
PID2=$!

# Play the rosbag in the foreground
rosbag play -r 2 "$1"

# Wait for all background processes to finish
wait "$PID1"
wait "$PID2"

echo "All processes have been completed."

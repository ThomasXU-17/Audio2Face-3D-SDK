#!/bin/bash
set -e

# At least one arg should be passed
if [ -z "$1" ]; then
    echo "Usage: test_benchmark.sh <executable> [args...]"
    exit 1
fi

# Define the list of arguments
args=(
    "--benchmark_filter=BM_InteractiveExecutorBatch/BatchSize:0/InferencesToSkip:0/Classifier:1/real_time"
    "--benchmark_filter=BM_InteractiveExecutorLayer/Layer:0/Classifier:1/real_time"
    "--benchmark_filter=BM_ExecutorPartial/ActiveTracks:1/iterations:10/real_time"
)

# Iterate over the arguments and execute the benchmark command
for current_arg in "${args[@]}"; do
    echo "Running: $@ $current_arg"
    bash "$@" "$current_arg"
done

#!/bin/bash
# Pre-benchmark environment check
# Validates GPU availability and environment before running benchmarks

set -e

GPU_ID="${HIP_VISIBLE_DEVICES:-0}"

# Check GPU is accessible
if ! rocm-smi --showid -d "$GPU_ID" &>/dev/null 2>&1; then
    echo "WARNING: GPU $GPU_ID may not be accessible via rocm-smi"
fi

# Check no other heavy processes on this GPU
GPU_UTIL=$(rocm-smi --showuse -d "$GPU_ID" 2>/dev/null | grep -oP '\d+(?=%)' | head -1 || echo "0")
if [ "${GPU_UTIL:-0}" -gt 50 ]; then
    echo "WARNING: GPU $GPU_ID utilization at ${GPU_UTIL}% — benchmark results may be unreliable"
fi

echo "Pre-benchmark check passed for GPU $GPU_ID"

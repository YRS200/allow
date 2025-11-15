#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-experiments/results_kitti}
mkdir -p "$OUTPUT_DIR"
python -m KalmanAttentionMOT.main eval --config configs/kitti_kalman_attn.yaml --output "$OUTPUT_DIR"

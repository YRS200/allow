#!/usr/bin/env bash
set -euo pipefail

python -m KalmanAttentionMOT.main train --config configs/kitti_kalman_attn.yaml

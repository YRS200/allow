#!/usr/bin/env bash
set -euo pipefail

python -m KalmanAttentionMOT.main train --config configs/synthetic_debug.yaml

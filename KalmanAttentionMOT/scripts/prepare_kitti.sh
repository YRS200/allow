#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/kitti/root" >&2
  exit 1
fi

ROOT=$1
mkdir -p "$ROOT"/data_tracking_det_02/training/det_02
mkdir -p "$ROOT"/data_tracking_label_02/training/label_02
mkdir -p "$ROOT"/data_tracking_image_02
mkdir -p "$ROOT"/data_tracking_calib

echo "Created KITTI directory scaffolding at $ROOT. Place the official files accordingly." >&2

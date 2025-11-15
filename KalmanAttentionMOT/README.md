# KalmanAttentionMOT

KalmanAttentionMOT is a research-oriented multi-object tracking (MOT) system
that couples a learnable KalmanNet motion filter with cross-attention based data
association. The project follows a HybridTrack-inspired layout with separate
configuration files, scripts and a modular `src/` tree.

## Project layout

```
KalmanAttentionMOT/
├── configs/                # YAML experiment definitions
├── experiments/            # Logs, checkpoints and generated results
├── scripts/                # Convenience launch scripts
├── src/                    # Library code (data, models, tracker, utils)
└── main.py                 # CLI entry point
```

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run a quick synthetic demo:

   ```bash
   python -m KalmanAttentionMOT.main demo --steps 10
   ```

3. Train on the synthetic toy dataset:

   ```bash
   bash scripts/train_synthetic.sh
   ```

4. (Optional) Prepare KITTI directories and export tracker predictions:

   ```bash
   bash scripts/prepare_kitti.sh /path/to/kitti
   bash scripts/eval_kitti.sh experiments/results_kitti
   ```

## Configuration

Experiments are driven by YAML files in `configs/`. Each file defines three
sections:

- `tracker`: hyper-parameters for the KalmanAttentionTracker (embedding size,
  gating threshold, etc.).
- `data`: dataset source and loader options.
- `training`: optimiser and runtime settings.

## Development notes

- Cross-attention data association lives in `src/association/` and is decoupled
  from the tracker lifecycle.
- `src/tracker/tracker.py` owns the high-level loop: prediction, attention-based
  matching, Kalman updates, and track management.
- Evaluation helpers in `src/evaluation/` can export KITTI-compatible result
  files and compute classic MOT metrics.


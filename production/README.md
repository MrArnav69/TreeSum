# Production Directory Structure (A40 GPU)

This directory is reserved exclusively for high-performance execution on the A40 GPU. It is isolated from local exploratory scripts to ensure data integrity and reproducibility.

## Layout

```text
production/
├── scripts/             # High-performance runners (Batching/FP16 enabled)
├── data/                # Golden 500-sample sets and large-scale shards
├── results/             # Production-grade outputs
│   ├── alpha_sweep_500/ # Detailed parameter sweep (α=0.0 to 1.0)
│   └── ablation_500/    # Full-scale strategy comparisons
├── logs/                # Terminal output and error tracking per run
└── configs/             # Hyperparameter and batch size specifications
```

## Optimization Roadmap

- [ ] Implement `bfloat16` precision in `scripts/`.
- [ ] Implement cross-document batching.
- [ ] Implement incremental checkpointing (every 50 samples).

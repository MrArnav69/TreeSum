# TreeSum: Semantic Chunking and Recursive Tree-Reduction

Research implementation of **TreeSum**, a hierarchical abstractive summarization framework designed for long-form multi-document clusters. This repository provides tools for semantic document segmentation and recursive information reduction.

## Repository Structure

```text
├── src/                        # Library core
│   ├── semantic_document_chunker.py   # Hybrid semantic-lexical segmentation
│   └── hierarchical_summarizer.py     # Recursive tree-reduction logic
├── experiments/                 # Systematic studies
│   ├── ablation/               # Strategy comparisons (Baseline vs TreeSum)
│   └── alpha_sweep_study/       # Parameter sensitivity analysis (α weight)
├── results/                    # Experimental data
│   ├── ablation_20_samples/    # Initial strategy comparisons
│   └── alpha_sweep_20_samples/ # Hybrid weight sensitivity results
├── notebooks/                  # Post-experimental analysis
│   └── visualization/          # Plotting and LaTeX table generation
├── archive/                    # Legacy and exploratory scripts
└── environment.yml             # Conda environment specification
```

## Implementation Overview

### 1. Hybrid Semantic Chunking

The segmentation logic fuses embedding-based cosine similarity with lexical Jaccard similarity to identify optimal split points that preserve entity flow.

### 2. Recursive Tree-Reduction

A hierarchical "Map-Reduce" architecture that handles documents exceeding the LLM context window by recursively summarizing clusters of summaries.

## Reproduction

To reproduce the 20-sample ablation study:

1. `python experiments/ablation/prepare_data.py`
2. `python experiments/ablation/run_ablation.py`
3. `python notebooks/visualization/plot_ablation.py`

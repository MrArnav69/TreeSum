# TreeSum: Hierarchical Multi-Document Summarization

Research repository for **TreeSum**, an abstractive summarization model designed for long-form multi-document context using Semantic Chunking and Recursive Tree-Reduction.

## Core Contribution

- **Semantic Chunking**: Hybrid sentence/token-level segmentation based on local semantic coherence.
- **Recursive Tree-Reduction**: Map-Reduce architecture for infinite-length document processing without truncation.
- **Improved Performance**: Benchmarked on Multi-News with ~47 ROUGE-1.

## Repository Structure

- `src/`: Core model implementation (`HierarchicalSummarizer.py`, `SemanticDocumentChunker.py`).
- `notebooks/`: Evaluation and verification scripts.
- `results/`: Experiment data and ROUGE metrics.
- `tests/`: Unit tests and sanity checks.

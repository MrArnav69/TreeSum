# TreeSum Production Study: A40 Deployment

This directory contains the finalized, sample-aligned scripts for running TreeSum on high-performance hardware (A40/H100/H200).

## 📁 Directory Structure

- `scripts/`: Production-ready execution scripts.
  - `run_a40_mega_sota.py`: The master script for the 1000-sample study.
  - `generate_golden_data.py`: Re-generates the alignment set if needed.
- `data/`:
  - `golden_1000_shared.json`: **CRITICAL**. This contains the 1000 documents that match the "Flat" baselines perfectly.
- `results/`: Output directory for summaries and metrics.
- `logs/`: Execution logs.

## 🚀 How to Run (A40 Hardware)

1. **Environment Setup**:
   Ensure you have the required dependencies installed:

   ```bash
   pip install transformers datasets evaluate rouge_score bert_score sentence-transformers accelerate
   ```

2. **Run the Study**:
   Navigate to the production directory and execute:
   ```bash
   python3 scripts/run_a40_mega_sota.py
   ```

## 🛠️ Optimizations

- **Precision**: Uses `torch.bfloat16` for maximum speed and memory safety on Ampere GPUs.
- **Alignment**: Uses the `golden_1000_shared.json` to fix the index mismatch found in the Kaggle runs.
- **Resumption**: Automatically resumes from the last successfully processed sample if interrupted.

## 📊 Alignment Note

The previous Kaggle runs for TreeSum used a different random subset (16% overlap). This production study re-runs TreeSum on the **exact same 1000 documents** used by the "Flat 1024" and "Flat Overlap" baselines, ensuring a valid head-to-head comparison for the final paper.

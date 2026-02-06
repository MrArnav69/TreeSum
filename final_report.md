# Ablation Study: Final Comparison Report
Generated on: 2026-02-06 13:41:04

## Results Summary

| source_folder   |   num_samples |   rouge1 |   rouge2 |   rougeL |   bertscore_f1 |
|:----------------|--------------:|---------:|---------:|---------:|---------------:|
| Overlap 1024    |          1000 |    20.56 |     1.79 |    12.86 |          47.75 |
| Flat 1024       |          1000 |    20.12 |     1.75 |    12.58 |          46.77 |

## Analysis
- **Flat 1024**: Baseline chunking strategy.
- **Overlap 1024**: Sliding window strategy (128-token overlap).

> [!NOTE]
> Metrics were calculated locally on Mac M3 Pro using the DeBERTa-XLarge-MNLI model for BERTScore.

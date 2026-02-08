# Ablation Study: Final Comparison Report
Generated on: 2026-02-06 22:24:59

## Results Summary

| source_folder     | method                |   num_samples |   rouge1 |   rouge2 |   rougeL |   bertscore_f1 |
|:------------------|:----------------------|--------------:|---------:|---------:|---------:|---------------:|
| Flat 1024         | Flat_1024_NoOverlap   |          1000 |    45.56 |    17.76 |    23.22 |          65.04 |
| Flat 1024 Overlap | Flat_1024_128_Overlap |          1000 |    45.59 |    17.86 |    23.29 |          65.09 |

## Analysis
- **Flat 1024**: Baseline chunking strategy.
- **Overlap 1024**: Sliding window strategy (128-token overlap).

> [!NOTE]
> Metrics were calculated locally on Mac M3 Pro using the DeBERTa-XLarge-MNLI model for BERTScore.

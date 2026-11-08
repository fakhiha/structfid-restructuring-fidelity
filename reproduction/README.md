# StructScore reproduction (base paper)

Reproduction of StructScore (Zhong & Litman, NAACL 2025) on the DiverSumm test file (451 summaries) with the MiniCheck-FlanT5-Large backbone, rows 11 to 14 of Table 4 in the original paper. Run on Google Colab with one T4 GPU.

| File | What it is |
|---|---|
| `LevelA_released_scores.ipynb` | Level A: clones the authors' repo and applies their re-weighting code to their released sentence scores |
| `LevelB_recomputed_minicheck.ipynb` | Level B: recomputes every MiniCheck score from raw text (3,675 summary sentences, 500-token chunks, 2 h 16 min), then re-weights |
| `my_minicheck_scores.jsonl` | Level B MiniCheck scores, one row per summary |
| `levelB_results.csv` | Level B AUC per split |
| `reproduction_results.csv` | Level A AUC per split vs. the paper (values as printed in the Level A notebook output) |

Main result: sentence scores match the released ones exactly (Pearson r = 1.000); re-weighted macro-AUC 68.59 vs. 68.27 reported; StructScore-LV1 67.14 vs. 67.31.

Authors' code: https://github.com/cs329yangzhong/discourse-driven-summary-factuality-evaluation

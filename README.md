# Detecting Structural Fidelity Failures in GenAI Text Restructuring

Code and data for the Research Methodology Assignment 2 paper by Fakhiha Rameen (26I-7821).

GenAI assistants often *restructure* text for accessibility. They keep the words but add headings, split paragraphs, turn prose into bullets and reorder sentences. This repository contains:

- a taxonomy of five structural fidelity failures, F1–F5: detached connective, orphaned anaphor, orphaned qualifier, order inversion and broken cross-reference
- a synthetic benchmark of 587 restructurings that keep every word of the source, with clean and benign-move controls
- 60 real restructurings of 20 OpenStax passages by GPT-6, Gemini 3.6 Flash and Claude Opus 5.5
- the Context-Shift Detector (CSD), which aligns each restructured sentence to the source and asks an instruction-tuned LM (Qwen2.5-7B-Instruct, 4-bit) two questions: does the sentence depend on its original context, and does its meaning change in the new context?
- baselines (ROUGE-L, BERTScore, MiniCheck, perplexity, a structure-only ablation) and four open LLM judges validated on known-answer items
- a reproduction of the base paper, StructScore (Zhong & Litman, NAACL 2025), on DiverSumm

## Repository layout

```
src/
  structfid.py         units, cue-based dependency links, the F1–F5 break rules
  inject.py            synthetic benchmark: clean / benign / harmful moves, word-equality check
  extract_corpus.py    DiverSumm (GovReport, ChemSum) source paragraphs
  parse_outputs.py     parse and align the real Markdown restructurings
  detector.py          CSD: segmentation, exact-then-fuzzy alignment, probe scorers
  run_synthetic.py     v1–v3 experiments and all baselines
  run_v4.py            final CSD (need × change), learned combiners, bootstrap
  judges.py            LLM-as-judge validation (Phi-3.5, OLMo-2, Granite-3.1)
  judges_v2.py         Phi-4 two-step judge
  analyze_judges.py    judge AUC, Fleiss' kappa, bias checks
  csd_real.py          CSD on the 60 real restructurings
  make_annotation.py   blind 100-item human annotation sheet
  make_llm_batches.py  70-item external-LLM annotation batch with hidden known-answer items
  analyze_annotation.py  agreement of annotations with CSD and judges
data/                  passages, synthetic benchmark, real outputs, annotation keys
results_prev/          v1 baseline scores (MiniCheck, BERTScore, ROUGE-L, PPL)
results_v3/            v3 run (3B dependency probe) from notebooks/colab_run_v3.ipynb
notebooks/             Colab notebook used to run the v1-v3 experiments
results_v4/            final CSD scores and summary metrics
results_judges/        round-1 judge outputs and analysis
results_judges2/       Phi-4 judge outputs
reproduction/          StructScore reproduction notebooks and result files
```

## Setup

Python 3.10+, one 16 GB GPU (a free Colab or Kaggle T4 is enough for the 7B probe in 4-bit).

```
pip install -r requirements.txt
pip install bitsandbytes accelerate
```

## Reproducing the results

```
# 1. Build the synthetic benchmark (seed 13)
python src/extract_corpus.py <diversumm_test.csv> data/diversumm_paragraphs.json
python src/inject.py data/openstax_passages.json data/synthetic_openstax.jsonl
python src/inject.py data/diversumm_paragraphs.json data/synthetic_diversumm.jsonl

# 2. Baselines (v1)
python src/run_synthetic.py --data data --out results_prev --lm Qwen/Qwen2.5-1.5B --minicheck

# 3. Final detector (v4) on the synthetic benchmark
python src/run_v4.py --data data --out results_v4 --four-bit --baselines results_prev/synthetic_scores.jsonl

# 4. LLM judges
python src/judges.py --data data --out results_judges
python src/judges_v2.py --data data --out results_judges2
python src/analyze_judges.py results_judges

# 5. CSD on the real restructurings
python src/csd_real.py --out results_judges/csd_real.jsonl
```

Model choice and the decision threshold are set on the DiverSumm dev split only. The OpenStax split is the held-out test set, and the real outputs are never used for tuning.

## Main results (synthetic benchmark, harmful vs benign moves)

| Method | Dev AUC (DiverSumm) | Test AUC (OpenStax) |
|---|---|---|
| **CSD (need × change)** | 0.670 | **0.693** |
| Perplexity | – | 0.614 |
| CSD-struct (moves only) | – | 0.539 |
| BERTScore | – | 0.519 |
| MiniCheck | – | 0.496 |
| ROUGE-L | – | 0.413 |

Over all 587 samples CSD beats every baseline (paired bootstrap, 1,000 resamples, p ≤ 0.003). The four open LLM judges reach AUC 0.53–0.60 on known-answer items. Full tables are in the paper.

## Data and licences

- OpenStax passages: *Biology 2e*, *Chemistry 2e*, *College Physics 2e* and *Astronomy 2e*, CC BY 4.0, © OpenStax, Rice University.
- DiverSumm: from Zhang et al. (EACL 2024), redistributed under its original terms through the StructScore repository (github.com/cs329yangzhong/discourse-driven-summary-factuality-evaluation).
- The real restructurings were collected on 7 November 2026 through the consumer chat interfaces, using one fixed prompt in a new chat with memory turned off.

## Limitations

The real-output analysis is descriptive. Human or validated-LLM labels for the real moved sentences are future work, and the annotation tooling for them is included (`make_annotation.py`, `make_llm_batches.py`, `analyze_annotation.py`).

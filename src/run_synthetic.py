"""Experiments on the synthetic set (RQ2, RQ4, ablation). Run on a Colab GPU.

Detectors / baselines (every one sees the same source S and restructured R):
  CSD (ours)      max context shift, causal LM            (needs --lm)
  CSD-struct      ablation: number of moved segments, no LM
  PPL             ablation/baseline: LM log-prob of S minus log-prob of R (no alignment)
  ROUGE-L         1 - ROUGE-L F1(S, R)
  BERTScore       1 - BERTScore F1(S, R)
  MiniCheck       1 - mean MiniCheck support of R's sentences given S   (--minicheck)

Protocol: DiverSumm part = dev (threshold tuning), OpenStax part = test.
Metrics: ROC-AUC (harmful vs not harmful; harmful vs benign only), F1 at the dev
threshold, recall per failure type, localisation accuracy (CSD), paired
bootstrap (1000 resamples) for AUC(CSD) - AUC(baseline).

Usage: python src/run_synthetic.py --lm Qwen/Qwen2.5-1.5B --out results/ [--minicheck]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from detector import (HFScorer, OverlapScorer, OverlapProbe, ProbeScorer, detect_probe,  # noqa: E402
                      detect_variants, segment_restructured)

HARMFUL = {"F1", "F2", "F3", "F4", "F5"}


def plain(md: str) -> str:
    return " ".join(g["text"] for g in segment_restructured(md))


def rouge_l(ref: str, hyp: str) -> float:
    """ROUGE-L F1 on lower-cased word tokens (LCS), no external package needed."""
    a, b = re.findall(r"\w+", ref.lower()), re.findall(r"\w+", hyp.lower())
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, yv in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == yv else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(b), lcs / len(a)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def load(data_dir):
    src = {}
    for f in ["openstax_passages.json", "diversumm_paragraphs.json"]:
        for p in json.load(open(os.path.join(data_dir, f))):
            src[p["passage_id"]] = p["text"]
    rows = []
    for f, split in [("synthetic_diversumm.jsonl", "dev"), ("synthetic_openstax.jsonl", "test")]:
        for line in open(os.path.join(data_dir, f)):
            r = json.loads(line)
            r["split"] = split
            r["source"] = src[r["passage_id"]]
            rows.append(r)
    return rows


def best_threshold(y, s):
    cands = np.unique(s)
    f1s = [f1_score(y, s >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def bootstrap_diff(y, a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lm", default=None, help="HF causal LM name; omit to use the overlap test scorer")
    ap.add_argument("--minicheck", action="store_true")
    ap.add_argument("--probe", default=None, help="instruction LM for the dependency probe, e.g. Qwen/Qwen2.5-3B-Instruct; 'test' = overlap stub")
    ap.add_argument("--no-bertscore", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.data)
    print(f"{len(rows)} samples")

    scorer = HFScorer(a.lm) if a.lm else OverlapScorer()
    var, prepared = detect_variants([(r["source"], r["text"]) for r in rows], scorer)
    if a.probe:
        probe = OverlapProbe() if a.probe == "test" else ProbeScorer(a.probe)
        pvar, _ = detect_probe([(r["source"], r["text"]) for r in rows], probe)
        for v, pv in zip(var, pvar):
            v.update(pv)
        del probe
    y_all = np.array([r["label"] in HARMFUL for r in rows])
    lab_all = np.array([r["label"] for r in rows])
    dev_mask = np.array([r["split"] == "dev" for r in rows])
    # ---- variant selection on DEV only, by AUC on the hard question: harmful vs benign move
    vtab = {}
    for vn in var[0]:
        sc = np.array([v[vn][0] for v in var])
        hb = lab_all != "clean"
        vtab[vn] = {"dev_auc_benign_vs_harmful": roc_auc_score(y_all[dev_mask & hb], sc[dev_mask & hb]),
                    "dev_auc_all": roc_auc_score(y_all[dev_mask], sc[dev_mask]),
                    "test_auc_benign_vs_harmful": roc_auc_score(y_all[~dev_mask & hb], sc[~dev_mask & hb]),
                    "test_auc_all": roc_auc_score(y_all[~dev_mask], sc[~dev_mask])}
    best = max(vtab, key=lambda n: vtab[n]["dev_auc_benign_vs_harmful"])
    print("selected on dev:", best, vtab[best])
    S = {"CSD": [v[best][0] for v in var],
         "CSD-v1 (harm,long,all)": [v["harm[long,k=all]"][0] for v in var],
         "CSD-struct": [sum(1 for g in segs if g.get("moved")) for segs in prepared]}
    tops = [v[best][1] for v in var]

    # localisation: does the top segment contain the injected (moved) unit?
    loc_hits = []
    for r, segs, t in zip(rows, prepared, tops):
        if r["label"] in HARMFUL and t is not None:
            unit = r["units"][r["injected"]["d"]]
            top = segs[t]["text"]
            loc_hits.append(unit.strip(" .")[:40].lower() in top.lower() or top.strip(" .")[:40].lower() in unit.lower())

    # PPL baseline: how much less likely the restructured text is than the source, same LM
    if a.lm:
        lp_S = scorer.mean_logprob([("", r["source"]) for r in rows])
        lp_R = scorer.mean_logprob([("", plain(r["text"])) for r in rows])
        S["PPL"] = [x - y for x, y in zip(lp_S, lp_R)]

    S["ROUGE-L"] = [1 - rouge_l(r["source"], plain(r["text"])) for r in rows]

    if not a.no_bertscore:
        from bert_score import score as bscore
        _, _, F = bscore([plain(r["text"]) for r in rows], [r["source"] for r in rows], lang="en", verbose=False)
        S["BERTScore"] = [1 - float(f) for f in F]

    if a.minicheck:
        from minicheck.minicheck import MiniCheck
        from nltk.tokenize import sent_tokenize
        mc = MiniCheck(model_name="flan-t5-large", cache_dir="./ckpts")
        docs, claims, owner = [], [], []
        for k, r in enumerate(rows):
            for s in sent_tokenize(plain(r["text"])):
                docs.append(r["source"]); claims.append(s); owner.append(k)
        _, probs, _, _ = mc.score(docs=docs, claims=claims)
        agg = [[] for _ in rows]
        for k, p in zip(owner, probs):
            agg[k].append(p)
        S["MiniCheck"] = [1 - float(np.mean(v)) for v in agg]

    y = np.array([r["label"] in HARMFUL for r in rows])
    split = np.array([r["split"] for r in rows])
    lab = np.array([r["label"] for r in rows])
    res = {"n": {"dev": int((split == "dev").sum()), "test": int((split == "test").sum())}, "methods": {}}
    for name, sc in S.items():
        sc = np.asarray(sc, dtype=float)
        dev, te = split == "dev", split == "test"
        thr = best_threshold(y[dev], sc[dev])
        nb = lab != "clean"  # harmful vs benign only
        m = {"auc_all": roc_auc_score(y, sc),
             "auc_test": roc_auc_score(y[te], sc[te]),
             "auc_benign_vs_harmful": roc_auc_score(y[nb], sc[nb]),
             "threshold": thr,
             "test_precision": precision_score(y[te], sc[te] >= thr, zero_division=0),
             "test_recall": recall_score(y[te], sc[te] >= thr, zero_division=0),
             "test_f1": f1_score(y[te], sc[te] >= thr, zero_division=0),
             "recall_by_type_all": {t: float((sc[lab == t] >= thr).mean()) for t in sorted(HARMFUL)},
             "false_alarm_benign_all": float((sc[lab == "benign"] >= thr).mean()),
             "false_alarm_clean_all": float((sc[lab == "clean"] >= thr).mean())}
        res["methods"][name] = m
    res["methods"]["CSD"]["localisation_acc"] = float(np.mean(loc_hits)) if loc_hits else None
    res["methods"]["CSD"]["selected_variant"] = best
    res["variants"] = vtab
    te = split == "test"
    res["bootstrap_auc_test_vs_CSD"] = {n: bootstrap_diff(y[te], np.asarray(S["CSD"])[te], np.asarray(S[n])[te]) for n in S if n != "CSD"}
    nbm = (lab != "clean")
    res["bootstrap_auc_benign_vs_harmful_vs_CSD"] = {n: bootstrap_diff(y[nbm], np.asarray(S["CSD"])[nbm], np.asarray(S[n])[nbm]) for n in S if n != "CSD"}
    res["bootstrap_auc_all_vs_CSD"] = {n: bootstrap_diff(y, S["CSD"], S[n]) for n in S if n != "CSD"}
    json.dump(res, open(os.path.join(a.out, "synthetic_results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "synthetic_scores.jsonl"), "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"id": r["id"], "label": r["label"], "split": r["split"],
                                **{n: float(S[n][k]) for n in S}}) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

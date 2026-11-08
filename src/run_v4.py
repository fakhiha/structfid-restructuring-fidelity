"""Detector v4 on the synthetic set: dependency probe + meaning-shift probe + trained combiner.

Scores per text (all from ONE instruction LM, prompts name no cue words):
  need        max P(moved segment depends on its ORIGINAL previous text)
  harm        max [need - P(depends on its NEW previous text)]
  change      max [1 - P(understood the same before vs after editing)]
  need*change max over moved segments of need x change
  CSD-LR      logistic regression over all probe features, trained on the dev split
              (DiverSumm) and applied to the test split (OpenStax); also 5-fold
              GroupKFold over all passages (out-of-fold scores) for a larger estimate.
Baselines are merged from earlier runs' synthetic_scores.jsonl files (same sample ids).

Usage: python src/run_v4.py --probe Qwen/Qwen2.5-7B-Instruct --four-bit --out results_v4 \
           --baselines results_v3/synthetic_scores.jsonl results/synthetic_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from detector import OverlapProbe, ProbeScorer, probe_features  # noqa: E402
from run_synthetic import HARMFUL, best_threshold, bootstrap_diff, load, plain, rouge_l  # noqa: E402

FEATS = ["max_need", "max_harm", "max_change", "max_need_x_change", "mean_change", "n_moved", "frac_moved"]


def metrics(y, lab, split, sc, thr_from="dev"):
    sc = np.asarray(sc, float)
    dev, te, hb = split == "dev", split == "test", lab != "clean"
    thr = best_threshold(y[dev], sc[dev])
    return {
        "dev_auc_benign_vs_harmful": roc_auc_score(y[dev & hb], sc[dev & hb]),
        "test_auc_benign_vs_harmful": roc_auc_score(y[te & hb], sc[te & hb]),
        "dev_auc_all": roc_auc_score(y[dev], sc[dev]),
        "test_auc_all": roc_auc_score(y[te], sc[te]),
        "threshold": thr,
        "test_precision": precision_score(y[te], sc[te] >= thr, zero_division=0),
        "test_recall": recall_score(y[te], sc[te] >= thr, zero_division=0),
        "test_f1": f1_score(y[te], sc[te] >= thr, zero_division=0),
        "recall_by_type_all": {t: float((sc[lab == t] >= thr).mean()) for t in sorted(HARMFUL)},
        "false_alarm_benign_all": float((sc[lab == "benign"] >= thr).mean()),
        "false_alarm_clean_all": float((sc[lab == "clean"] >= thr).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_v4")
    ap.add_argument("--probe", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--baselines", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = load(a.data)
    probe = OverlapProbe() if a.probe == "test" else ProbeScorer(a.probe, batch_size=a.batch_size, four_bit=a.four_bit)
    feats, prepared = probe_features([(r["source"], r["text"]) for r in rows], probe)
    json.dump([{"id": r["id"], **{k: v for k, v in f.items()}} for r, f in zip(rows, feats)],
              open(os.path.join(a.out, "probe_features.json"), "w"))

    y = np.array([r["label"] in HARMFUL for r in rows])
    lab = np.array([r["label"] for r in rows])
    split = np.array([r["split"] for r in rows])
    groups = np.array([r["passage_id"] for r in rows])
    X = np.array([[f[k] for k in FEATS] for f in feats], float)

    S = {f"probe-{k}": X[:, FEATS.index(k)] for k in ["max_need", "max_harm", "max_change", "max_need_x_change"]}

    # combiner trained on dev only, applied to all (dev scores are in-sample; report test)
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    lr.fit(X[split == "dev"], y[split == "dev"])
    S["CSD-LR"] = lr.predict_proba(X)[:, 1]
    coefs = dict(zip(FEATS, lr[-1].coef_[0].round(3).tolist()))
    # ablation: probe features only (no structural counts), to rule out a structure-only shortcut
    PF = [FEATS.index(k) for k in ["max_need", "max_harm", "max_change", "max_need_x_change", "mean_change"]]
    lr_p = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    lr_p.fit(X[split == "dev"][:, PF], y[split == "dev"])
    S["CSD-LR (probe feats only)"] = lr_p.predict_proba(X[:, PF])[:, 1]

    # out-of-fold scores over ALL passages (GroupKFold by passage) for a larger, unbiased estimate
    oof = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    # baselines from earlier runs
    ids = [r["id"] for r in rows]
    for path in a.baselines:
        if not os.path.exists(path):
            print("missing baseline file:", path); continue
        prev = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
        for name in ["PPL", "ROUGE-L", "BERTScore", "MiniCheck", "CSD-struct"]:
            if name not in S and all(name in prev.get(i, {}) for i in ids):
                S[name] = np.array([prev[i][name] for i in ids])
    if "ROUGE-L" not in S:
        S["ROUGE-L"] = np.array([1 - rouge_l(r["source"], plain(r["text"])) for r in rows])

    res = {"probe_model": a.probe, "four_bit": a.four_bit, "lr_coefficients": coefs, "methods": {}}
    for name, sc in S.items():
        res["methods"][name] = metrics(y, lab, split, sc)

    hb = lab != "clean"
    res["cv_oof_CSD-LR"] = {"auc_all": roc_auc_score(y, oof), "auc_benign_vs_harmful": roc_auc_score(y[hb], oof[hb])}
    for name in S:
        if name.startswith("probe") or name.startswith("CSD-LR"):
            continue
        res["cv_oof_CSD-LR"][f"baseline_{name}_auc_benign_vs_harmful_all"] = roc_auc_score(y[hb], np.asarray(S[name])[hb])
        res["cv_oof_CSD-LR"][f"bootstrap_vs_{name}_benign_vs_harmful"] = bootstrap_diff(y[hb], oof[hb], np.asarray(S[name])[hb])

    te = split == "test"
    res["bootstrap_test_benign_vs_harmful_CSD-LR_minus"] = {
        n: bootstrap_diff(y[te & hb], S["CSD-LR"][te & hb], np.asarray(S[n])[te & hb]) for n in S if n != "CSD-LR"}

    # localisation (harmful samples): is the segment with max need*change the injected one?
    hits = []
    for r, f, segs in zip(rows, feats, prepared):
        if r["label"] in HARMFUL and f["top"] is not None:
            unit = r["units"][r["injected"]["d"]].strip(" .").lower()
            top = segs[f["top"]]["text"].strip(" .").lower()
            hits.append(unit[:40] in top or top[:40] in unit)
    res["localisation_acc"] = float(np.mean(hits)) if hits else None

    json.dump(res, open(os.path.join(a.out, "v4_results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "v4_scores.jsonl"), "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"id": r["id"], "label": r["label"], "split": r["split"], "CSD-LR_oof": float(oof[k]),
                                **{n: float(S[n][k]) for n in S}}) + "\n")
    print("\n=== SUMMARY (AUC, harmful vs benign move) ===")
    for n, m in res["methods"].items():
        print(f"{n:24s} dev {m['dev_auc_benign_vs_harmful']:.3f}   test {m['test_auc_benign_vs_harmful']:.3f}   test F1 {m['test_f1']:.3f}")
    print(f"CSD-LR 5-fold out-of-fold (all 587): {res['cv_oof_CSD-LR']['auc_benign_vs_harmful']:.3f}")
    print("localisation:", res["localisation_acc"], "| LR coefficients:", coefs)


if __name__ == "__main__":
    main()

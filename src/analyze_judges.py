"""Analyse LLM-judge labels and CSD on real outputs (Step 5 / RQ2-RQ3 on real data).

Usage: python src/analyze_judges.py results_judges [threshold]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

TYPES = ["none", "F1", "F2", "F3", "F4", "F5"]


def fleiss_kappa(labels: np.ndarray, cats) -> float:
    """labels: items x raters (category values)."""
    n_items, n_raters = labels.shape
    M = np.array([[np.sum(row == c) for c in cats] for row in labels], float)
    p_j = M.sum(0) / (n_items * n_raters)
    P_i = ((M ** 2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def bootstrap_auc(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main(d, thr=0.5):
    judges = {}
    for p in sorted(glob.glob(os.path.join(d, "judge_*.jsonl"))):
        judges[os.path.basename(p)[6:-6]] = pd.read_json(p, lines=True)
    names = list(judges)
    base = judges[names[0]][["set", "out_id", "app", "seg", "gold"]].copy()
    for n in names:
        base[n] = judges[n]["label"].values
        base[n + "_p"] = judges[n]["p_problem"].values
    out = {"judges": names}

    # ---------------- 1. validation on synthetic (gold known)
    syn = base[base.set == "synthetic"].copy()
    yb = (syn.gold != "none").values
    val = {}
    for n in names:
        pb = (syn[n] != "none").values
        harm = syn.gold != "none"
        val[n] = {"accuracy": float((pb == yb).mean()), "balanced_accuracy": balanced_accuracy_score(yb, pb),
                  "auc_p_problem": roc_auc_score(yb, syn[n + "_p"]),
                  "recall_harmful": float(pb[yb].mean()), "false_alarm_benign": float(pb[~yb].mean()),
                  "type_accuracy_on_harmful": float((syn.loc[harm, n] == syn.loc[harm, "gold"]).mean()),
                  "share_answering_problem": float(pb.mean())}
    maj_syn = (np.sum([(syn[n] != "none").values for n in names], 0) >= 2)
    val["majority"] = {"accuracy": float((maj_syn == yb).mean()), "balanced_accuracy": balanced_accuracy_score(yb, maj_syn),
                       "recall_harmful": float(maj_syn[yb].mean()), "false_alarm_benign": float(maj_syn[~yb].mean())}
    out["validation_synthetic"] = val

    # ---------------- 2. agreement on real sentences
    real = base[base.set == "real"].copy()
    B = np.array([(real[n] != "none").astype(int).values for n in names]).T
    T = np.array([real[n].values for n in names]).T
    out["agreement_real"] = {
        "fleiss_kappa_binary": fleiss_kappa(B, [0, 1]),
        "fleiss_kappa_6way": fleiss_kappa(T, TYPES),
        "pairwise_cohen_binary": {f"{a} vs {b}": cohen_kappa_score(B[:, i], B[:, j])
                                  for i, a in enumerate(names) for j, b in enumerate(names) if i < j},
        "share_problem_by_judge": {n: float(B[:, k].mean()) for k, n in enumerate(names)},
        "unanimous_share": float(((B.sum(1) == 0) | (B.sum(1) == len(names))).mean()),
    }
    real["votes"] = B.sum(1)
    real["majority"] = real["votes"] >= 2
    real["unanimous_problem"] = real["votes"] == len(names)

    # majority type among problem votes
    def maj_type(row):
        c = Counter(row[n] for n in names if row[n] != "none")
        return c.most_common(1)[0][0] if c else "none"
    real["maj_type"] = [maj_type(r) if m else "none" for (_, r), m in zip(real.iterrows(), real.majority)]

    # ---------------- 3. how often each LLM breaks structure (RQ3, real data)
    per_app = {}
    for app, g in real.groupby("app"):
        outs = g.groupby("out_id")
        per_app[app] = {"moved_sentences": int(len(g)),
                        "flagged_majority": int(g.majority.sum()),
                        "flagged_share": float(g.majority.mean()),
                        "outputs_with_any_failure": int(outs.majority.any().sum()),
                        "outputs": int(outs.ngroups),
                        "unanimous_flags": int(g.unanimous_problem.sum()),
                        "types": dict(Counter(g.loc[g.majority, "maj_type"]))}
    out["failures_by_model_real"] = per_app

    # ---------------- 4. CSD vs judge majority
    csd = pd.read_json(os.path.join(d, "csd_real.jsonl"), lines=True)
    m = real.merge(csd[["out_id", "seg", "need", "fit", "change", "need_x_change", "text"]], on=["out_id", "seg"], how="left")
    assert m.need_x_change.notna().all(), "segment index mismatch"
    y = m.majority.values
    res = {}
    for col in ["need_x_change", "need", "change"]:
        lo, hi = bootstrap_auc(y, m[col])
        res[col] = {"auc_vs_majority": roc_auc_score(y, m[col]), "ci95": [lo, hi]}
    pred = m.need_x_change.values >= thr
    res["at_dev_threshold"] = {"threshold": thr, "precision": float(y[pred].mean()) if pred.any() else None,
                               "recall": float(pred[y].mean()), "flag_rate": float(pred.mean()),
                               "base_rate_majority": float(y.mean())}
    u = m.votes.isin([0, len(names)])
    res["auc_vs_unanimous_only"] = roc_auc_score(m.loc[u, "unanimous_problem"], m.loc[u, "need_x_change"])
    # output level: max over moved sentences vs any majority failure
    o = m.groupby("out_id").agg(y=("majority", "any"), s=("need_x_change", "max"), app=("app", "first"))
    res["output_level_auc"] = roc_auc_score(o.y, o.s) if o.y.nunique() == 2 else None
    # judge-vs-judge AUC for reference: each judge's p_problem against the majority of the OTHER two
    ref = {}
    for k, n in enumerate(names):
        others = np.delete(B, k, 1).sum(1) >= 2 if len(names) > 2 else None
        if others is not None and len(set(others)) == 2:
            ref[n] = roc_auc_score(others, real[n + "_p"].values)
    res["judge_vs_other_two_auc"] = ref
    out["csd_vs_judges_real"] = res

    json.dump(out, open(os.path.join(d, "judge_analysis.json"), "w"), indent=1, default=float)
    m.to_csv(os.path.join(d, "real_sentences_labelled.csv"), index=False)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)

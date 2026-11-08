"""Evaluate CSD and the LLM judges against the author's blind labels on real moved sentences.

Usage: python src/analyze_annotation.py <filled annotation_sheet.xlsx> [csd_threshold]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def boot_auc(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def boot_diff(y, a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed); y, a, b = map(np.asarray, (y, a, b)); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    v = np.array(v)
    return {"diff": round(float(v.mean()), 3), "ci95": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)],
            "p": round(float((v <= 0).mean()), 3)}


def main(xlsx, thr=0.5000019066403244):
    ann = pd.read_excel(xlsx, sheet_name="Annotate")
    ann = ann.rename(columns={"broken (Y/N)": "broken", "confidence (1-3)": "conf"})
    key = pd.DataFrame(json.load(open("data/annotation_key.json")))
    d = ann.merge(key, on="item")
    d["broken"] = d["broken"].astype(str).str.strip().str.upper()
    d = d[d.broken.isin(["Y", "N"])].copy()
    d["y"] = d.broken == "Y"
    csd = pd.read_json("results_judges/csd_real.jsonl", lines=True)
    d = d.merge(csd[["out_id", "seg", "need", "change", "need_x_change"]], on=["out_id", "seg"], how="left")
    judges = {}
    for p in sorted(glob.glob("results_judges*/judge_*.jsonl")):
        j = pd.read_json(p, lines=True); j = j[j.set == "real"]
        name = p.split("judge_")[1][:-6]
        d = d.merge(j[["out_id", "seg", "p_problem", "label"]].rename(columns={"p_problem": f"{name}_p", "label": f"{name}_lab"}),
                    on=["out_id", "seg"], how="left")
        judges[name] = f"{name}_p"
    out = {"n_labelled": int(len(d)), "broken_share": round(float(d.y.mean()), 3),
           "by_model": {a: {"n": int(len(g)), "broken": int(g.y.sum()), "share": round(float(g.y.mean()), 3)} for a, g in d.groupby("app")},
           "types": d.loc[d.y, "type"].value_counts(dropna=False).to_dict(),
           "confidence": d["conf"].value_counts(dropna=False).to_dict()}
    y = d.y.values
    if len(set(y)) == 2:
        res = {}
        for name, col in [("CSD need x change", "need_x_change"), ("CSD need", "need"), ("CSD change", "change")] + list(judges.items()):
            res[name] = {"auc": round(roc_auc_score(y, d[col]), 3), "ci95": boot_auc(y, d[col])}
        out["auc_vs_human"] = res
        pred = d.need_x_change.values >= thr
        out["csd_at_dev_threshold"] = {"precision": round(float(y[pred].mean()), 3) if pred.any() else None,
                                       "recall": round(float(pred[y].mean()), 3), "flag_rate": round(float(pred.mean()), 3),
                                       "cohen_kappa": round(cohen_kappa_score(y, pred), 3)}
        out["judges_decision_kappa"] = {n: round(cohen_kappa_score(y, d[f"{n}_lab"] != "none"), 3) for n in judges}
        out["csd_minus_judge_auc"] = {n: boot_diff(y, d.need_x_change, d[c]) for n, c in judges.items()}
        hc = d["conf"].fillna(0) >= 2
        if hc.sum() > 10 and len(set(y[hc])) == 2:
            out["auc_high_confidence_only"] = {"n": int(hc.sum()), "CSD": round(roc_auc_score(y[hc], d.loc[hc, "need_x_change"]), 3)}
        # output level: any human-broken sentence vs max CSD over the labelled sentences of that output
        o = d.groupby("out_id").agg(y=("y", "any"), s=("need_x_change", "max"))
        if o.y.nunique() == 2:
            out["output_level_auc_labelled"] = round(roc_auc_score(o.y, o.s), 3)
    json.dump(out, open("results_judges/annotation_results.json", "w"), indent=1, default=str)
    d.to_csv("results_judges/annotation_merged.csv", index=False)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:]))

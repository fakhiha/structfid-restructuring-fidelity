"""LLM judges for structural fidelity (Step 5) + CSD on the real outputs. Run on a Kaggle/Colab GPU.

Unit of judgement: one MOVED sentence of a restructured text (found by exact alignment).
Each judge sees the original passage, the full restructured text and the target sentence,
and answers with one letter: N (no problem) or A-E (failure types F1-F5).

Judges are three open model families that are neither the restructuring models
(GPT, Gemini, Claude) nor the detector's probe model (Qwen):
  microsoft/Phi-3.5-mini-instruct, allenai/OLMo-2-1124-7B-Instruct, ibm-granite/granite-3.1-8b-instruct

1) VALIDATION on synthetic samples with known labels (up to 15 per failure type + as many benign):
   each judge's accuracy, so we know how far to trust it.
2) REAL outputs: all 259 moved sentences of the 60 ChatGPT / Gemini / Claude outputs.
3) CSD (Qwen2.5-7B-Instruct probe) scores for the same real sentences.

Usage: python src/judges.py --out results_judges [--judges ...] [--n-per-type 15]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from detector import (ProbeScorer, align_to_source, contexts, probe_features,  # noqa: E402
                      segment_restructured)

LETTERS = ["N", "A", "B", "C", "D", "E"]
LETTER2TYPE = {"N": "none", "A": "F1", "B": "F2", "C": "F3", "D": "F4", "E": "F5"}

JUDGE = """You are checking a restructured version of a textbook passage. The restructuring was supposed to only reorganise the text (headings, bullet points, order) without changing its meaning.

ORIGINAL PASSAGE:
{source}

RESTRUCTURED VERSION:
{restructured}

TARGET SENTENCE (as it appears in the restructured version):
{segment}

Because of where the target sentence now sits, is its meaning broken or changed compared with the original? Choose one:
N - No problem: it is understood the same way as in the original.
A - Detached connective: it contains a linking expression (cause, result, contrast, example) whose partner text is no longer next to it, or it is now attached to the wrong text.
B - Orphaned reference: it uses a pronoun or a "this/these ..." phrase whose referent no longer comes before it in the same section.
C - Orphaned qualifier: it is a condition, exception or limitation now separated from the claim it limits (or the claim lost its limitation).
D - Order inversion: it now comes before something it depends on (a definition, an earlier step, a cause).
E - Broken cross-reference: it points to another part of the text (above, below, later, the following) that is no longer where it says.
Answer with a single letter."""


def load_rows(data):
    src = {}
    for f in ["openstax_passages.json", "diversumm_paragraphs.json"]:
        for p in json.load(open(os.path.join(data, f))):
            src[p["passage_id"]] = p["text"]
    return src


def segment_index(md, source):
    segs = segment_restructured(md)
    align_to_source(segs, source)
    contexts(segs, source)
    return segs


def build_items(data, n_per_type, seed=7):
    src = load_rows(data)
    items = []
    # ---- real outputs: every moved sentence
    for line in open(os.path.join(data, "real_outputs.jsonl")):
        r = json.loads(line)
        segs = segment_index(r["output_text"], src[r["passage_id"]])
        for i, g in enumerate(segs):
            if g.get("moved"):
                items.append({"set": "real", "out_id": r["id"], "app": r["app"], "passage_id": r["passage_id"],
                              "seg": i, "segment": g["text"], "source": src[r["passage_id"]],
                              "restructured": r["output_text"]})
    # ---- synthetic validation: the moved/injected sentence, labels known by construction
    rng = random.Random(seed)
    syn = []
    for f in ["synthetic_openstax.jsonl", "synthetic_diversumm.jsonl"]:
        syn += [json.loads(l) for l in open(os.path.join(data, f))]
    picked = []
    for t in ["F1", "F2", "F3", "F4", "F5"]:
        pool = [r for r in syn if r["label"] == t]
        picked += rng.sample(pool, min(n_per_type, len(pool)))
    n_harm = len(picked)
    picked += rng.sample([r for r in syn if r["label"] == "benign"], n_harm)
    for r in picked:
        segs = segment_index(r["text"], src[r["passage_id"]])
        unit = r["units"][r["injected"]["d"]].strip(" .").lower()
        idx = next((i for i, g in enumerate(segs) if unit[:40] in g["text"].lower() or g["text"].strip(" .").lower()[:40] in unit), None)
        if idx is None:
            continue
        items.append({"set": "synthetic", "out_id": r["id"], "app": None, "passage_id": r["passage_id"], "seg": idx,
                      "segment": segs[idx]["text"], "source": src[r["passage_id"]], "restructured": r["text"],
                      "gold": r["label"] if r["label"] != "benign" else "none"})
    return items


def run_judge(model_name, items, batch_size, four_bit):
    import torch
    m = ProbeScorer(model_name, batch_size=batch_size, four_bit=four_bit)
    tok = m.tok
    ids = {L: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)}) for L in LETTERS}
    out = []
    for b in range(0, len(items), batch_size):
        batch = items[b:b + batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": JUDGE.format(**{k: it[k] for k in ("source", "restructured", "segment")})}],
                                           tokenize=False, add_generation_prompt=True) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            logits = m.model(**enc).logits[:, -1].float()
        lp = torch.stack([torch.logsumexp(logits[:, ids[L]], -1) for L in LETTERS], -1)
        pr = torch.softmax(lp, -1).cpu().tolist()
        for it, p in zip(batch, pr):
            letter = LETTERS[max(range(len(LETTERS)), key=lambda j: p[j])]
            out.append({"p": dict(zip(LETTERS, [round(x, 4) for x in p])), "label": LETTER2TYPE[letter],
                        "p_problem": round(1 - p[0], 4)})
        if b % (batch_size * 20) == 0:
            print(f"  {model_name}: {b}/{len(items)}", flush=True)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_judges")
    ap.add_argument("--judges", nargs="+", default=["microsoft/Phi-3.5-mini-instruct",
                                                    "allenai/OLMo-2-1124-7B-Instruct",
                                                    "ibm-granite/granite-3.1-8b-instruct"])
    ap.add_argument("--probe", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-per-type", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = build_items(a.data, a.n_per_type)
    print(f"{sum(i['set']=='real' for i in items)} real sentences, {sum(i['set']=='synthetic' for i in items)} synthetic validation sentences")

    # judges (each saved as soon as it finishes)
    for jm in a.judges:
        path = os.path.join(a.out, "judge_" + jm.split("/")[-1] + ".jsonl")
        if os.path.exists(path):
            print("skip (done):", jm); continue
        try:
            res = run_judge(jm, items, a.batch_size, four_bit=("mini" not in jm.lower()))
        except Exception as e:  # keep going with the other judges
            print("JUDGE FAILED:", jm, repr(e)[:300]); continue
        with open(path, "w") as f:
            for it, r in zip(items, res):
                f.write(json.dumps({"set": it["set"], "out_id": it["out_id"], "app": it["app"], "seg": it["seg"],
                                    "gold": it.get("gold"), **r}) + "\n")
        print("saved", path)

    # CSD on the real outputs (same segmentation -> same seg indices)
    path = os.path.join(a.out, "csd_real.jsonl")
    if not os.path.exists(path):
        src = load_rows(a.data)
        reals = [json.loads(l) for l in open(os.path.join(a.data, "real_outputs.jsonl"))]
        probe = ProbeScorer(a.probe, batch_size=8, four_bit=True)
        feats, prepared = probe_features([(src[r["passage_id"]], r["output_text"]) for r in reals], probe)
        with open(path, "w") as f:
            for r, segs in zip(reals, prepared):
                for i, g in enumerate(segs):
                    if g.get("moved"):
                        f.write(json.dumps({"out_id": r["id"], "app": r["app"], "seg": i, "need": g["need"],
                                            "fit": g["fit"], "change": g["change"], "need_x_change": g["need"] * g["change"],
                                            "text": g["text"]}) + "\n")
        print("saved", path)


if __name__ == "__main__":
    main()

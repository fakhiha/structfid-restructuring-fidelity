"""Judge round 2: one stronger open judge (default microsoft/phi-4, 14B, 4-bit) with a two-step prompt.

Step 1 (all items): plain Yes/No - is the target sentence's meaning broken by its new position?
         P(problem) = P(Yes) / (P(Yes) + P(No)); no lettered options, so no answer-position bias.
Step 2 (only items with P(problem) >= 0.5): which kind (1-5), asked separately.
Items are the same as round 1 (judges.build_items): 150 synthetic validation + 259 real sentences.

Usage: python src/judges_v2.py --out results_judges2 [--judge microsoft/phi-4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from detector import ProbeScorer  # noqa: E402
from judges import build_items  # noqa: E402

STEP1 = """You are checking a restructured version of a textbook passage. The restructuring was only supposed to reorganise the text (headings, bullet points, order) without changing its meaning.

ORIGINAL PASSAGE:
{source}

RESTRUCTURED VERSION:
{restructured}

TARGET SENTENCE (as it appears in the restructured version):
{segment}

Read the target sentence where it now stands, after the text and under the heading that now come before it. Compare with how it reads in the original passage.

Is the target sentence's meaning broken or changed because of where it now stands? For example: it now refers to the wrong thing or to nothing, a linking word now connects it to the wrong statement, a condition or limitation got separated from what it limits, it now comes before something it depends on, or it points to a part of the text that is no longer there.

If it reads the same as in the original, answer No. Answer Yes or No."""

STEP2 = """{step1}

The answer is Yes. Which kind of problem is it? Answer with one digit:
1 - a linking expression (cause, result, contrast, example) now attaches to the wrong text or lost its partner
2 - a pronoun or "this/these ..." now has no referent or the wrong one
3 - a condition, exception or limitation is separated from the claim it limits
4 - it now comes before something it depends on (a definition, earlier step, cause)
5 - it points to another part of the text (above, below, later, the following) that is no longer there"""


def ask(m, contents, options):
    import torch
    tok = m.tok
    ids = [sorted({tok.encode(o, add_special_tokens=False)[0], tok.encode(" " + o, add_special_tokens=False)[0]}) for o in options]
    out = []
    for b in range(0, len(contents), m.bs):
        prompts = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
                   for c in contents[b:b + m.bs]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            logits = m.model(**enc).logits[:, -1].float()
        lp = torch.stack([torch.logsumexp(logits[:, i], -1) for i in ids], -1)
        out += torch.softmax(lp, -1).cpu().tolist()
        if b % (m.bs * 25) == 0:
            print(f"  {b}/{len(contents)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_judges2")
    ap.add_argument("--judge", default="microsoft/phi-4")
    ap.add_argument("--batch-size", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = build_items(a.data, 15)
    m = ProbeScorer(a.judge, batch_size=a.batch_size, four_bit=True)
    s1 = [STEP1.format(source=i["source"], restructured=i["restructured"], segment=i["segment"]) for i in items]
    p1 = ask(m, s1, ["Yes", "No"])
    pos = [k for k, p in enumerate(p1) if p[0] >= 0.5]
    p2 = ask(m, [STEP2.format(step1=s1[k]) for k in pos], ["1", "2", "3", "4", "5"]) if pos else []
    t2 = {k: "F" + str(1 + max(range(5), key=lambda j: p[j])) for k, p in zip(pos, p2)}
    path = os.path.join(a.out, "judge_" + a.judge.split("/")[-1] + "_2step.jsonl")
    with open(path, "w") as f:
        for k, it in enumerate(items):
            f.write(json.dumps({"set": it["set"], "out_id": it["out_id"], "app": it["app"], "seg": it["seg"],
                                "gold": it.get("gold"), "p_problem": round(p1[k][0], 4),
                                "label": t2.get(k, "none")}) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()

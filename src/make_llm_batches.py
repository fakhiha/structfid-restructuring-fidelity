"""70 items for an external LLM judge: the first 50 annotation-sheet items (real) + 20 hidden
known-answer synthetic items from OpenStax passages (2 per failure type where available + benign),
shuffled and formatted identically. Writes paste-ready batches of 10 and a hidden key."""
import json, random, sys
sys.path.insert(0, "src")
from detector import segment_restructured, align_to_source, contexts
random.seed(99)
P = {p["passage_id"]: p["text"] for p in json.load(open("data/openstax_passages.json"))}
key_real = json.load(open("data/annotation_key.json"))[:50]
# rebuild real items' context exactly as in the annotation sheet
reals = {json.loads(l)["id"]: json.loads(l) for l in open("data/real_outputs.jsonl")}
def ctx_item(src, md, seg):
    segs = segment_restructured(md); align_to_source(segs, src); contexts(segs, src); g = segs[seg]
    prev = next((segs[j]["text"] for j in range(seg - 1, -1, -1) if segs[j]["sec"] == g["sec"]), "")
    return {"orig_before": g["ctx_S1"] or "(start of the passage)", "target": g["text"],
            "heading": g["heading"] or "(no heading)", "new_before": prev or "(first sentence of this section)"}
items = []
for k in key_real:
    r = reals[k["out_id"]]
    items.append({"kind": "real", "ann_item": k["item"], "out_id": k["out_id"], "app": k["app"], "seg": k["seg"],
                  **ctx_item(P[r["passage_id"]], r["output_text"], k["seg"])})
syn = [json.loads(l) for l in open("data/synthetic_openstax.jsonl")]
pick = []
for t, n in [("F1", 2), ("F2", 2), ("F3", 2), ("F4", 2), ("F5", 2), ("benign", 10)]:
    pool = [r for r in syn if r["label"] == t]; pick += random.sample(pool, min(n, len(pool)))
for r in pick:
    segs = segment_restructured(r["text"]); align_to_source(segs, P[r["passage_id"]]); contexts(segs, P[r["passage_id"]])
    unit = r["units"][r["injected"]["d"]].strip(" .").lower()
    idx = next(i for i, g in enumerate(segs) if unit[:40] in g["text"].lower() or g["text"].strip(" .").lower()[:40] in unit)
    items.append({"kind": "synthetic", "gold": "N" if r["label"] == "benign" else "Y", "gold_type": r["label"],
                  "syn_id": r["id"], **ctx_item(P[r["passage_id"]], r["text"], idx)})
random.shuffle(items)
for i, it in enumerate(items, 1):
    it["id"] = f"L{i:02d}"
json.dump(items, open("data/llm_label_key.json", "w"), indent=1, ensure_ascii=False)
lines = []
for b in range(0, len(items), 10):
    lines.append(f"===================== BATCH {b//10 + 1} of {(len(items)+9)//10} (items {items[b]['id']}-{items[min(b+9,len(items)-1)]['id']}) =====================\n")
    for it in items[b:b + 10]:
        lines.append(f"[{it['id']}]\nBEFORE (original text right before it): {it['orig_before']}\n"
                     f"AFTER - new section heading: {it['heading']}\nAFTER - new text right before it: {it['new_before']}\n"
                     f"TARGET SENTENCE: {it['target']}\n")
    lines.append("")
open("/home/claude/out/llm_label_items.txt", "w").write("\n".join(lines))
print(len(items), "items:", sum(i["kind"] == "real" for i in items), "real,", sum(i["kind"] == "synthetic" for i in items), "synthetic")

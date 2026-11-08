"""Score every moved sentence of the 60 real outputs with CSD (after the aligner fix).
Usage: python src/csd_real.py --out results_judges/csd_real.jsonl"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from detector import ProbeScorer, probe_features
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results_judges/csd_real.jsonl")
ap.add_argument("--probe", default="Qwen/Qwen2.5-7B-Instruct"); a = ap.parse_args()
src = {p["passage_id"]: p["text"] for p in json.load(open("data/openstax_passages.json"))}
reals = [json.loads(l) for l in open("data/real_outputs.jsonl")]
probe = ProbeScorer(a.probe, batch_size=8, four_bit=True)
feats, prepared = probe_features([(src[r["passage_id"]], r["output_text"]) for r in reals], probe)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    for r, segs in zip(reals, prepared):
        for i, g in enumerate(segs):
            if g.get("moved"):
                f.write(json.dumps({"out_id": r["id"], "app": r["app"], "seg": i, "need": g["need"], "fit": g["fit"],
                                    "change": g["change"], "need_x_change": g["need"] * g["change"], "text": g["text"]}) + "\n")
print("saved", a.out)

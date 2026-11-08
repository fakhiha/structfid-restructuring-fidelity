"""Extract candidate paragraphs from DiverSumm source documents (GovReport, ChemSum).

arXiv documents are skipped: they are lower-cased and pre-tokenised, so sentence
splitting and cue matching would not behave like on normal text.
Selection: 80-220 words, at least two link types present (same rule as OpenStax),
at most PER_DOC paragraphs per document so no single document dominates.

Usage: python src/extract_corpus.py <diversumm_test.csv> data/diversumm_paragraphs.json
"""
import json
import re
import sys

import pandas as pd

from structfid import extract_links, split_units

PER_DOC = 6
ORIGINS = ["GovReport", "ChemSum"]


def main(src, dst):
    df = pd.read_csv(src)
    out = []
    for origin in ORIGINS:
        docs = df[df.origin == origin].drop_duplicates("doc")
        for di, doc in enumerate(docs.doc):
            paras = [" ".join(p.split()) for p in re.split(r"<p>|\n\s*\n", doc)]
            picked = []
            for pi, p in enumerate(paras):
                n = len(p.split())
                if not 80 <= n <= 220 or not p[:1].isupper() or not p.rstrip().endswith((".", ")")):
                    continue
                types = {L.type for L in extract_links(split_units(p))}
                if len(types) < 2:
                    continue
                # rarer link types first, so F3-F5 get enough material
                rare = len(types & {"qual", "ord", "xref"})
                picked.append((rare, len(types), pi, p, sorted(types)))
            picked.sort(key=lambda x: (-x[0], -x[1], x[2]))
            for rare, nt, pi, p, types in picked[:PER_DOC]:
                out.append({"passage_id": f"{origin[:3].upper()}{di:02d}-{pi:03d}", "book": f"DiverSumm {origin}",
                            "section": f"doc {di}, paragraph {pi}", "url": "", "text": p, "link_types": types})
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} paragraphs to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

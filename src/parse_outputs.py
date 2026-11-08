"""Parse real LLM restructurings and align them to their source passages.

For every output:
  * strip Markdown (headings, bold, bullet marks) into a structure of
    sections (heading-delimited) and blocks (bullets / paragraphs);
  * locate every source unit in the output by exact word-sequence match
    (fuzzy match >= 0.8 as a fallback, flagged as 'modified');
  * report lexical preservation: source units missing or changed, and
    words the model added (headings, intros, summaries);
  * evaluate the dependency links of the source on the output's structure
    (break predicates from structfid) -> candidate failures per type.

Usage: python src/parse_outputs.py <passage_tracker.xlsx> data/real_outputs.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

from structfid import Restructured, extract_links, failures, split_units, as_dict

TOK = re.compile(r"[\w’'-]+")


def toks(s: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in TOK.findall(s)]


def parse_markdown(md: str):
    """Return a list of (section_id, block_id, is_heading, text) lines."""
    out, sec, blk = [], 0, 0
    lines = md.replace("\r", "").split("\n")
    seen_content = False
    for raw in lines:
        line = raw.strip()
        if not line:
            blk += 1
            continue
        is_md_heading = bool(re.match(r"^#{1,6}\s", line))
        bold_only = bool(re.fullmatch(r"\*\*[^*]+\*\*:?", line)) or bool(re.fullmatch(r"__[^_]+__", line))
        is_bullet = bool(re.match(r"^([-*•]|\d+[.)])\s+", line))
        text = re.sub(r"^#{1,6}\s+|^([-*•]|\d+[.)])\s+", "", line)
        text = text.replace("**", "").replace("__", "").strip()
        if is_md_heading or bold_only:
            if seen_content:
                sec += 1
            blk += 1
            out.append((sec, blk, True, text))
            continue
        if is_bullet:
            blk += 1
        out.append((sec, blk, False, text))
        seen_content = True
    return out


def align(units: list[dict], md: str):
    lines = parse_markdown(md)
    words, wsec, wblk, whead = [], [], [], []
    for sec, blk, head, text in lines:
        for t in toks(text):
            words.append(t); wsec.append(sec); wblk.append(blk); whead.append(head)
    used = [False] * len(words)
    loc = {}
    for i, u in enumerate(units):
        q = toks(u["text"])
        if not q:
            continue
        hit = None
        for s in range(len(words) - len(q) + 1):
            if words[s:s + len(q)] == q and not any(used[s:s + len(q)]):
                hit = (s, s + len(q), "exact"); break
        if hit is None:  # fuzzy fallback: best window of the same length (+-20%)
            best = (0.0, None)
            L = len(q)
            for w in {L, int(L * 0.8), int(L * 1.2)}:
                for s in range(0, max(1, len(words) - w + 1)):
                    if any(used[s:s + w]):
                        continue
                    r = SequenceMatcher(None, q, words[s:s + w], autojunk=False).ratio()
                    if r > best[0]:
                        best = (r, (s, s + w))
            if best[1] and best[0] >= 0.8:
                hit = (*best[1], f"modified ({best[0]:.2f})")
        if hit:
            s, e, how = hit
            for k in range(s, e):
                used[k] = True
            secs = Counter(wsec[s:e]); blks = Counter(wblk[s:e])
            loc[i] = {"start": s, "sec": wsec[s], "blk": wblk[s], "how": how,
                      "split_across_sections": len(secs) > 1, "in_heading": whead[s]}
    added = [words[k] for k in range(len(words)) if not used[k]]
    heading_words = sum(1 for k in range(len(words)) if not used[k] and whead[k])
    return loc, added, heading_words, lines


def build_R(units, loc):
    """Order units by output position; chunks = output sections."""
    found = sorted(loc, key=lambda i: loc[i]["start"])
    secs = sorted({loc[i]["sec"] for i in found})
    chunks = [[i for i in found if loc[i]["sec"] == s] for s in secs]
    return Restructured([c for c in chunks if c]), found


def main(xlsx: str, dst: str):
    P = pd.read_excel(xlsx, sheet_name="Passages")
    O = pd.read_excel(xlsx, sheet_name="Outputs")
    src = {r.passage_id: str(r.source_text).strip() for _, r in P.iterrows() if str(r.passage_id).startswith("P")}
    recs = []
    for _, r in O.iterrows():
        pid = str(r.passage_id)
        if not pid.startswith("P") or pd.isna(r.output_text):
            continue
        units = split_units(src[pid])
        links = extract_links(units)
        loc, added, head_w, lines = align(units, str(r.output_text))
        n = len(units)
        missing = [i for i in range(n) if i not in loc]
        modified = [i for i in loc if loc[i]["how"] != "exact"]
        R, found = build_R(units, loc)
        # evaluate only links whose both ends were located
        ok_links = [L for L in links if L.d in loc and (L.h is None or L.h in loc)]
        # re-index units for the break check: Restructured holds original unit ids
        br = failures(ok_links, R, n) if R.chunks else []
        recs.append({
            "id": f"{pid}-{r.app}", "passage_id": pid, "app": r.app,
            "model_name": None if pd.isna(r.model_name) else str(r.model_name),
            "n_units": n, "n_sections": len(R.chunks), "n_blocks": len({l[1] for l in lines if not l[2]}),
            "missing_units": missing, "modified_units": modified,
            "units_in_headings": [i for i in loc if loc[i]["in_heading"]],
            "split_units": [i for i in loc if loc[i]["split_across_sections"]],
            "added_words": len(added), "added_heading_words": head_w, "added_other_words": len(added) - head_w,
            "lexically_preserved": not missing and not modified,
            "order": found, "chunks": R.chunks, "links": [as_dict(L) for L in links],
            "candidate_breaks": br, "units": [u["text"] for u in units],
            "output_text": str(r.output_text),
        })
    with open(dst, "w") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} parsed outputs to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

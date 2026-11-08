"""Synthetic corruption injection (Step 3, synthetic set).

For each source passage:
  * one CLEAN control: the passage chunked into 2-3 bulleted groups at
    boundaries that cut no detected link, original order kept;
  * up to K CORRUPTED versions per failure type F1-F5, each made by moving
    exactly ONE unit, starting from the clean control.
Labels come from construction (the injected type and link) and are
re-checked by evaluating every detected link on the result.
Every output keeps the source's words exactly (checked).

Usage: python src/inject.py data/openstax_passages.json data/synthetic_openstax.jsonl [K]
"""
from __future__ import annotations

import copy
import json
import random
import re
import sys
from collections import Counter

from structfid import (Restructured, extract_links, failures, render, split_units, as_dict)

random.seed(13)


def words(text: str) -> Counter:
    return Counter(re.findall(r"[\w’'-]+", text.lower()))


def safe_boundaries(n: int, links) -> list[int]:
    """Boundary b sits between unit b-1 and b; safe if no non-xref link spans it."""
    spans = [(min(L.d, L.h), max(L.d, L.h)) for L in links if L.h is not None]
    return [b for b in range(2, n - 1) if not any(lo < b <= hi for lo, hi in spans)]


def clean_control(n: int, links) -> Restructured:
    safe = safe_boundaries(n, links)
    if not safe:
        return Restructured([list(range(n))])
    want = 2 if n >= 9 and len(safe) >= 2 else 1
    # spread the cuts: pick the safe boundaries closest to even thirds/halves
    targets = [round(n * (k + 1) / (want + 1)) for k in range(want)]
    cuts = sorted({min(safe, key=lambda b: abs(b - t)) for t in targets})
    chunks, start = [], 0
    for c in cuts + [n]:
        chunks.append(list(range(start, c)))
        start = c
    return Restructured([c for c in chunks if c])


def move(R: Restructured, u: int, chunk: int | None, at: int | str = "end") -> Restructured:
    R = copy.deepcopy(R)
    for c in R.chunks:
        if u in c:
            c.remove(u)
    R.chunks = [c for c in R.chunks if c]
    if chunk is None:                     # new chunk of its own
        if at == "start":
            R.chunks.insert(0, [u])
        else:
            R.chunks.append([u])
        return R
    c = R.chunks[chunk]
    if at == "end":
        c.append(u)
    else:
        c.insert(at, u)
    return R


def corrupt(R0: Restructured, L, ftype: str) -> Restructured | None:
    if ftype in ("F1", "F2", "F3"):
        hc = R0.chunk_of(L.h)
        others = [k for k in range(len(R0.chunks)) if k != hc]
        if others:
            # prefer a LATER chunk, as in the real ChatGPT example
            later = [k for k in others if k > hc]
            return move(R0, L.d, (later or others)[0], "end")
        return move(R0, L.d, None, "end")
    if ftype == "F4":
        hc = R0.chunk_of(L.h)
        R = move(R0, L.d, hc, R0.chunks[hc].index(L.h))
        return R
    if ftype == "F5":
        return move(R0, L.d, None, "end" if L.direction == "fwd" else "start")
    return None


def main(src: str, dst: str, K: int = 2) -> None:
    passages = json.load(open(src))
    out, stats = [], Counter()
    for P in passages:
        units = split_units(P["text"])
        n = len(units)
        links = extract_links(units)
        R0 = clean_control(n, links)
        base = {"passage_id": P["passage_id"], "n_units": n,
                "units": [u["text"] for u in units], "links": [as_dict(L) for L in links]}
        f0 = failures(links, R0, n)
        rec = {**base, "id": f"{P['passage_id']}-clean", "label": "clean",
               "injected": None, "chunks": R0.chunks, "breaks": f0, "text": render(units, R0)}
        assert words(rec["text"]) == words(P["text"]), P["passage_id"]
        out.append(rec)
        stats["clean"] += 1
        stats["clean_with_breaks"] += bool(f0)
        # BENIGN controls: move one unit that no detected link touches (neither dependent nor head),
        # so clean and corrupted samples both contain a move; detectors must judge the move's effect.
        touched = {L.d for L in links} | {L.h for L in links if L.h is not None}
        free = [u for u in range(n) if u not in touched and (u + 1 >= n or (u + 1) not in touched or
                not any(L.d == u + 1 for L in links))]
        random.shuffle(free)
        nb = 0
        for u in free:
            if nb >= K:
                break
            cands = []
            for k in range(len(R0.chunks)):
                if k != R0.chunk_of(u):
                    cands.append(move(R0, u, k, "end"))
            cands.append(move(R0, u, None, "end" if u < n - 1 else "start"))
            for R in cands:
                if R.order() == R0.order() and R.chunks == R0.chunks:
                    continue
                if failures(links, R, n):
                    continue
                rec = {**base, "id": f"{P['passage_id']}-benign-{nb}", "label": "benign",
                       "injected": {"d": u, "h": None, "type": "benign", "cue": "", "direction": ""},
                       "chunks": R.chunks, "breaks": [], "text": render(units, R)}
                assert words(rec["text"]) == words(P["text"]), rec["id"]
                out.append(rec)
                stats["benign"] += 1
                nb += 1
                break
        ltype = {"F1": "conn", "F2": "ana", "F3": "qual", "F4": "ord", "F5": "xref"}
        for ftype, lt in ltype.items():
            cands = [L for L in links if L.type == lt]
            random.shuffle(cands)
            for k, L in enumerate(cands[:K]):
                R = corrupt(R0, L, ftype)
                if R is None:
                    continue
                br = failures(links, R, n)
                ok = any(b["type"] == ftype and b["d"] == L.d for b in br)
                if not ok:
                    stats[f"{ftype}_not_realised"] += 1
                    continue
                rec = {**base, "id": f"{P['passage_id']}-{ftype}-{k}", "label": ftype,
                       "injected": as_dict(L), "chunks": R.chunks, "breaks": br,
                       "text": render(units, R)}
                assert words(rec["text"]) == words(P["text"]), rec["id"]
                out.append(rec)
                stats[ftype] += 1
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} samples to {dst}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)

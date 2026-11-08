"""Core data structures for structural-fidelity analysis.

A source passage S is split into units (sentences, or clauses split at a
connective/qualifier). A restructured text R is an ordered list of chunks,
each an ordered list of unit indices. Dependency links e = (d, h, type) are
extracted from S; break predicates sep / inv / reatt are evaluated on R.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- segmentation
_ABBR = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bU\.S)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bvs)(?<!\bca)"
_SENT_SPLIT = re.compile(_ABBR + r"(?<=[.!?])[\"”’)]?\s+(?=[\"“(]?[A-Z0-9])")

# clause-level split points (the clause after the comma depends on the clause before)
CONN_CLAUSE = r"which is why|this is why|which means|so that|and thus"
QUAL_CLAUSE = r"although|unless|except|only when|only if|whereas|but only"
_CLAUSE_SPLIT = re.compile(rf",\s+(?=(?:{CONN_CLAUSE}|{QUAL_CLAUSE})\b)", re.I)


def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        out += [s.strip() for s in _SENT_SPLIT.split(para) if s.strip()]
    return out


def split_units(text: str) -> list[dict]:
    """Return units with their kind: 'sent' or 'clause' (split off a sentence)."""
    units = []
    for s in split_sentences(text):
        parts = _CLAUSE_SPLIT.split(s)
        for k, p in enumerate(parts):
            if k < len(parts) - 1 and p.endswith(","):
                p = p[:-1]
            if k < len(parts) - 1:
                p = p + "."  # clause cut at a comma; words are unchanged
            units.append({"text": p.strip(), "kind": "sent" if k == 0 else "clause",
                          "sent_start": k == 0})
    return units


# ---------------------------------------------------------------- link extraction
CONN_START = re.compile(r"^(however|thus|therefore|consequently|as a result|hence|instead|"
                        r"on the other hand|in this way|in the same way|in addition|for comparison|"
                        r"for example|this is why|that is why|so|then|also|but|yet)\b", re.I)
CONN_CL = re.compile(rf"^({CONN_CLAUSE})\b", re.I)
QUAL_CL = re.compile(rf"^({QUAL_CLAUSE})\b", re.I)
ANA_START = re.compile(r"^(?:[A-Z][a-z]+,\s+)?(it|its|they|their|them|this|these|those|such|he|his|"
                       r"she|her|that part|that)\b", re.I)
ORD_START = re.compile(r"^(second|secondly|next|finally|then|third|lastly)\b", re.I)
DEFINE = re.compile(r"\b(?:is|are) called (?:the |a |an )?([A-Za-z][\w-]*(?: [a-z][\w-]*)?)|"
                    r"\bcalled (?:the |a |an )?([A-Za-z][\w-]*(?: [a-z][\w-]*)?)", re.I)
# positional references to other parts of the document (not physical "above the horizon")
XREF_FWD = re.compile(r"(later in this chapter|later on|you will learn more about this later|see later|"
                      r"(?:let.s look at|we will (?:look at|discuss|describe|examine|return to)|(?:is|are) (?:discussed|described|explained|examined)|(?:discuss|describe|explain|examine)(?:ed)? (?:later|below))[^.]{0,60}in more detail|the following(?![^:]*:)|(?:see|shown|described|listed|discussed) below|"
                      r"\(see below\))", re.I)
XREF_BWD = re.compile(r"(previous section|described earlier|mentioned earlier|just mentioned|"
                      r"earlier in this|(?:see|shown|described|listed|discussed|mentioned) above)", re.I)


@dataclass
class Link:
    d: int            # dependent unit
    h: int | None     # head unit (None for xref: target is a layout position)
    type: str         # conn | ana | qual | ord | xref
    cue: str
    direction: str = ""  # xref only: fwd | bwd


def extract_links(units: list[dict]) -> list[Link]:
    links: list[Link] = []
    for i, u in enumerate(units):
        t = u["text"]
        if i > 0:
            m = CONN_CL.match(t) or (CONN_START.match(t) if u["sent_start"] else None)
            if m:
                links.append(Link(i, i - 1, "conn", m.group(0).lower()))
            m = QUAL_CL.match(t) if u["kind"] == "clause" else None
            if m:
                links.append(Link(i, i - 1, "qual", m.group(0).lower()))
            m = ANA_START.match(t) if u["sent_start"] else None
            if m:
                links.append(Link(i, i - 1, "ana", m.group(1).lower()))
            m = ORD_START.match(t) if u["sent_start"] else None
            if m:
                links.append(Link(i, i - 1, "ord", m.group(0).lower()))
        m = XREF_FWD.search(t) or XREF_BWD.search(t)
        if m:
            links.append(Link(i, None, "xref", m.group(0).lower(),
                              "fwd" if XREF_FWD.search(t) else "bwd"))
    # definitions: a later unit that uses a defined term depends on the defining unit
    for i, u in enumerate(units):
        for m in DEFINE.finditer(u["text"]):
            term = (m.group(1) or m.group(2) or "").strip().lower()
            if len(term) < 4:
                continue
            for j in range(i + 1, len(units)):
                if re.search(rf"\b{re.escape(term)}\b", units[j]["text"], re.I):
                    links.append(Link(j, i, "ord", f"defined term: {term}"))
                    break
    # de-duplicate (same d, h, type)
    seen, out = set(), []
    for L in links:
        k = (L.d, L.h, L.type)
        if k not in seen:
            seen.add(k)
            out.append(L)
    return out


# ---------------------------------------------------------------- restructured text
@dataclass
class Restructured:
    chunks: list[list[int]]

    def order(self) -> list[int]:
        return [u for c in self.chunks for u in c]

    def chunk_of(self, u: int) -> int:
        return next(k for k, c in enumerate(self.chunks) if u in c)

    def pos(self, u: int) -> int:
        return self.order().index(u)

    def prev_in_chunk(self, u: int) -> int | None:
        c = self.chunks[self.chunk_of(u)]
        i = c.index(u)
        return c[i - 1] if i > 0 else None


def broken(L: Link, R: Restructured, n_units: int) -> dict:
    """Evaluate the three break predicates for one link on a restructured text."""
    if L.type == "xref":
        p = R.pos(L.d)
        # forward reference broken if nothing follows it in R (but something did in S);
        # backward reference broken if nothing precedes it in R (but something did in S)
        if L.direction == "fwd":
            b = p == n_units - 1 and L.d != n_units - 1
        else:
            b = p == 0 and L.d != 0
        # also broken if it was moved into a different chunk than all its original neighbours
        nb = [x for x in (L.d - 1, L.d + 1) if 0 <= x < n_units]
        iso = all(R.chunk_of(x) != R.chunk_of(L.d) for x in nb) and len(R.chunks) > 1
        return {"sep": iso, "inv": False, "reatt": False, "xref": b or iso}
    sep = R.chunk_of(L.d) != R.chunk_of(L.h)
    inv = R.pos(L.d) < R.pos(L.h)
    reatt = R.prev_in_chunk(L.d) not in (L.h,) and not sep
    return {"sep": sep, "inv": inv, "reatt": reatt}


RULES = {  # failure type -> (link type, predicates that count as a break)
    "F1": ("conn", ("sep", "inv", "reatt")),
    "F2": ("ana", ("sep", "inv")),
    "F3": ("qual", ("sep",)),
    "F4": ("ord", ("inv",)),
    "F5": ("xref", ("xref",)),
}


def failures(links: list[Link], R: Restructured, n_units: int) -> list[dict]:
    out = []
    for ftype, (ltype, preds) in RULES.items():
        for L in links:
            if L.type != ltype:
                continue
            b = broken(L, R, n_units)
            hit = [p for p in preds if b.get(p)]
            if hit:
                out.append({"type": ftype, "d": L.d, "h": L.h, "cue": L.cue, "how": hit})
    return out


def render(units: list[dict], R: Restructured) -> str:
    """Bulleted chunks separated by blank lines; no headings, so every word is the source's."""
    return "\n\n".join("\n".join("• " + units[u]["text"] for u in c) for c in R.chunks)


def as_dict(L: Link) -> dict:
    return asdict(L)

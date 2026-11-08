"""Context-Shift Detector (CSD) for structural fidelity failures.

Idea: restructuring keeps the words, so every segment of the restructured text R
can be located exactly in the source S. A segment that now follows a different
neighbour than in S (or starts a new chunk) has lost its original left context.
Whether that loss matters is measured with a language model, not with cue lists:

    shift(r) = mean log P(r | ctx_S(r)) - mean log P(r | ctx_R(r))

ctx_S(r) = the source text right before r; ctx_R(r) = the text right before r in
R (previous segment in the same chunk, else the chunk heading, else nothing).
A large positive shift means r was predictable from its original neighbour and
is not from its new one: a dependency was cut. The text score is the maximum
shift; the arg-max segment localises the break.

The scorer is pluggable: `HFScorer` (causal LM, run on Colab) or `OverlapScorer`
(word overlap, only for testing the plumbing without a GPU).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from structfid import split_sentences

TOK = re.compile(r"[\w’'-]+")
CTX_WORDS = 60  # max words of left context


def _norm(t: str) -> str:
    return t.lower().replace("’", "'")


# ------------------------------------------------------------------ R segmentation
def segment_restructured(md: str) -> list[dict]:
    """Sentences of R with their chunk (section) id and the section heading.
    Headings: Markdown '#' lines or bold-only lines. If R has no headings,
    blank-line-separated groups are the chunks."""
    lines = md.replace("\r", "").split("\n")
    has_head = any(re.match(r"^\s*#{1,6}\s", l) or re.fullmatch(r"\s*\*\*[^*]+\*\*:?\s*", l) for l in lines)
    segs, sec, heading, seen = [], 0, "", False
    for raw in lines:
        line = raw.strip()
        if not line:
            if not has_head and seen:
                sec += 1; seen = False; heading = ""
            continue
        is_head = bool(re.match(r"^#{1,6}\s", line)) or bool(re.fullmatch(r"\*\*[^*]+\*\*:?", line))
        text = re.sub(r"^#{1,6}\s+|^\\?([-*•]|\d+[.)])\s+", "", line).replace("**", "").replace("__", "").strip()
        if is_head:
            if seen:
                sec += 1; seen = False
            heading = text
            continue
        for s in split_sentences(text):
            segs.append({"text": s, "sec": sec, "heading": heading})
        seen = True
    return segs


# ------------------------------------------------------------------ alignment to S
def align_to_source(segs: list[dict], source: str) -> None:
    """Add 'span' = (start_word, end_word) in S to each segment (None if not found).
    Pass 1: exact word-sequence matches for all segments. Pass 2: fuzzy matches (>= 0.8) for the
    rest, only over unused words, trimmed to the first and last matching word so a rewritten
    sentence cannot swallow words of its neighbours."""
    src_tok = [(_norm(m.group(0)), m.start(), m.end()) for m in TOK.finditer(source)]
    sw = [t for t, _, _ in src_tok]
    used = [False] * len(sw)
    qs = [[_norm(t) for t in TOK.findall(g["text"])] for g in segs]
    for g in segs:
        g["span"] = None
    for g, q in zip(segs, qs):                      # pass 1: exact
        if not q:
            continue
        for s in range(len(sw) - len(q) + 1):
            if sw[s:s + len(q)] == q and not any(used[s:s + len(q)]):
                g["span"] = (s, s + len(q)); break
        if g["span"]:
            for k in range(*g["span"]):
                used[k] = True
    for g, q in zip(segs, qs):                      # pass 2: fuzzy, trimmed
        if g["span"] is not None or not q:
            continue
        best, L = (0.0, None, None), len(q)
        for w in {L, max(1, int(L * 0.8)), int(L * 1.2)}:
            for s in range(0, max(1, len(sw) - w + 1)):
                if any(used[s:s + w]):
                    continue
                sm = SequenceMatcher(None, q, sw[s:s + w], autojunk=False)
                r = sm.ratio()
                if r > best[0]:
                    best = (r, (s, s + w), sm)
        if best[1] and best[0] >= 0.8:
            blocks = [b for b in best[2].get_matching_blocks() if b.size]
            s0 = best[1][0] + blocks[0].b
            e0 = best[1][0] + blocks[-1].b + blocks[-1].size
            g["span"] = (s0, e0)
            for k in range(s0, e0):
                used[k] = True
    for g in segs:
        if g["span"]:
            g["char"] = (src_tok[g["span"][0]][1], src_tok[g["span"][1] - 1][2])
    return None


def _sentence_starts(source: str) -> list[int]:
    starts, pos = [], 0
    for s in split_sentences(source):
        k = source.find(s[:20], pos)
        if k >= 0:
            starts.append(k); pos = k + 1
    return starts or [0]


def contexts(segs: list[dict], source: str) -> None:
    """Fill ctx_S (long), ctx_S1 (previous sentence only), ctx_R and 'moved'."""
    src_tok = list(TOK.finditer(source))
    starts = _sentence_starts(source)
    for i, g in enumerate(segs):
        if g["span"] is None:
            g["moved"] = False
            continue
        s0 = g["span"][0]
        lo = max(0, s0 - CTX_WORDS)
        c0 = src_tok[s0].start()
        g["ctx_S"] = source[src_tok[lo].start():c0].strip() if s0 > 0 else ""
        # previous sentence only: from the start of the sentence that holds the character before r
        # r starts a source sentence -> previous sentence; r starts mid-sentence -> that sentence's beginning
        at_start = any(abs(x - c0) <= 1 for x in starts)
        b = max([x for x in starts if x < c0 - 1] or [0]) if c0 > 0 else 0
        g["ctx_S1"] = source[b:c0].strip().rstrip("(").strip() if (at_start or b < c0) else ""
        prev = next((segs[j] for j in range(i - 1, -1, -1) if segs[j]["sec"] == g["sec"]), None)
        if prev is not None:
            g["ctx_R"] = prev["text"]
            g["moved"] = prev.get("span") is None or prev["span"][1] != s0
        else:
            g["ctx_R"] = g["heading"]
            g["moved"] = s0 != 0
        # keep only the last CTX_WORDS words of the new context too
        w = g["ctx_R"].split()
        g["ctx_R"] = " ".join(w[-CTX_WORDS:])


# ------------------------------------------------------------------ scorers
class OverlapScorer:
    """Plumbing test only: 'log-prob' = log of word overlap between context and segment."""

    def token_logprobs(self, pairs):
        out = []
        for ctx, seg in pairs:
            c = Counter(_norm(t) for t in TOK.findall(ctx))
            out.append([math.log(0.6 if c[_norm(t)] else 0.2) for t in TOK.findall(seg)] or [0.0])
        return out

    def mean_logprob(self, pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        for ctx, seg in pairs:
            c, s = Counter(_norm(t) for t in TOK.findall(ctx)), [_norm(t) for t in TOK.findall(seg)]
            hits = sum(1 for t in s if c[t])
            out.append(math.log((hits + 1) / (len(s) + 2)))
        return out


class HFScorer:
    """Mean per-token log-probability of a segment given a context, from a causal LM."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.bs = batch_size

    def mean_logprob(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [sum(t) / max(1, len(t)) for t in self.token_logprobs(pairs)]

    def token_logprobs(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        """Per-token log-probabilities of each segment given its context."""
        torch, out = self.torch, []
        for b in range(0, len(pairs), self.bs):
            batch = pairs[b:b + self.bs]
            ids, tgt_mask = [], []
            for ctx, seg in batch:
                c = self.tok("\n" + ctx.strip(), add_special_tokens=False)["input_ids"]
                s = self.tok(" " + seg.strip(), add_special_tokens=False)["input_ids"]
                ids.append(c + s); tgt_mask.append([0] * len(c) + [1] * len(s))
            L = max(map(len, ids))
            pad = self.tok.pad_token_id
            x = torch.tensor([i + [pad] * (L - len(i)) for i in ids], device=self.device)
            m = torch.tensor([t + [0] * (L - len(t)) for t in tgt_mask], device=self.device)
            att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i in ids], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=x, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            mm = m[:, 1:].bool()
            for row in range(lp.shape[0]):
                out.append(lp[row][mm[row]].tolist() or [0.0])
        return out


# ------------------------------------------------------------------ detector
@dataclass
class Detection:
    score: float          # max shift over moved segments (0 if nothing moved)
    top_segment: int | None
    segments: list[dict]
    n_moved: int


def detect_many(items: list[tuple[str, str]], scorer) -> list[Detection]:
    """items = [(source_text, restructured_markdown), ...]; one batched LM pass."""
    prepared, pairs, where = [], [], []
    for k, (src, md) in enumerate(items):
        segs = segment_restructured(md)
        align_to_source(segs, src)
        contexts(segs, src)
        prepared.append(segs)
        for i, g in enumerate(segs):
            if g.get("moved"):
                pairs += [(g["ctx_S"], g["text"]), (g["ctx_R"], g["text"])]
                where.append((k, i))
    lps = scorer.mean_logprob(pairs) if pairs else []
    for j, (k, i) in enumerate(where):
        g = prepared[k][i]
        g["lp_S"], g["lp_R"] = lps[2 * j], lps[2 * j + 1]
        g["shift"] = g["lp_S"] - g["lp_R"]
    out = []
    for segs in prepared:
        moved = [i for i, g in enumerate(segs) if g.get("moved")]
        if moved:
            top = max(moved, key=lambda i: segs[i]["shift"])
            out.append(Detection(segs[top]["shift"], top, segs, len(moved)))
        else:
            out.append(Detection(0.0, None, segs, 0))
    return out


# ------------------------------------------------------------------ CSD variants (selected on dev)
def _mean(v, k):
    v = v[:k] if k else v
    return sum(v) / max(1, len(v))


VARIANT_KS = (3, 5, 8, 0)  # 0 = all tokens


def detect_variants(items: list[tuple[str, str]], scorer) -> list[dict]:
    """For every item return {variant_name: text score}. Variants:
       harm[ctx,k] = mean_k lp(r|ctx_S*) - mean_k lp(r|ctx_R)      (lost support)
       need[ctx,k] = mean_k lp(r|ctx_S*) - mean_k lp(r|empty)      (how much r needs its source context)
       nh[ctx,k]   = min(need, harm): needs its context AND the new one does not provide it
       ctx in {long (60 words), sent (previous sentence)}; k in first 3/5/8/all tokens.
       Text score = max over moved segments (0 if nothing moved)."""
    prepared, pairs, where = [], [], []
    for k, (src, md) in enumerate(items):
        segs = segment_restructured(md)
        align_to_source(segs, src)
        contexts(segs, src)
        prepared.append(segs)
        for i, g in enumerate(segs):
            if g.get("moved"):
                pairs += [(g["ctx_S"], g["text"]), (g["ctx_S1"], g["text"]), (g["ctx_R"], g["text"]), ("", g["text"])]
                where.append((k, i))
    tl = scorer.token_logprobs(pairs) if pairs else []
    results = [dict() for _ in items]
    for j, (k, i) in enumerate(where):
        lS, lS1, lR, l0 = tl[4 * j: 4 * j + 4]
        g = prepared[k][i]
        for kk in VARIANT_KS:
            for cname, lc in (("long", lS), ("sent", lS1)):
                harm = _mean(lc, kk) - _mean(lR, kk)
                need = _mean(lc, kk) - _mean(l0, kk)
                for vname, val in (("harm", harm), ("need", need), ("nh", min(need, harm))):
                    key = f"{vname}[{cname},k={kk or 'all'}]"
                    if val > results[k].get(key, (-1e9, None))[0]:
                        results[k][key] = (val, i)
    names = sorted({n for r in results for n in r})
    out = []
    for r in results:
        out.append({n: r.get(n, (0.0, None)) for n in names})
    return out, prepared


# ------------------------------------------------------------------ dependency probe (instruction LM)
PROBE = ("Here are two consecutive pieces of text from a textbook.\n"
         "Text 1: {a}\n"
         "Text 2: {b}\n"
         "Does Text 2 depend on Text 1 to be understood correctly? For example, does it refer back to "
         "something in Text 1, continue or contrast with its argument, or qualify it? Answer Yes or No.")


class ProbeScorer:
    """P(Yes) that segment b depends on context a, from an instruction-tuned LM (Yes/No next-token)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str | None = None, batch_size: int = 16,
                 four_bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if four_bit:
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=q, device_map="auto").eval()
            self.device = self.model.device
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.bs = batch_size
        self.yes = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        self.no = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]

    def p_yes(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.p_yes_prompts([PROBE.format(a=a.strip() or "(start of a new section)", b=s.strip()) for a, s in pairs])

    def p_yes_prompts(self, contents: list[str]) -> list[float]:
        torch, out = self.torch, []
        for b in range(0, len(contents), self.bs):
            prompts = [self.tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                                    add_generation_prompt=True) for c in contents[b:b + self.bs]]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1].float()
            ly = torch.logsumexp(logits[:, self.yes], -1)
            ln = torch.logsumexp(logits[:, self.no], -1)
            out += torch.sigmoid(ly - ln).tolist()
        return out


class OverlapProbe:
    """Plumbing test only."""

    def p_yes_prompts(self, contents):
        return [((len(c) * 7919) % 100) / 100 for c in contents]

    def p_yes(self, pairs):
        out = []
        for a, s in pairs:
            ca = {_norm(t) for t in TOK.findall(a)}
            ws = [_norm(t) for t in TOK.findall(s)]
            out.append(sum(1 for t in ws[:4] if t in ca) / 4)
        return out


def detect_probe(items: list[tuple[str, str]], probe) -> tuple[list[dict], list[list[dict]]]:
    """Variants: need = P(dep | original previous text); harm = need - P(dep | new previous text);
    nh = min(need, harm); ctx = previous sentence ('sent') or 60 words ('long').
    Text score = max over moved segments."""
    prepared, pairs, where = [], [], []
    for k, (src, md) in enumerate(items):
        segs = segment_restructured(md)
        align_to_source(segs, src)
        contexts(segs, src)
        prepared.append(segs)
        for i, g in enumerate(segs):
            if g.get("moved"):
                pairs += [(g["ctx_S1"], g["text"]), (g["ctx_S"], g["text"]), (g["ctx_R"], g["text"])]
                where.append((k, i))
    ps = probe.p_yes(pairs) if pairs else []
    results = [dict() for _ in items]
    for j, (k, i) in enumerate(where):
        pS1, pS, pR = ps[3 * j: 3 * j + 3]
        g = prepared[k][i]
        g["p_dep_S1"], g["p_dep_S"], g["p_dep_R"] = pS1, pS, pR
        for cname, pc in (("sent", pS1), ("long", pS)):
            for vname, val in (("need", pc), ("harm", pc - pR), ("nh", min(pc, pc - pR))):
                key = f"probe-{vname}[{cname}]"
                if val > results[k].get(key, (-1e9, None))[0]:
                    results[k][key] = (val, i)
    names = sorted({n for r in results for n in r}) or ["probe-need[sent]"]
    return [{n: r.get(n, (0.0, None)) for n in names} for r in results], prepared


# ------------------------------------------------------------------ meaning-shift probe (v4)
SHIFT = ("An editor reorganised a textbook passage without changing any words. One sentence now has a "
         "different text before it.\n\n"
         "BEFORE editing, the text right before the sentence was:\n{a}\n\n"
         "AFTER editing, the text right before the sentence is:\n{c}\n\n"
         "The sentence:\n{b}\n\n"
         "Is the sentence understood in exactly the same way after editing? Consider what it refers to, "
         "what it explains or contrasts with, and what it limits. Answer Yes or No.")


def _ctx_R_label(g):
    if g.get("ctx_R"):
        return g["ctx_R"]
    return "(the sentence now starts a new section" + (f' titled "{g["heading"]}")' if g.get("heading") else ")")


def probe_features(items: list[tuple[str, str]], probe) -> tuple[list[dict], list[list[dict]]]:
    """Per moved segment: need (P dep | original prev sentence), fit (P dep | new prev text),
    change = 1 - P(same meaning | before vs after). Per text: features aggregated over moved segments."""
    prepared, prompts, where = [], [], []
    for k, (src, md) in enumerate(items):
        segs = segment_restructured(md)
        align_to_source(segs, src)
        contexts(segs, src)
        prepared.append(segs)
        for i, g in enumerate(segs):
            if g.get("moved"):
                a = g["ctx_S1"] or "(start of the passage)"
                prompts += [PROBE.format(a=a, b=g["text"]),
                            PROBE.format(a=_ctx_R_label(g), b=g["text"]),
                            SHIFT.format(a=a, c=_ctx_R_label(g), b=g["text"])]
                where.append((k, i))
    ps = probe.p_yes_prompts(prompts) if prompts else []
    for j, (k, i) in enumerate(where):
        g = prepared[k][i]
        g["need"], g["fit"], g["change"] = ps[3 * j], ps[3 * j + 1], 1 - ps[3 * j + 2]
        g["harm"] = g["need"] - g["fit"]
    feats = []
    for segs in prepared:
        mv = [g for g in segs if g.get("moved")]
        if not mv:
            feats.append({"max_need": 0, "max_harm": 0, "max_change": 0, "max_need_x_change": 0,
                          "mean_change": 0, "n_moved": 0, "frac_moved": 0, "top": None})
            continue
        ix = max(range(len(segs)), key=lambda i: (segs[i].get("need", -1) * segs[i].get("change", 0)) if segs[i].get("moved") else -1)
        feats.append({"max_need": max(g["need"] for g in mv), "max_harm": max(g["harm"] for g in mv),
                      "max_change": max(g["change"] for g in mv),
                      "max_need_x_change": max(g["need"] * g["change"] for g in mv),
                      "mean_change": sum(g["change"] for g in mv) / len(mv),
                      "n_moved": len(mv), "frac_moved": len(mv) / max(1, len(segs)), "top": ix})
    return feats, prepared

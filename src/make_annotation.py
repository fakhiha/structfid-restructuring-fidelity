"""Blind annotation sheet: 100 real moved sentences (stratified by source model), shown with their
original and new context. Model names, detector scores and judge labels are hidden (key kept separately)."""
import json, random, sys
sys.path.insert(0, "src")
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from detector import segment_restructured, align_to_source, contexts

random.seed(2026)
P = {p["passage_id"]: p for p in json.load(open("data/openstax_passages.json"))}
items = []
for line in open("data/real_outputs.jsonl"):
    r = json.loads(line)
    segs = segment_restructured(r["output_text"]); src = P[r["passage_id"]]["text"]
    align_to_source(segs, src); contexts(segs, src)
    for i, g in enumerate(segs):
        if g.get("moved"):
            prev = next((segs[j]["text"] for j in range(i - 1, -1, -1) if segs[j]["sec"] == g["sec"]), "")
            items.append({"out_id": r["id"], "app": r["app"], "passage_id": r["passage_id"], "seg": i,
                          "target": g["text"], "orig_before": g["ctx_S1"] or "(start of the passage)",
                          "heading": g["heading"] or "(no heading)", "new_before": prev or "(first sentence of this section)",
                          "output_text": r["output_text"]})
per = {"ChatGPT": 34, "Claude": 33, "Gemini": 33}
sample = []
for app, n in per.items():
    pool = [x for x in items if x["app"] == app]
    sample += random.sample(pool, min(n, len(pool)))
random.shuffle(sample)
for k, x in enumerate(sample, 1):
    x["item"] = f"S{k:03d}"
json.dump([{k: x[k] for k in ("item", "out_id", "app", "passage_id", "seg")} for x in sample],
          open("data/annotation_key.json", "w"), indent=1)

F = "Arial"; hdr = Font(name=F, bold=True, color="FFFFFF"); hfill = PatternFill("solid", start_color="4B3B8F")
inp = PatternFill("solid", start_color="FFF9DB"); tgt = PatternFill("solid", start_color="E8F0FE")
thin = Side(style="thin", color="CCCCCC"); bd = Border(left=thin, right=thin, top=thin, bottom=thin)
wrap = Alignment(wrap_text=True, vertical="top")
wb = Workbook()
ins = wb.active; ins.title = "Instructions"
ins.column_dimensions["A"].width = 28; ins.column_dimensions["B"].width = 110
rows = [("Annotation: is the moved sentence broken?", ""), ("", ""),
 ("Task", "Each row is a sentence that an LLM moved when restructuring a passage. Compare how it reads AFTER its original text vs. AFTER its new text (and under its new heading). Decide if its meaning is broken or changed by the move."),
 ("Fill in", "Light-yellow columns only: broken (Y/N), type (if Y), confidence (1 = unsure, 3 = sure), notes (optional)."),
 ("Blind", "The model that wrote each output, the detector's scores and the judges' answers are hidden on purpose. Please do not look them up while labelling."),
 ("Full text", "If the context columns are not enough, the 'Full outputs' sheet has the whole restructured text for every passage/output (match by output_ref)."),
 ("", ""), ("Y = broken if ...", ""),
 ("F1 Detached connective", "A link expression (which is why, however, for example, as a result, thus) now attaches to the wrong text, or its partner is gone."),
 ("F2 Orphaned anaphor", "A pronoun or 'this/these ...' has no clear referent before it in its new section, or now seems to point to something else."),
 ("F3 Orphaned qualifier", "A condition, exception or limitation is separated from the claim it limits (the claim now reads as more general)."),
 ("F4 Order inversion", "It now comes before something it depends on (a definition, an earlier step, a cause)."),
 ("F5 Broken cross-reference", "'Above', 'below', 'later', 'the following', 'previous section' no longer point to the right place."),
 ("Other", "Broken in another way caused by the move (describe in notes)."),
 ("N = not broken", "It reads the same as in the original, even if it moved (e.g. a self-contained sentence placed under a fitting heading)."),
 ("Pace", "About 20-40 seconds per row. If genuinely unsure, pick your best guess and set confidence = 1.")]
for r_, (a, b) in enumerate(rows, 1):
    ins.cell(r_, 1, a).font = Font(name=F, bold=True, size=10); c = ins.cell(r_, 2, b); c.font = Font(name=F, size=10); c.alignment = wrap
ins["A1"].font = Font(name=F, bold=True, size=14)

ws = wb.create_sheet("Annotate")
cols = [("item", 7), ("output_ref", 10), ("BEFORE: original previous text", 45), ("TARGET sentence", 45),
        ("AFTER: new heading", 22), ("AFTER: new previous text", 45), ("broken (Y/N)", 9), ("type", 11),
        ("confidence (1-3)", 10), ("notes", 30)]
for i, (h, w) in enumerate(cols, 1):
    c = ws.cell(1, i, h); c.font = hdr; c.fill = hfill; c.alignment = Alignment(wrap_text=True, vertical="center"); c.border = bd
    ws.column_dimensions[c.column_letter].width = w
ws.freeze_panes = "D2"
refs = {}
for x in sample:
    refs.setdefault(x["out_id"], f"O{len(refs)+1:02d}")
dv_y = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
dv_t = DataValidation(type="list", formula1='"F1,F2,F3,F4,F5,Other"', allow_blank=True)
dv_c = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
for dv in (dv_y, dv_t, dv_c): ws.add_data_validation(dv)
for k, x in enumerate(sample, 2):
    vals = [x["item"], refs[x["out_id"]], x["orig_before"], x["target"], x["heading"], x["new_before"], None, None, None, None]
    for j, v in enumerate(vals, 1):
        c = ws.cell(k, j, v); c.font = Font(name=F, size=10, bold=(j == 4)); c.alignment = wrap; c.border = bd
        if j == 4: c.fill = tgt
        if j >= 7: c.fill = inp
    dv_y.add(f"G{k}"); dv_t.add(f"H{k}"); dv_c.add(f"I{k}")
    ws.row_dimensions[k].height = 95
n = len(sample) + 1
pr = wb.create_sheet("Progress")
pr["A1"], pr["B1"] = "Labelled", f'=COUNTA(Annotate!G2:G{n})'
pr["A2"], pr["B2"] = "Total", len(sample)
pr["A3"], pr["B3"] = "Marked broken", f'=COUNTIF(Annotate!G2:G{n},"Y")'
for c in ("A1", "A2", "A3"): pr[c].font = Font(name=F, bold=True)
full = wb.create_sheet("Full outputs")
full.column_dimensions["A"].width = 10; full.column_dimensions["B"].width = 120
full["A1"], full["B1"] = "output_ref", "restructured text (as pasted)"
for c in ("A1", "B1"): full[c].font = hdr; full[c].fill = hfill
seen = {}
for x in sample: seen[refs[x["out_id"]]] = x["output_text"]
for k, (ref, t) in enumerate(sorted(seen.items()), 2):
    full.cell(k, 1, ref).font = Font(name=F, size=10)
    c = full.cell(k, 2, t); c.font = Font(name=F, size=10); c.alignment = wrap
json.dump(refs, open("data/annotation_refs.json", "w"), indent=1)
wb.move_sheet("Progress", offset=-2)
wb.save("/home/claude/out/annotation_sheet.xlsx")
print(len(sample), "items;", len(refs), "outputs")

"""Readable error report for the fine-tuned model on the test split."""
import json, re, collections
from src.data.prompts import INCLUSION_LABELS

recs = {}
for l in open("data/processed/test.jsonl", encoding="utf-8"):
    r = json.loads(l); recs[r["annotation_id"]] = r
preds = [json.loads(l) for l in open("results/lora_r16_v1_test_preds.jsonl", encoding="utf-8")]

def sents(note):
    d, last = {}, None
    for line in note.split("\n"):
        m = re.match(r"^\s*(\d+)\.\s?(.*)$", line)
        if m: last = int(m.group(1)); d[last] = m.group(2)
        elif last is not None and line.strip(): d[last] += " " + line.strip()
    return d

errs = [p for p in preds if p["pred_label"] != p["gold_label"]]
print("label errors:", len(errs), "of", len(preds))
print(collections.Counter((e["gold_label"], e["pred_label"]) for e in errs).most_common())
lines = []
for e in errs:
    r = recs[e["annotation_id"]]; s = sents(r["note"])
    lines.append(f"=== annotation {e['annotation_id']} | {e['criterion_type']} | gold={e['gold_label']} | pred={e['pred_label']}\n"
                 f"criterion: {r['criterion_text'][:400]}\n"
                 f"gold evidence {e['gold_ids']}: {' | '.join(s.get(i,'?') for i in e['gold_ids'])[:400]}\n"
                 f"pred evidence {e['pred_ids']}: {' | '.join(s.get(i,'?') for i in e['pred_ids'])[:400]}\n")
open("results/test_error_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("wrote results/test_error_report.txt")

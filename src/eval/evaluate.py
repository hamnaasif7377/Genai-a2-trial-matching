"""Greedy evaluation of base model or base+LoRA adapter on a split. Saves predictions + metrics."""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.prompts import build_messages, parse_output
from src.eval.metrics import full_report

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test", "train"])
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--out", required=True, help="output prefix, e.g. results/zeroshot_val")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max_new_tokens", type=int, default=64)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
model.eval()

recs = [json.loads(l) for l in open(f"data/processed/{a.split}.jsonl", encoding="utf-8")]
if a.limit:
    recs = recs[:a.limit]
prompts = [tok.apply_chat_template(build_messages(r["note"], r["criterion_type"], r["criterion_text"]),
                                   tokenize=False, add_generation_prompt=True) for r in recs]
raw, t0 = [], time.time()
for i in range(0, len(prompts), a.bs):
    enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    raw += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"{min(i + a.bs, len(prompts))}/{len(prompts)}", end="\r")
print(f"\ndone in {time.time() - t0:.0f}s")

preds = [parse_output(t) for t in raw]
rep = full_report(recs, preds)
rep["settings"] = {"model": a.model, "adapter": a.adapter, "split": a.split, "decoding": "greedy",
                   "max_new_tokens": a.max_new_tokens, "dtype": "fp16"}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(a.out + "_metrics.json", "w"), indent=2)
with open(a.out + "_preds.jsonl", "w", encoding="utf-8") as f:
    for r, t, p in zip(recs, raw, preds):
        f.write(json.dumps({"annotation_id": r["annotation_id"], "criterion_type": r["criterion_type"],
                            "gold_label": r["label"], "gold_ids": r["evidence_ids"],
                            "raw": t, "pred_label": p[0], "pred_ids": p[1], "valid": p[2]}) + "\n")
for k in ("overall", "inclusion", "exclusion"):
    print(k, "acc=%.3f macroF1=%.3f evid_macroF1=%.3f" % (
        rep[k]["label"]["accuracy"], rep[k]["label"]["macro_f1"], rep[k]["evidence"]["macro_f1"]))
print("json valid rate:", rep["json_valid_rate"])

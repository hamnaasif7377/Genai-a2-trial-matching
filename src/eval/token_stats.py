"""Token-length statistics to document truncation risk."""
import json, numpy as np
from transformers import AutoTokenizer
from src.data.prompts import build_messages, build_target

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
res = {}
for s in ("train", "val", "test"):
    recs = [json.loads(l) for l in open(f"data/processed/{s}.jsonl", encoding="utf-8")]
    L = [len(tok.apply_chat_template(build_messages(r["note"], r["criterion_type"], r["criterion_text"]),
                                     add_generation_prompt=True, tokenize=True, return_dict=False))
         + len(tok(build_target(r["label"], r["evidence_ids"]))["input_ids"]) + 1 for r in recs]
    res[s] = {"n": len(L), "mean": float(np.mean(L)), "p95": float(np.percentile(L, 95)),
              "max": int(max(L)), "over_1024": int(sum(l > 1024 for l in L)),
              "over_2048": int(sum(l > 2048 for l in L))}
    print(s, res[s])
json.dump(res, open("results/token_stats.json", "w"), indent=2)

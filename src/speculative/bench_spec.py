"""Task 5: speculative vs ordinary cached greedy decoding on real validation prompts."""
import argparse, json, time
from pathlib import Path
import torch
from src.inference.common import load_model, load_split, task_prompt_ids, weights_mb, sync, get_device
from src.inference.kv_decode import greedy_cached
from src.speculative.spec_decode import speculative_greedy

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--target_adapter", default=None)
ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--draft_adapter", default=None)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--max_new", type=int, default=64)
ap.add_argument("--split", default="val")
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok, target = load_model(a.target, a.target_adapter, a.dtype)
tok_d, draft = load_model(a.draft, a.draft_adapter, a.dtype)
compat = tok.get_vocab() == tok_d.get_vocab() and target.config.vocab_size == draft.config.vocab_size
print("tokenizer/vocab compatible:", compat)
assert compat, "draft and target must share a tokenizer"
dev, eos, cuda = get_device(), tok.eos_token_id, get_device() == "cuda"
prompts = [task_prompt_ids(tok, r, dev) for r in load_split(a.split)[:a.n]]


def run(fn):
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    sync(); t = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t, (torch.cuda.max_memory_allocated() / 2**20 if cuda else None)


greedy_cached(target, prompts[0], 16, eos)                      # warm-up
speculative_greedy(target, draft, prompts[0], 16, 4, eos)
base = [run(lambda p=p: greedy_cached(target, p, a.max_new, eos, False)["tokens"]) for p in prompts]
base_time = sum(b[1] for b in base)
base_tokens = sum(len(b[0]) for b in base)
res = {"target": a.target, "target_adapter": a.target_adapter, "draft": a.draft, "draft_adapter": a.draft_adapter,
       "dtype": a.dtype, "n_prompts": len(prompts), "max_new": a.max_new,
       "target_weights_MB": weights_mb(target), "draft_weights_MB": weights_mb(draft),
       "baseline": {"time_s": base_time, "tokens": base_tokens, "tok_per_s": base_tokens / base_time,
                    "peak_mem_MB": max(b[2] for b in base) if cuda else None}, "spec": []}
print(f"baseline: {base_tokens / base_time:.1f} tok/s ({base_tokens} tokens, {base_time:.1f}s)")

for k in a.ks:
    runs = [run(lambda p=p: speculative_greedy(target, draft, p, a.max_new, k, eos)) for p in prompts]
    equal = [r[0][0] == b[0] for r, b in zip(runs, base)]
    mism = [i for i, e in enumerate(equal) if not e]
    S = {key: sum(r[0][1][key] for r in runs) for key in ("rounds", "drafted", "accepted", "target_fwd", "draft_fwd")}
    toks = sum(len(r[0][0]) for r in runs)
    t = sum(r[1] for r in runs)
    row = {"k": k, "identical": f"{sum(equal)}/{len(equal)}", "mismatch_indices": mism,
           "acceptance_rate": S["accepted"] / max(S["drafted"], 1),
           "tokens_per_round": toks / S["rounds"], "target_forwards_per_token": S["target_fwd"] / toks,
           "tok_per_s": toks / t, "time_s": t, "speedup_vs_greedy": base_time / t,
           "peak_mem_MB": max(r[2] for r in runs) if cuda else None, **S}
    res["spec"].append(row)
    print(f"k={k}: identical {row['identical']} | accept {row['acceptance_rate']:.2f} | "
          f"{row['tokens_per_round']:.2f} tok/round | {row['tok_per_s']:.1f} tok/s | speedup x{row['speedup_vs_greedy']:.2f}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out + ".json", "w"), indent=2)
print("saved", a.out + ".json")

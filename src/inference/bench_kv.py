"""Task 3: cached vs uncached greedy decoding. Correctness (token equality, logit drift) + benchmark."""
import argparse, json, statistics
from pathlib import Path
import torch
from src.inference.common import load_model, load_split, task_prompt_ids, synthetic_prompt_ids, weights_mb, get_device
from src.inference.kv_decode import greedy_nocache, greedy_cached, kv_cache_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--out", required=True)
ap.add_argument("--in_lens", type=int, nargs="+", default=[128, 512, 1024])
ap.add_argument("--out_lens", type=int, nargs="+", default=[32, 128])
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--n_real", type=int, default=20)
a = ap.parse_args()

tok, model = load_model(a.model, a.adapter, a.dtype)
dev, eos, cuda = get_device(), tok.eos_token_id, get_device() == "cuda"
recs = load_split("val")


def compare(ids, N, ignore_eos, label):
    r = greedy_nocache(model, ids, N, eos, ignore_eos, record_logits=8)
    c = greedy_cached(model, ids, N, eos, ignore_eos, record_logits=8)
    rt, ct = r["tokens"], c["tokens"]
    first = next((i for i, (x, y) in enumerate(zip(rt, ct)) if x != y), None)
    if first is None and len(rt) != len(ct):
        first = min(len(rt), len(ct))
    n = min(len(r["logits"]), len(c["logits"]))
    drift = max((r["logits"][i] - c["logits"][i]).abs().max().item() for i in range(n)) if n else 0.0
    row = {"case": label, "input_len": ids.shape[1], "new_tokens": len(rt), "identical": first is None,
           "first_divergence": first, "max_logit_abs_diff_first8": drift,
           "margin_at_divergence": (r["margins"][first] if first is not None and first < len(r["margins"]) else None)}
    print(row)
    return row


correctness = []
for L in [64, 256, 1024]:
    for N in [16, 64]:
        correctness.append(compare(synthetic_prompt_ids(tok, L, recs, dev), N, True, f"synthetic L={L} N={N}"))
for rec in recs[:a.n_real]:
    correctness.append(compare(task_prompt_ids(tok, rec, dev), 64, False, f"val annotation {rec['annotation_id']}"))
n_same = sum(r["identical"] for r in correctness)
print(f"\nTOKEN-IDENTICAL: {n_same}/{len(correctness)}")


def timed(fn, ids, N):
    if cuda:
        import torch as T
        T.cuda.empty_cache(); T.cuda.reset_peak_memory_stats()
    out = fn(model, ids, N, eos, True)           # ignore_eos: exactly N tokens, comparable across paths
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    return out, peak


bench = []
for L in a.in_lens:
    ids = synthetic_prompt_ids(tok, L, recs, dev)
    for N in a.out_lens:
        row = {"input_len": L, "new_tokens": N, "cache_MB_analytic": kv_cache_bytes(model, L + N) / 2**20}
        for name, fn in (("nocache", greedy_nocache), ("cached", greedy_cached)):
            timed(fn, ids, N)                    # warm-up
            runs = [timed(fn, ids, N) for _ in range(a.repeats)]
            pre = statistics.median(o["step_times"][0] for o, _ in runs)
            dec = statistics.median(sum(o["step_times"][1:]) for o, _ in runs)
            row[name] = {"prefill_ms": 1000 * pre, "decode_ms": 1000 * dec,
                         "decode_tok_per_s": (N - 1) / dec, "total_tok_per_s": N / (pre + dec),
                         "peak_mem_MB": statistics.median(p for _, p in runs) if cuda else None}
        row["speedup_total"] = (row["nocache"]["prefill_ms"] + row["nocache"]["decode_ms"]) / \
                               (row["cached"]["prefill_ms"] + row["cached"]["decode_ms"])
        bench.append(row)
        print(f"L={L:5d} N={N:4d} | nocache dec {row['nocache']['decode_tok_per_s']:7.1f} tok/s "
              f"| cached dec {row['cached']['decode_tok_per_s']:7.1f} tok/s | speedup x{row['speedup_total']:.1f} "
              f"| peak MB {row['nocache']['peak_mem_MB']} vs {row['cached']['peak_mem_MB']}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"model": a.model, "adapter": a.adapter, "dtype": a.dtype, "device": dev,
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "weights_MB": weights_mb(model),
           "token_identical": f"{n_same}/{len(correctness)}", "correctness": correctness, "benchmark": bench},
          open(a.out + ".json", "w"), indent=2)
print("saved", a.out + ".json")

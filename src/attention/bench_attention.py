"""Speed/memory: reference vs tiled (educational PyTorch) vs torch SDPA (fused backend)."""
import json, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from src.attention.attention import reference_attention, tiled_attention

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, H, D = 1, 8, 64
lengths = [128, 256, 512, 1024, 2048]
bq = bk = 128
impls = {
    "reference": lambda q, k, v: reference_attention(q, k, v, True),
    "tiled": lambda q, k, v: tiled_attention(q, k, v, True, bq, bk),
    "torch_sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
}
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
res = []
for L in lengths:
    q, k, v = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
    for name, fn in impls.items():
        for _ in range(2): fn(q, k, v)                      # warm-up
        sync()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        ts = []
        for _ in range(5):
            sync(); t = time.perf_counter(); fn(q, k, v); sync(); ts.append(time.perf_counter() - t)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev == "cuda" else None
        score_mb = {"reference": B * H * L * L * 4, "tiled": B * H * bq * bk * 4, "torch_sdpa": float("nan")}[name] / 2**20
        res.append({"L": L, "impl": name, "median_ms": 1000 * sorted(ts)[len(ts) // 2],
                    "peak_extra_mem_MB": peak, "score_buffer_MB_analytic": score_mb})
        print(f"L={L:5d} {name:11s} {res[-1]['median_ms']:9.2f} ms  peak={peak}  score_buf={score_mb:.1f}MB")
Path("results").mkdir(exist_ok=True)
tag = "gpu" if dev == "cuda" else "cpu"
json.dump({"device": dev, "torch": torch.__version__, "B": B, "H": H, "D": D, "blocks": [bq, bk], "rows": res},
          open(f"results/attention_bench_{tag}.json", "w"), indent=2)

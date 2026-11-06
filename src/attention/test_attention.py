"""Numerical validation: reference vs tiled vs torch SDPA. float32 and float64, several shapes,
causal / non-causal, normal and large-magnitude inputs."""
import json, itertools
from pathlib import Path
import torch
import torch.nn.functional as F
from src.attention.attention import reference_attention, tiled_attention

torch.manual_seed(0)
shapes = [  # (B, H, Lq, Lk, D)
    (1, 1, 8, 8, 16), (2, 4, 33, 33, 32), (1, 2, 64, 64, 64), (2, 2, 100, 100, 64),
    (1, 4, 257, 257, 64), (1, 2, 1, 130, 64), (1, 2, 5, 70, 64),   # last two: decode-style Lq < Lk
]
cases = [("normal", 1.0), ("large-magnitude", 30.0)]
# tolerance by dtype/case: float32 rounding is amplified by exp() on large scores
TOL = {("float32", "normal"): 1e-5, ("float32", "large-magnitude"): 1e-3,
       ("float64", "normal"): 1e-10, ("float64", "large-magnitude"): 1e-10}
rows, ok = [], True
for dt, (B, H, Lq, Lk, D), causal, (cname, mag), (bq, bk) in itertools.product(
        [torch.float32, torch.float64], shapes, [True, False], cases, [(16, 16), (64, 32)]):
    q = (torch.randn(B, H, Lq, D, dtype=torch.float64) * mag).to(dt)
    k = (torch.randn(B, H, Lk, D, dtype=torch.float64) * mag).to(dt)
    v = torch.randn(B, H, Lk, D, dtype=torch.float64).to(dt)
    ref = reference_attention(q, k, v, causal)
    til = tiled_attention(q, k, v, causal, bq, bk)
    err = (ref - til).abs()
    sd = None
    if Lq == Lk:
        sd = (F.scaled_dot_product_attention(q, k, v, is_causal=causal) - ref).abs().max().item()
    tol = TOL[(str(dt).split(".")[1], cname)]
    passed = err.max().item() < tol and torch.isfinite(til).all().item()
    ok &= passed
    rows.append({"dtype": str(dt), "shape": [B, H, Lq, Lk, D], "causal": causal, "case": cname,
                 "blocks": [bq, bk], "tolerance": tol, "max_abs_err": err.max().item(),
                 "mean_abs_err": err.mean().item(), "ref_vs_torch_sdpa_max_err": sd, "pass": passed})
    print(f"{'PASS' if passed else 'FAIL'} {str(dt)[6:]:8s} shape={(B,H,Lq,Lk,D)} causal={causal} "
          f"{cname:16s} blocks={(bq,bk)} max={err.max().item():.2e} mean={err.mean().item():.2e}")

q = torch.randn(1, 1, 32, 16); k = torch.randn(1, 1, 32, 16); v = torch.randn(1, 1, 32, 16)
o1 = tiled_attention(q, k, v, True, 8, 8)
k2, v2 = k.clone(), v.clone(); k2[..., 20:, :] += 5; v2[..., 20:, :] += 5
o2 = tiled_attention(q, k2, v2, True, 8, 8)
causal_ok = torch.allclose(o1[..., :20, :], o2[..., :20, :])
print("causality check (outputs before pos 20 unchanged):", causal_ok)
Path("results").mkdir(exist_ok=True)
json.dump({"all_pass": bool(ok and causal_ok), "causality_ok": bool(causal_ok), "cases": rows},
          open("results/attention_tests.json", "w"), indent=2)
print("ALL PASS" if ok and causal_ok else "SOME FAILED")

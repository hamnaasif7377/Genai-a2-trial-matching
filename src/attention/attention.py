"""Reference and tiled (online-softmax) causal attention. Shapes: (B, H, L, D).
Causal convention: query i sits at absolute position i + (Lk - Lq), so a short query
block attending to a longer key/value cache (as in KV-cached decoding) is handled correctly."""
import math
import torch


def reference_attention(q, k, v, causal=True):
    """Materialises the full (Lq x Lk) score matrix. Max-subtracted softmax written out explicitly."""
    Lq, Lk, D = q.shape[-2], k.shape[-2], q.shape[-1]
    cd = torch.promote_types(q.dtype, torch.float32)
    assert Lk >= Lq
    s = (q.to(cd) @ k.to(cd).transpose(-1, -2)) / math.sqrt(D)
    if causal:
        qi = torch.arange(Lq, device=q.device)[:, None] + (Lk - Lq)
        kj = torch.arange(Lk, device=q.device)[None, :]
        s = s.masked_fill(kj > qi, float("-inf"))
    s = s - s.amax(dim=-1, keepdim=True)          # numerical stability
    p = torch.exp(s)
    p = p / p.sum(dim=-1, keepdim=True)
    return (p @ v.to(cd)).to(q.dtype)


def tiled_attention(q, k, v, causal=True, block_q=64, block_k=64):
    """FlashAttention-style forward pass: never builds the full score matrix.
    Per query row keeps a running max m, running normaliser l and running output acc;
    when a new key block raises the max, old statistics are rescaled by exp(m_old - m_new)."""
    B, H, Lq, D = q.shape
    cd = torch.promote_types(q.dtype, torch.float32)
    Lk = k.shape[-2]
    assert Lk >= Lq
    scale = 1.0 / math.sqrt(D)
    offset = Lk - Lq
    out = torch.empty(B, H, Lq, D, dtype=cd, device=q.device)
    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[..., qs:qe, :].to(cd)
        nq = qe - qs
        m = torch.full((B, H, nq), float("-inf"), dtype=cd, device=q.device)
        l = torch.zeros(B, H, nq, dtype=cd, device=q.device)
        acc = torch.zeros(B, H, nq, D, dtype=cd, device=q.device)
        kend = min(Lk, qe + offset) if causal else Lk      # skip key blocks entirely above the diagonal
        for ks in range(0, kend, block_k):
            ke = min(ks + block_k, kend)
            s = (qb @ k[..., ks:ke, :].to(cd).transpose(-1, -2)) * scale
            if causal:
                qi = torch.arange(qs, qe, device=q.device)[:, None] + offset
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # fully masked rows
            p = torch.exp(s - m_safe[..., None])
            alpha = torch.exp(m - m_safe)                  # rescale factor for previous blocks
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[..., None] + p @ v[..., ks:ke, :].to(cd)
            m = m_new
        out[..., qs:qe, :] = acc / l.clamp_min(1e-30)[..., None]
    return out.to(q.dtype)


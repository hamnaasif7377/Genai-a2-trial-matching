"""Greedy speculative decoding, written from scratch (no assisted-generation API).
Invariant at the start of every round: both caches hold seq[:-1]; the last token is 'pending'.
 1. DRAFT  : feed pending token, then each draft token, taking argmax -> d_1..d_k   (k cheap forwards)
 2. VERIFY : ONE target forward over [pending, d_1..d_k]; logits[i] is the target's greedy token that
             follows input i, so d_{i+1} is accepted iff it equals argmax(logits[i]).
 3. ACCEPT : m = length of the matching prefix. Append d_1..d_m plus target's own token at position m
             (the correction if m<k, a free bonus token if m==k). m+1 tokens per round, always >= 1.
 4. SYNC   : target cache grew by k+1 -> crop to the accepted prefix. Draft cache is cropped (rejection)
             or extended by the one unseen token d_k (full acceptance).
The output is the same sequence plain greedy decoding of the target would produce."""
import torch
from transformers import DynamicCache
from src.inference.kv_decode import forward_logits


def trim_cache(cache, length):
    cur = cache.get_seq_length()
    if cur > length:
        cache.crop(length - cur)   # newer transformers: negative = tokens to remove



@torch.no_grad()
def speculative_greedy(target, draft, prompt_ids, max_new, k, eos_id, ignore_eos=False):
    dev = prompt_ids.device
    seq = prompt_ids[0].tolist()
    L0 = len(seq)
    tc, dc = DynamicCache(), DynamicCache()
    forward_logits(target, prompt_ids[:, :-1], tc, 0, 1)
    forward_logits(draft, prompt_ids[:, :-1], dc, 0, 1)
    st = {"rounds": 0, "drafted": 0, "accepted": 0, "target_fwd": 1, "draft_fwd": 1}
    done = False
    while not done and len(seq) - L0 < max_new:
        n = len(seq)
        kk = min(k, max_new - (n - L0) - 1)               # never overshoot max_new
        drafts, cur = [], torch.tensor([[seq[-1]]], device=dev)
        for _ in range(kk):
            lg = forward_logits(draft, cur, dc, dc.get_seq_length(), 1)
            d = int(lg[0, -1].argmax()); drafts.append(d)
            cur = torch.tensor([[d]], device=dev)
            st["draft_fwd"] += 1
        ver = torch.tensor([[seq[-1]] + drafts], device=dev)
        lg = forward_logits(target, ver, tc, tc.get_seq_length(), kk + 1)
        st["target_fwd"] += 1
        pred = lg[0].argmax(-1).tolist()
        m = 0
        while m < kk and drafts[m] == pred[m]:
            m += 1
        new = drafts[:m] + [pred[m]]
        if not ignore_eos and eos_id in new:
            new = new[:new.index(eos_id) + 1]
            done = True
        seq.extend(new)
        st["rounds"] += 1; st["drafted"] += kk; st["accepted"] += m
        if not done:
            trim_cache(tc, n + m)                                # target keeps pending + accepted drafts
            dl, want = dc.get_seq_length(), n + m
            if dl > want:
                trim_cache(dc, want)                             # rejection: drop draft tokens past the match
            elif dl < want:                               # full acceptance: d_k was never fed to the draft
                forward_logits(draft, torch.tensor([seq[dl:want]], device=dev), dc, dl, 1)
                st["draft_fwd"] += 1
    return seq[L0:L0 + max_new], st


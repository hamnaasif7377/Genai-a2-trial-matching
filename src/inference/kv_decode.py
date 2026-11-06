"""Hand-written greedy decoders on top of the bare transformer (model.model) + lm_head.
 - greedy_nocache : reference. Every step re-runs the WHOLE sequence (O(L^2) work per token).
 - greedy_cached  : prefill once, then feed ONE new token per step; we own the cache object,
                    the position ids and the attention mask.
We never call model.generate()."""
import time
import torch
from transformers import DynamicCache
from src.inference.common import sync


@torch.no_grad()
def forward_logits(model, ids, cache, start_pos, n_last):
    """Run `ids` (1, n) whose first token sits at absolute position `start_pos`.
    Positions are explicit (RoPE needs the absolute index, not the index inside this chunk).
    The mask spans cache + new tokens (all ones, batch 1, no padding); causal structure inside the
    new chunk is applied by the model. Returns float32 logits for the last n_last positions only,
    so the vocab projection is not computed for the whole prompt."""
    n = ids.shape[1]
    pos = torch.arange(start_pos, start_pos + n, device=ids.device)[None]
    mask = torch.ones(1, start_pos + n, dtype=torch.long, device=ids.device)
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      past_key_values=cache, use_cache=cache is not None)
    return model.lm_head(out.last_hidden_state[:, -n_last:]).float()


def _step_result(logits_row):
    top = logits_row.topk(2).values
    return int(logits_row.argmax()), float(top[0] - top[1])


@torch.no_grad()
def greedy_nocache(model, prompt_ids, max_new, eos_id, ignore_eos=False, record_logits=0):
    seq, toks, times, margins, logs = prompt_ids, [], [], [], []
    for step in range(max_new):
        sync(); t = time.perf_counter()
        lg = forward_logits(model, seq, None, 0, 1)[0, -1]
        nxt, mg = _step_result(lg)
        sync(); times.append(time.perf_counter() - t)
        toks.append(nxt); margins.append(mg)
        if step < record_logits:
            logs.append(lg.cpu())
        if nxt == eos_id and not ignore_eos:
            break
        seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
    return {"tokens": toks, "step_times": times, "margins": margins, "logits": logs}


@torch.no_grad()
def greedy_cached(model, prompt_ids, max_new, eos_id, ignore_eos=False, record_logits=0):
    cache = DynamicCache()
    toks, times, logs = [], [], []
    sync(); t = time.perf_counter()
    lg = forward_logits(model, prompt_ids, cache, 0, 1)[0, -1]       # prefill: fills the cache
    nxt, _ = _step_result(lg)
    sync(); times.append(time.perf_counter() - t)
    toks.append(nxt)
    if record_logits:
        logs.append(lg.cpu())
    while len(toks) < max_new and not (toks[-1] == eos_id and not ignore_eos):
        sync(); t = time.perf_counter()
        pos = cache.get_seq_length()                                  # == number of tokens already cached
        lg = forward_logits(model, torch.tensor([[toks[-1]]], device=prompt_ids.device), cache, pos, 1)[0, -1]
        nxt, _ = _step_result(lg)
        sync(); times.append(time.perf_counter() - t)
        toks.append(nxt)
        if len(toks) <= record_logits:
            logs.append(lg.cpu())
    return {"tokens": toks, "step_times": times, "logits": logs, "cache_len": cache.get_seq_length()}


def kv_cache_bytes(model, seq_len):
    c = model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    el = next(model.parameters()).element_size()
    return c.num_hidden_layers * 2 * c.num_key_value_heads * head_dim * seq_len * el

"""Screening engine: model loading, input handling, prediction with the project's own decoders,
evidence extraction, templated explanations, rule-based aggregation, live benchmarks."""
import hashlib, os, re, threading, time
from datetime import datetime, timezone
from pathlib import Path
import torch
from src.data.prompts import INCLUSION_LABELS, EXCLUSION_LABELS, parse_output
from src.inference.common import load_model, task_prompt_ids, get_device
from src.inference.kv_decode import greedy_cached, greedy_nocache, kv_cache_bytes
from src.speculative.spec_decode import speculative_greedy

ROOT = Path(__file__).resolve().parents[1]
AR = ROOT.parent / "adapter"
CFG = {
    "base": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    "adapter": os.environ.get("ADAPTER_PATH", str(AR / "lora_r16_v1" / "final_adapter")),
    "draft": os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    "draft_adapter": os.environ.get("DRAFT_ADAPTER_PATH", str(AR / "lora_draft_0p5b_v1" / "final_adapter")),
}
LIMITS = {"note_chars": 8000, "max_sentences": 80, "max_criteria": 20, "criterion_chars": 1500,
          "max_prompt_tokens": 3000, "warn_prompt_tokens": 600}
DISCLAIMER = ("Educational screening prototype. This is not a clinical enrollment decision. "
              "Model outputs can be wrong and must be reviewed by a qualified person.")
AGG_RULE = ("Explicit rule (not a ground-truth label): (1) any inclusion criterion 'not included' or any exclusion "
            "criterion 'excluded' -> Likely ineligible; (2) else any invalid model output -> Needs manual review; "
            "(3) else any 'not enough information' -> Undetermined; (4) else all criteria 'not applicable' -> "
            "No applicable criteria; (5) else Preliminarily eligible. Missing information is never treated as "
            "eligibility or ineligibility.")

NUM = re.compile(r"^\s*(\d+)\.\s?(.*)$")
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
HDR = re.compile(r"^\W*(inclusion|exclusion)\s+criteria\b\W*(.*)$", re.I)
BUL = re.compile(r"^(?:[-*\u2022\u2013]+|\d+[.)])\s+")


def prepare_note(text):
    text = (text or "").replace("\r", "").strip()
    if not text:
        raise ValueError("The patient note is empty. Paste a patient description.")
    if len(text) > LIMITS["note_chars"]:
        raise ValueError(f"The patient note is too long ({len(text)} characters; limit {LIMITS['note_chars']}).")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    m = [NUM.match(l) for l in lines]
    if all(m) and [int(x.group(1)) for x in m] == list(range(len(lines))):
        sents, auto = [x.group(2).strip() for x in m], False
    else:
        flat = " ".join(NUM.sub(r"\2", l) if NUM.match(l) else l for l in lines)
        sents, auto = [s.strip() for s in SPLIT.split(flat) if s.strip()], True
    if not sents:
        raise ValueError("No sentences could be found in the patient note.")
    if len(sents) > LIMITS["max_sentences"]:
        raise ValueError(f"The note has {len(sents)} sentences; the limit is {LIMITS['max_sentences']}.")
    return sents, "\n".join(f"{i}. {s}" for i, s in enumerate(sents)), auto


def parse_criteria(text):
    cur, out = None, []
    for raw in (text or "").replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        h = HDR.match(line)
        if h:
            cur = h.group(1).lower()
            line = h.group(2).strip()
            if not line:
                continue
        if cur is None:
            raise ValueError("Start the criteria with a header line 'Inclusion criteria:' or "
                             "'Exclusion criteria:', then put one criterion per line.")
        line = BUL.sub("", line).strip()
        if len(line) < 3:
            continue
        if len(line) > LIMITS["criterion_chars"]:
            raise ValueError(f"A criterion is too long ({len(line)} characters; limit {LIMITS['criterion_chars']}).")
        out.append((cur, line))
    if not out:
        raise ValueError("No criteria were found. Use 'Inclusion criteria:' / 'Exclusion criteria:' headers "
                         "followed by one criterion per line.")
    if len(out) > LIMITS["max_criteria"]:
        raise ValueError(f"{len(out)} criteria were entered; the limit is {LIMITS['max_criteria']} per run.")
    return out


def explain(label, evidence, valid):
    if not valid:
        return ("No reliable decision: the model output was not a valid label for this criterion type. "
                "Review this criterion manually.")
    base = {
        "included": "The model judged that the patient meets this inclusion criterion.",
        "not included": "The model judged that the patient does not meet this inclusion criterion.",
        "excluded": "The model judged that this exclusion criterion applies to the patient.",
        "not excluded": "The model found nothing in the note indicating that this exclusion criterion applies.",
        "not enough information": ("The note does not contain enough information to decide. Missing information "
                                   "is not treated as evidence of eligibility or of ineligibility."),
        "not applicable": "The model judged that this criterion does not apply to this patient.",
    }[label]
    if evidence:
        return base + " Supporting sentences from the note: " + " ".join(f'[{e["id"]}] "{e["text"]}"' for e in evidence)
    return base + " No supporting sentence was cited."


def aggregate(results):
    counts = {}
    for r in results:
        k = r["label"] if r["label_valid"] else "invalid output"
        counts[k] = counts.get(k, 0) + 1
    short = lambda r: (r["criterion"][:90] + ("..." if len(r["criterion"]) > 90 else ""))
    neg = [r for r in results if r["label_valid"] and (
        (r["type"] == "inclusion" and r["label"] == "not included") or
        (r["type"] == "exclusion" and r["label"] == "excluded"))]
    bad = [r for r in results if not r["label_valid"]]
    nei = [r for r in results if r["label"] == "not enough information"]
    na = [r for r in results if r["label"] == "not applicable"]
    weak = [r for r in results if r["label"] in ("not excluded", "not included", "included", "excluded") and not r["evidence"]]
    reasons = []
    if neg:
        code, out = "likely_ineligible", "Likely ineligible (preliminary)"
        reasons = ["Failed criterion: " + short(r) for r in neg]
    elif bad:
        code, out = "needs_review", "Needs manual review"
        reasons = [f"{len(bad)} criterion/criteria produced no valid decision."]
    elif nei:
        code, out = "undetermined", "Undetermined: information missing"
        reasons = [f"{len(nei)} criterion/criteria cannot be decided from the note."]
    elif results and len(na) == len(results):
        code, out = "no_applicable", "No applicable criteria"
    else:
        code, out = "likely_eligible", "Preliminarily eligible"
        reasons = ["No criterion was failed, undecided or invalid."]
    if weak:
        reasons.append(f"{len(weak)} decision(s) cite no evidence sentence; for exclusion criteria this can simply mean "
                       "the note does not mention the criterion. Verify them manually.")
    return {"outcome": out, "code": code, "reasons": reasons, "counts": counts, "rule": AGG_RULE}


def assemble(meta, results, total_s):
    return {"meta": meta, "criteria": results, "overall": aggregate(results),
            "total_seconds": total_s, "disclaimer": DISCLAIMER}


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return None


class Engine:
    def __init__(self):
        self.tok = self.target = self.draft = None
        self.state, self.error = "loading", None
        self.draft_adapter_used = None
        self.device = get_device()
        self.lock = threading.Lock()

    def load(self):
        try:
            torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 4)))
            if not Path(CFG["adapter"]).exists():
                raise FileNotFoundError("LoRA adapter folder not found: " + CFG["adapter"] +
                                        ". Set ADAPTER_PATH or copy the adapter there.")
            self.tok, self.target = load_model(CFG["base"], CFG["adapter"])
            self.state = "ready"
        except Exception as e:
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def status(self):
        return {"status": self.state, "error": self.error, "device": self.device, "threads": torch.get_num_threads(),
                "base_model": CFG["base"], "adapter": CFG["adapter"], "draft_loaded": self.draft is not None,
                "draft_adapter_found": Path(CFG["draft_adapter"]).exists(), "torch": torch.__version__,
                "limits": LIMITS}

    def ensure_draft(self):
        if self.draft is not None:
            return
        ad = CFG["draft_adapter"] if Path(CFG["draft_adapter"]).exists() else None
        tok_d, d = load_model(CFG["draft"], ad)
        if tok_d.get_vocab() != self.tok.get_vocab():
            raise RuntimeError("Draft and target tokenizers differ; speculative decoding is not valid.")
        self.draft, self.draft_adapter_used = d, ad is not None

    def make_meta(self, sents, numbered, auto, mode, k):
        return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_model": CFG["base"], "adapter": Path(CFG["adapter"]).parent.name,
                "decoding_mode": mode, "speculative_k": k if mode == "speculative" else None,
                "decoding": "greedy, max_new_tokens=64", "device": self.device,
                "dtype": str(next(self.target.parameters()).dtype).replace("torch.", ""),
                "note_sha256": hashlib.sha256(numbered.encode()).hexdigest()[:16],
                "sentences": sents, "note_auto_numbered": auto, "torch": torch.__version__}

    def _decode(self, prompt, mode, k, max_new):
        eos = self.tok.eos_token_id
        if mode == "cached":
            return greedy_cached(self.target, prompt, max_new, eos, False), None
        if mode == "nocache":
            return greedy_nocache(self.target, prompt, max_new, eos, False), None
        toks, st = speculative_greedy(self.target, self.draft, prompt, max_new, k, eos)
        return {"tokens": toks}, st

    def screen_one(self, sents, numbered, ctype, ctext, mode, k, max_new=64):
        prompt = task_prompt_ids(self.tok, {"note": numbered, "criterion_type": ctype, "criterion_text": ctext}, self.device)
        n_prompt = int(prompt.shape[1])
        base = {"type": ctype, "criterion": ctext, "prompt_tokens": n_prompt, "spec_stats": None}
        if n_prompt > LIMITS["max_prompt_tokens"]:
            return {**base, "label": None, "label_valid": False, "evidence_ids": [], "evidence": [], "new_tokens": 0,
                    "seconds": 0.0, "raw_output": "", "explanation": explain(None, [], False),
                    "flags": [f"Prompt has {n_prompt} tokens (limit {LIMITS['max_prompt_tokens']}); criterion skipped."]}
        t = time.perf_counter()
        out, stats = self._decode(prompt, mode, k, max_new)
        dt = time.perf_counter() - t
        raw = self.tok.decode(out["tokens"], skip_special_tokens=True)
        label, ids, valid_json = parse_output(raw)
        allowed = INCLUSION_LABELS if ctype == "inclusion" else EXCLUSION_LABELS
        flags = []
        valid = valid_json and label in allowed
        if not valid_json:
            flags.append("Model output was not valid JSON.")
        elif label not in allowed:
            flags.append(f"Label '{label}' is not a valid {ctype} label.")
        good = sorted({i for i in ids if 0 <= i < len(sents)})
        bad_ids = sorted({i for i in ids if not 0 <= i < len(sents)})
        if bad_ids:
            flags.append(f"Cited sentence IDs {bad_ids} do not exist in the note and were discarded.")
        evidence = [{"id": i, "text": sents[i]} for i in good]
        if valid and label in ("included", "not included", "excluded", "not excluded") and not evidence:
            if label == "not excluded":
                flags.append("Marked 'not excluded' with no cited evidence. This can simply mean the note does not "
                             "mention the criterion. Verify manually.")
            else:
                flags.append("Decision cited no evidence sentence; treat it as unsupported.")
        if n_prompt > LIMITS["warn_prompt_tokens"]:
            flags.append(f"Prompt ({n_prompt} tokens) is longer than any training example (max 562); accuracy may degrade.")
        return {**base, "label": label if valid else (label or None), "label_valid": bool(valid),
                "evidence_ids": good, "evidence": evidence, "explanation": explain(label, evidence, valid),
                "flags": flags, "raw_output": raw, "new_tokens": len(out["tokens"]), "seconds": dt, "spec_stats": stats}

    def benchmark(self, job, note, ctype, ctext, max_new, k, inc_nocache, inc_spec):
        sents, numbered, _ = prepare_note(note)
        prompt = task_prompt_ids(self.tok, {"note": numbered, "criterion_type": ctype, "criterion_text": ctext}, self.device)
        eos = self.tok.eos_token_id
        job["message"] = "Warm-up"
        greedy_cached(self.target, prompt, 2, eos, False)
        plan = [("cached", "KV-cached (own decoder)")]
        if inc_nocache:
            plan.append(("nocache", "Uncached reference"))
        if inc_spec:
            plan.append(("speculative", f"Speculative (k={k})"))
        job["total"], rows, ref = len(plan), [], None
        for i, (mode, label) in enumerate(plan):
            job["message"] = f"Running: {label}"
            try:
                if mode == "speculative":
                    self.ensure_draft()
                t = time.perf_counter()
                out, st = self._decode(prompt, mode, k, max_new)
                dt = time.perf_counter() - t
            except Exception as e:
                rows.append({"mode": mode, "label": label, "tokens": 0, "total_s": 1e-9, "error": str(e),
                             "details": "failed: " + str(e)})
                job["progress"] = i + 1
                continue
            toks = out["tokens"]
            if mode == "cached":
                ref = toks
            sts = out.get("step_times")
            pre = sts[0] if sts else None
            dec = sum(sts[1:]) if sts else None
            det = ""
            if st:
                det = (f"acceptance {st['accepted'] / max(st['drafted'], 1):.2f}, "
                       f"{len(toks) / max(st['rounds'], 1):.2f} tokens/round, {st['target_fwd']} target forwards"
                       + ("" if self.draft_adapter_used else "; draft adapter NOT found (untuned draft)"))
            rows.append({"mode": mode, "label": label, "tokens": len(toks), "prefill_s": pre, "decode_s": dec,
                         "total_s": dt, "tok_per_s": len(toks) / dt, "text": self.tok.decode(toks, skip_special_tokens=True),
                         "identical_to_cached": None if mode == "cached" else (toks == ref),
                         "rss_MB": _rss_mb(), "details": det})
            job["partial"] = rows
            job["progress"] = i + 1
        c = next((r for r in rows if r["mode"] == "cached"), None)
        n = next((r for r in rows if r["mode"] == "nocache" and "error" not in r), None)
        for r in rows:
            r["speedup_vs_cached"] = (c["total_s"] / r["total_s"]) if c and "error" not in r else None
            r["speedup_vs_nocache"] = (n["total_s"] / r["total_s"]) if n and "error" not in r else None
        job["result"] = {"rows": rows, "prompt_tokens": int(prompt.shape[1]), "max_new": max_new, "device": self.device,
                         "threads": torch.get_num_threads(),
                         "kv_cache_MB_analytic": kv_cache_bytes(self.target, int(prompt.shape[1]) + max_new) / 2**20}


def attention_demo(job, L, H, D, block, causal):
    import torch.nn.functional as F
    from src.attention.attention import reference_attention, tiled_attention
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, H, L, D, generator=g) for _ in range(3))
    fns = {"reference": lambda: reference_attention(q, k, v, causal),
           "tiled": lambda: tiled_attention(q, k, v, causal, block, block),
           "torch_sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)}
    outs, ms = {}, {}
    for n, f in fns.items():
        job["message"] = "Running " + n
        outs[n] = f()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        ms[n] = 1000 * sorted(ts)[1]
    e = (outs["reference"] - outs["tiled"]).abs()
    job["result"] = {"L": L, "H": H, "D": D, "block": block, "causal": causal,
                     "max_abs_err": e.max().item(), "mean_abs_err": e.mean().item(),
                     "sdpa_vs_reference_max_err": (outs["torch_sdpa"] - outs["reference"]).abs().max().item(),
                     "ms": ms, "score_buffer_MB": {"reference": H * L * L * 4 / 2**20, "tiled": H * block * block * 4 / 2**20}}

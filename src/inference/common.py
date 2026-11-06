"""Shared helpers: model loading (optionally merging a LoRA adapter), prompts, timing."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.prompts import build_messages


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(name, adapter=None, dtype="float16"):
    dev = get_device()
    if dev == "cpu":
        dtype = "float32"
    dt = {"float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dt)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to(dev).eval()
    return tok, model


def weights_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20


def load_split(split):
    return [json.loads(l) for l in open(f"data/processed/{split}.jsonl", encoding="utf-8")]


def task_prompt_ids(tok, rec, device):
    ids = tok.apply_chat_template(build_messages(rec["note"], rec["criterion_type"], rec["criterion_text"]),
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    return torch.tensor([ids], device=device)


def synthetic_prompt_ids(tok, n_tokens, recs, device):
    """Real clinical text, cut or repeated to exactly n_tokens (for controlled length sweeps)."""
    ids = tok("\n\n".join(r["note"] for r in recs))["input_ids"]
    while len(ids) < n_tokens:
        ids = ids + ids
    return torch.tensor([ids[:n_tokens]], device=device)

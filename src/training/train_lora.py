"""LoRA fine-tuning of Qwen2.5-1.5B-Instruct. Loss only on the answer tokens. Resumable."""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
from src.data.prompts import build_messages, build_target

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max_len", type=int, default=1536)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)


class DS(torch.utils.data.Dataset):
    def __init__(self, split):
        self.items, self.trunc = [], 0
        for l in open(f"data/processed/{split}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            p = tok.apply_chat_template(build_messages(r["note"], r["criterion_type"], r["criterion_text"]),
                                        add_generation_prompt=True, tokenize=True, return_dict=False)
            t = tok(build_target(r["label"], r["evidence_ids"]) + tok.eos_token)["input_ids"]
            ids, lab = p + t, [-100] * len(p) + t
            if len(ids) > a.max_len:
                self.trunc += 1                      # logged; expected to be 0
                ids, lab = ids[:a.max_len], lab[:a.max_len]
            self.items.append({"input_ids": ids, "labels": lab})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(b):
    m = max(len(x["input_ids"]) for x in b)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in b]),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in b]),
        "labels": torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b]),
    }


train, val = DS("train"), DS("val")
print("truncated: train", train.trunc, "val", val.trunc)

model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()                      # fp32 adapters for stable fp16 AMP
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
    warmup_steps=0.05, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
    gradient_accumulation_steps=a.accum, fp16=True, logging_steps=5,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=3,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    seed=a.seed, report_to="none", remove_unused_columns=False)
tr = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate)

last = None
ck = sorted(Path(a.out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if Path(a.out).exists() else []
if ck:
    last = str(ck[-1]); print("resuming from", last)
tr.train(resume_from_checkpoint=last)
tr.save_model(a.out + "/final_adapter")
json.dump(tr.state.log_history, open(a.out + "/log_history.json", "w"), indent=2)
json.dump(vars(a), open(a.out + "/train_config.json", "w"), indent=2)
print("saved adapter to", a.out + "/final_adapter")

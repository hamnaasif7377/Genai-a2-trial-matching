"""Figures + error tables from metrics/preds files.
usage: python -m src.eval.visualize --log results\logs\log_history.json --runs name=results\prefix ..."""
import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=None)
ap.add_argument("--runs", nargs="+", required=True, help="name=path_prefix (without _metrics.json)")
ap.add_argument("--split", default="val")
ap.add_argument("--out", default="results/figures")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
runs = {r.split("=")[0]: r.split("=")[1] for r in a.runs}
M = {n: json.load(open(p + "_metrics.json")) for n, p in runs.items()}

# 1. training curves
if a.log:
    h = json.load(open(a.log))
    tr = [(x["epoch"], x["loss"]) for x in h if "loss" in x]
    ev = [(x["epoch"], x["eval_loss"]) for x in h if "eval_loss" in x]
    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*tr), label="train loss", alpha=.7)
    plt.plot(*zip(*ev), "o-", label="validation loss", lw=2)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3)
    plt.title("LoRA training curves"); plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=200); plt.close()

# 2. before/after comparison
metrics = [("accuracy", "label", "accuracy"), ("macro-F1", "label", "macro_f1"), ("evidence macro-F1", "evidence", "macro_f1")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, part in zip(axes, ("overall", "inclusion", "exclusion")):
    w = 0.8 / len(M)
    for j, (n, m) in enumerate(M.items()):
        vals = [m[part][k][v] for _, k, v in metrics]
        bars = ax.bar(np.arange(3) + j * w, vals, w, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(3) + w * (len(M) - 1) / 2); ax.set_xticklabels([x[0] for x in metrics], fontsize=8)
    ax.set_title(part); ax.set_ylim(0, 1.05)
axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(out / "comparison.png", dpi=200); plt.close()

# 3. confusion matrices (overall) per run
for n, m in M.items():
    cm = m["overall"]["label"]["confusion_matrix"]
    mat, labs = np.array(cm["matrix"]), cm["labels"]
    keep = [i for i in range(len(labs)) if mat[i].sum() or mat[:, i].sum()]
    mat = mat[np.ix_(keep, keep)]; labs = [labs[i] for i in keep]
    plt.figure(figsize=(6.5, 5.5)); plt.imshow(mat, cmap="Blues")
    for i in range(len(labs)):
        for j in range(len(labs)):
            plt.text(j, i, mat[i, j], ha="center", va="center", color="white" if mat[i, j] > mat.max() / 2 else "black")
    plt.xticks(range(len(labs)), labs, rotation=40, ha="right", fontsize=8); plt.yticks(range(len(labs)), labs, fontsize=8)
    plt.xlabel("predicted"); plt.ylabel("gold"); plt.title(f"Confusion matrix: {n}"); plt.tight_layout()
    plt.savefig(out / f"confusion_{n}.png", dpi=200); plt.close()

# 4. per-class F1
labs = list(next(iter(M.values()))["overall"]["label"]["per_class"].keys())
plt.figure(figsize=(8, 4)); w = 0.8 / len(M)
for j, (n, m) in enumerate(M.items()):
    plt.bar(np.arange(len(labs)) + j * w, [m["overall"]["label"]["per_class"][l]["f1"] for l in labs], w, label=n)
plt.xticks(np.arange(len(labs)) + w * (len(M) - 1) / 2, labs, rotation=25, ha="right", fontsize=8)
plt.ylabel("F1"); plt.legend(fontsize=8); plt.title("Per-class F1 (overall)"); plt.tight_layout()
plt.savefig(out / "per_class_f1.png", dpi=200); plt.close()

# 5. error tables
data = {}
for l in open(f"data/processed/{a.split}.jsonl", encoding="utf-8"):
    r = json.loads(l); data[r["annotation_id"]] = r
for n, p in runs.items():
    rows = [json.loads(l) for l in open(p + "_preds.jsonl", encoding="utf-8")]
    errs = [r for r in rows if r["pred_label"] != r["gold_label"]]
    with open(out / f"errors_{n}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["annotation_id", "type", "criterion", "gold", "pred", "gold_ids", "pred_ids", "raw"])
        for r in errs:
            w.writerow([r["annotation_id"], r["criterion_type"], data[r["annotation_id"]]["criterion_text"][:300],
                        r["gold_label"], r["pred_label"], r["gold_ids"], r["pred_ids"], r["raw"]])
    print(n, "errors:", len(errs), "/", len(rows))
print("figures in", out)

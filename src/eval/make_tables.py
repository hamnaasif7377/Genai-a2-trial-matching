"""Build LaTeX tables for the report from the metrics JSON files (no numbers typed by hand).
Run from the repo root:   python -m src.eval.make_tables
Writes report/tab_*.tex"""
import json
from pathlib import Path

R, OUT = Path("results"), Path("report")
OUT.mkdir(exist_ok=True)
ABB = {"included": "Inc", "not included": "NotInc", "excluded": "Exc", "not excluded": "NotExc",
       "not enough information": "NEI", "not applicable": "NA", "INVALID": "Inv"}
MODELS = [("zeroshot", "Zero-shot, naive prompt"), ("zeroshot_hint", "Zero-shot, labels listed"),
          ("lora_r16_v1", "Fine-tuned (LoRA)")]
f3 = lambda v: f"{v:.3f}"


def load(name, split):
    p = R / f"{name}_{split}_metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def put(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", OUT / name)


def main_table(split, label, caption):
    rows = []
    for key, nm in MODELS:
        m = load(key, split)
        if not m:
            continue
        c = [m[p]["label"][k] for p in ("overall", "inclusion", "exclusion") for k in ("accuracy", "macro_f1")]
        rows.append(f"{nm} & " + " & ".join(f3(x) for x in c) + f" & {f3(m['overall']['evidence']['macro_f1'])} \\\\")
    put(f"tab_main_{split}.tex",
        "\\begin{table*}[t]\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n"
        "\\begin{tabular}{lccccccc}\\toprule\n"
        " & \\multicolumn{2}{c}{Overall} & \\multicolumn{2}{c}{Inclusion} & \\multicolumn{2}{c}{Exclusion} & Evidence\\\\\n"
        "Model & Acc. & Macro-F1 & Acc. & Macro-F1 & Acc. & Macro-F1 & macro-F1\\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")


def perclass_table(split):
    m = load("lora_r16_v1", split)
    if not m:
        return
    rows = []
    for part in ("inclusion", "exclusion"):
        L = m[part]["label"]
        for lab, d in L["per_class"].items():
            rows.append(f"{part} & {lab} & {f3(d['precision'])} & {f3(d['recall'])} & {f3(d['f1'])} & {d['support']} \\\\")
        rows.append(f"\\multicolumn{{2}}{{l}}{{\\emph{{{part}: accuracy / macro-F1}}}} & \\multicolumn{{4}}{{r}}"
                    f"{{{f3(L['accuracy'])} / {f3(L['macro_f1'])}}} \\\\\\midrule")
    put("tab_perclass.tex",
        "\\begin{table}[t]\\centering\\footnotesize\n\\caption{Per-class precision, recall and F1 of the fine-tuned model "
        "on the held-out test split, separately for inclusion and exclusion criteria.}\\label{tab:perclass}\n"
        "\\begin{tabular}{llrrrr}\\toprule\nType & Label & P & R & F1 & Support\\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def evidence_table(split):
    rows = []
    for key, nm in MODELS:
        m = load(key, split)
        if not m:
            continue
        for part in ("overall", "inclusion", "exclusion"):
            e = m[part]["evidence"]
            rows.append(f"{nm} & {part} & " + " & ".join(f3(e[k]) for k in (
                "macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1",
                "nonempty_gold_macro_f1")) + f" & {e['nonempty_gold_n']} \\\\")
        rows[-1] += "\\midrule"
    put("tab_evidence.tex",
        "\\begin{table*}[t]\\centering\\small\n\\caption{Evidence-sentence retrieval on the test split. Records with no gold and "
        "no predicted evidence count as P=R=F1=1; exactly one empty set counts as 0. The last two columns use only "
        "records with non-empty gold evidence.}\\label{tab:evidence}\n"
        "\\begin{tabular}{llccccccccc}\\toprule\n & & \\multicolumn{3}{c}{Macro} & \\multicolumn{3}{c}{Micro} & "
        "Non-empty & \\\\\nModel & Subset & P & R & F1 & P & R & F1 & macro-F1 & $n$\\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")


def conf_tex(m, part):
    cm = m[part]["label"]["confusion_matrix"]
    labs, mat = cm["labels"], cm["matrix"]
    keep = [i for i in range(len(labs)) if sum(mat[i]) or sum(r[i] for r in mat)]
    head = " & ".join(ABB.get(labs[j], labs[j]) for j in keep)
    body = "\\\\\n".join(ABB.get(labs[i], labs[i]) + " & " + " & ".join(str(mat[i][j]) for j in keep) for i in keep)
    return ("\\begin{tabular}{l" + "r" * len(keep) + "}\\toprule\n gold$\\backslash$pred & " + head +
            "\\\\\\midrule\n" + body + "\\\\\\bottomrule\\end{tabular}")


def conf_table(split):
    m = load("lora_r16_v1", split)
    if not m:
        return
    put("tab_conf.tex",
        "\\begin{table}[t]\\centering\\footnotesize\n\\caption{Test confusion matrices of the fine-tuned model "
        "(rows: gold, columns: predicted). Inc = included, NotInc = not included, Exc = excluded, NotExc = not excluded, "
        "NEI = not enough information, NA = not applicable. Top: inclusion criteria. Bottom: exclusion criteria.}"
        "\\label{tab:conf}\n" + conf_tex(m, "inclusion") + "\\\\[4pt]\n" + conf_tex(m, "exclusion") + "\n\\end{table}\n")


main_table("val", "tab:val", "Validation split (8 patients, 206 records), identical examples and greedy decoding for all rows.")
main_table("test", "tab:test", "Held-out test split (8 patients, 206 records), greedy decoding, evaluated once.")
perclass_table("test")
evidence_table("test")
conf_table("test")

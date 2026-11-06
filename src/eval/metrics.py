"""Shared metrics for the common evaluation protocol."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}


def label_metrics(gold, pred, labels):
    """Macro-F1 is computed over classes present in gold OR predictions (documented choice).
    Invalid/unparsable predictions are mapped to 'INVALID' and counted as errors."""
    pred = [p if p in labels else "INVALID" for p in pred]
    present = [l for l in labels if l in set(gold) | set(pred)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels + ["INVALID"])
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=present, average="macro", zero_division=0)),
        "classes_in_macro": present,
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": {"labels": labels + ["INVALID"], "matrix": cm.tolist()},
    }


def _prf(g, p):
    g, p = set(g), set(p)
    if not g and not p:
        return 1.0, 1.0, 1.0          # both empty: correct abstention
    if not g or not p:
        return 0.0, 0.0, 0.0          # one empty: total miss
    tp = len(g & p)
    pr, rc = tp / len(p), tp / len(g)
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def evidence_metrics(gold_list, pred_list):
    per = [_prf(g, p) for g, p in zip(gold_list, pred_list)]
    macro = np.mean(per, axis=0)
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold_list, pred_list))
    npred = sum(len(set(p)) for p in pred_list)
    ngold = sum(len(set(g)) for g in gold_list)
    mp = tp / npred if npred else 0.0
    mr = tp / ngold if ngold else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    ne = [(g, p) for g, p in zip(gold_list, pred_list) if g]
    ne_macro = np.mean([_prf(g, p) for g, p in ne], axis=0) if ne else [0, 0, 0]
    return {
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "nonempty_gold_n": len(ne),
        "nonempty_gold_macro_precision": float(ne_macro[0]),
        "nonempty_gold_macro_recall": float(ne_macro[1]),
        "nonempty_gold_macro_f1": float(ne_macro[2]),
    }


def full_report(records, preds):
    """records: processed jsonl rows; preds: list of (label, ids, valid_json)."""
    out = {"json_valid_rate": float(np.mean([p[2] for p in preds]))}
    for name, sel in (("overall", None), ("inclusion", "inclusion"), ("exclusion", "exclusion")):
        idx = [i for i, r in enumerate(records) if sel is None or r["criterion_type"] == sel]
        gold = [records[i]["label"] for i in idx]
        pl = [preds[i][0] or "INVALID" for i in idx]
        if sel is None:
            labels = sorted(set(LABELS["inclusion"]) | set(LABELS["exclusion"]))
        else:
            labels = LABELS[sel]
        out[name] = {
            "label": label_metrics(gold, pl, labels),
            "evidence": evidence_metrics([records[i]["evidence_ids"] for i in idx], [preds[i][1] for i in idx]),
        }
    return out

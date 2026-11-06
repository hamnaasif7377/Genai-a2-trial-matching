"""Patient-grouped split (seed 42, 37/8/8), manifest, leakage checks, processed JSONL.
GPT-4 columns are dropped here and never reach the model."""
import json, ast, re
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed"
MANIFEST = ROOT / "manifests" / "split_manifest.json"
SEED, N_TRAIN, N_VAL, N_TEST = 42, 37, 8, 8


def parse_note(note):
    """'0. text\n1. text' -> {0: 'text', 1: 'text'}; continuation lines are appended."""
    sents, last = {}, None
    for line in note.split("\n"):
        m = re.match(r"^\s*(\d+)\.\s?(.*)$", line)
        if m:
            last = int(m.group(1))
            sents[last] = m.group(2).strip()
        elif last is not None and line.strip():
            sents[last] += " " + line.strip()
    return sents


def parse_ids(s):
    if s is None or str(s).strip() in ("", "[]", "None"):
        return []
    try:
        v = json.loads(s)
    except Exception:
        v = ast.literal_eval(s)
    return sorted(int(x) for x in v)


def main():
    ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations")["train"]
    rows = [dict(r) for r in ds]
    print("rows:", len(rows))

    patients = np.array(sorted({r["patient_id"] for r in rows}))
    print("unique patients:", len(patients))
    assert len(patients) == N_TRAIN + N_VAL + N_TEST, "unexpected patient count"

    rng = np.random.default_rng(SEED)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    split_pat = {
        "train": list(shuffled[:N_TRAIN]),
        "val": list(shuffled[N_TRAIN:N_TRAIN + N_VAL]),
        "test": list(shuffled[N_TRAIN + N_VAL:]),
    }
    pat2split = {p: s for s, ps in split_pat.items() for p in ps}

    records = {"train": [], "val": [], "test": []}
    anomalies = 0
    for r in rows:
        sents = parse_note(r["note"])
        ids = sorted(sents)
        if ids != list(range(len(ids))):
            anomalies += 1
        gold = parse_ids(r["expert_sentences"])
        if any(g not in sents for g in gold):
            anomalies += 1
        rec = {
            "annotation_id": int(r["annotation_id"]),
            "patient_id": r["patient_id"],
            "trial_id": r["trial_id"],
            "criterion_type": r["criterion_type"],
            "criterion_text": r["criterion_text"],
            "note": r["note"],
            "n_sentences": len(sents),
            "label": r["expert_eligibility"],
            "evidence_ids": gold,
        }
        records[pat2split[r["patient_id"]]].append(rec)
    print("note-parsing anomalies:", anomalies)

    # ---- leakage checks ----
    sets = {s: {x["patient_id"] for x in v} for s, v in records.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"]), "PATIENT LEAKAGE"
    assert sum(len(v) for v in records.values()) == len(rows), "records lost/duplicated"
    all_ids = [x["annotation_id"] for v in records.values() for x in v]
    assert len(all_ids) == len(set(all_ids)), "duplicate annotation ids"
    print("leakage checks passed")

    # ---- write outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(exist_ok=True)
    for s, v in records.items():
        with open(OUT_DIR / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for x in v:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "ncbi/TrialGPT-Criterion-Annotations",
        "method": "sorted unique patient ids -> np.random.default_rng(42).shuffle -> 37/8/8",
        "seed": SEED,
        "patients": {s: {"count": len(p), "ids": [str(i) for i in p]} for s, p in split_pat.items()},
        "records": {s: len(v) for s, v in records.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- statistics ----
    for s, v in records.items():
        print(f"\n== {s}: {len(split_pat[s])} patients, {len(v)} records")
        for t in ("inclusion", "exclusion"):
            c = Counter(x["label"] for x in v if x["criterion_type"] == t)
            print(f"  {t}: {dict(c)}")
        empty = sum(1 for x in v if not x["evidence_ids"])
        print(f"  records with empty gold evidence: {empty}")
        print(f"  mean note sentences: {np.mean([x['n_sentences'] for x in v]):.1f}")


if __name__ == "__main__":
    main()

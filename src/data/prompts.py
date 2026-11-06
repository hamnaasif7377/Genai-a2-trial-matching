import json, os, re

SYSTEM = (
    "You are a clinical trial eligibility assistant. Given a patient note with numbered sentences "
    "and ONE trial criterion, output JSON with the eligibility label and the IDs of the patient "
    "sentences that support it. Use only the information in the note."
)

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]

HINT = (
    "\n\nAllowed labels for inclusion criteria: included, not included, not enough information, "
    "not applicable. Allowed labels for exclusion criteria: excluded, not excluded, "
    "not enough information, not applicable. evidence_sentence_ids is the list of supporting "
    "sentence numbers from the note ([] if none)."
)


def build_user_prompt(note, criterion_type, criterion_text):
    s = (
        f"Patient note:\n{note}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        'Answer with JSON: {"label": ..., "evidence_sentence_ids": [...]}'
    )
    if os.environ.get("LABEL_HINT") == "1":   # used only for the fair zero-shot baseline
        s += HINT
    return s


def build_messages(note, criterion_type, criterion_text):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(note, criterion_type, criterion_text)},
    ]


def build_target(label, evidence_ids):
    return json.dumps({"label": label, "evidence_sentence_ids": list(evidence_ids)})


def parse_output(text):
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None, [], False
    try:
        d = json.loads(m.group(0))
        ids = [int(i) for i in d.get("evidence_sentence_ids", [])]
        return str(d.get("label", "")).strip().lower(), ids, True
    except Exception:
        return None, [], False

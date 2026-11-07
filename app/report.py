"""Standalone HTML screening report with provenance."""
import html, json
from pathlib import Path

CSS = ("body{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1c2330}"
       "h1{font-size:22px}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}"
       "td,th{border:1px solid #ccd;padding:6px;vertical-align:top;text-align:left}th{background:#eef1f6}"
       ".warn{background:#fff4d6;border:1px solid #e0b84c;padding:10px;border-radius:6px;margin:12px 0}"
       ".flag{color:#9a6700;margin-top:4px}.prov td:first-child{width:200px;font-weight:bold}"
       ".out{font-size:18px;font-weight:bold;margin:8px 0}")


def e(x):
    return html.escape("" if x is None else str(x))


def build_html(res, jid):
    m, o = res["meta"], res["overall"]
    rows = []
    for i, r in enumerate(res["criteria"], 1):
        ev = "<br>".join("<b>[%s]</b> %s" % (x["id"], e(x["text"])) for x in r["evidence"]) or "<i>none cited</i>"
        fl = "".join("<div class='flag'>Warning: %s</div>" % e(f) for f in r["flags"])
        lab = r["label"] if r["label_valid"] else "invalid output"
        rows.append("<tr><td>%d</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s%s</td></tr>"
                    % (i, e(r["type"]), e(r["criterion"]), e(lab), ev, e(r["explanation"]), fl))
    note = "<br>".join("<b>%d.</b> %s" % (i, e(s)) for i, s in enumerate(m["sentences"]))
    prov = [("Report ID", jid), ("Created (UTC)", m["created_utc"]), ("Base model", m["base_model"]),
            ("Fine-tuned adapter", m["adapter"]), ("Decoding mode", m["decoding_mode"]), ("Decoding", m["decoding"]),
            ("Device / dtype", "%s / %s" % (m["device"], m["dtype"])), ("Patient note SHA-256 (first 16)", m["note_sha256"]),
            ("Note numbering", "automatic sentence split" if m["note_auto_numbered"] else "numbered by the user"),
            ("Total inference time", "%.1f s" % res["total_seconds"]), ("PyTorch", m["torch"])]
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Eligibility screening report %s</title>"
            "<style>%s</style></head><body><h1>Clinical trial eligibility screening report</h1>"
            "<div class='warn'><b>%s</b></div><div class='out'>Preliminary outcome: %s</div><ul>%s</ul>"
            "<p><i>%s</i></p><h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th>"
            "<th>Decision</th><th>Evidence from note</th><th>Explanation and warnings</th></tr>%s</table>"
            "<h2>Patient note (numbered)</h2><p>%s</p><h2>Provenance</h2><table class='prov'>%s</table>"
            "<p>Limitations: trained on 37 patients; rare labels (not included, excluded) are weak; absence of a mention "
            "is not evidence either way.</p></body></html>"
            % (e(jid), CSS, e(res["disclaimer"]), e(o["outcome"]), "".join("<li>%s</li>" % e(x) for x in o["reasons"]),
               e(o["rule"]), "".join(rows), note,
               "".join("<tr><td>%s</td><td>%s</td></tr>" % (e(a), e(b)) for a, b in prov)))


def save(res, jid, directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / ("screening_%s.json" % jid)).write_text(json.dumps(res, indent=2), encoding="utf-8")
    (d / ("screening_%s.html" % jid)).write_text(build_html(res, jid), encoding="utf-8")

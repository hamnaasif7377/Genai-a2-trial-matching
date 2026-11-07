"""FastAPI application: screening, decoding benchmark, stored experiments, reports."""
import json, os, re, threading, time, traceback, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app import engine as E
from app import report as R

APP = Path(__file__).resolve().parent
ROOT = APP.parent
REPORT_DIR = Path(os.environ.get("REPORT_DIR", ROOT / "reports"))
eng = E.Engine()
JOBS = {}


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=eng.load, daemon=True).start()
    yield


app = FastAPI(title="Trial eligibility screening (educational prototype)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP / "static")), name="static")


class ScreenReq(BaseModel):
    note: str = ""
    criteria_text: str = ""
    mode: str = "cached"
    k: int = 4


class BenchReq(BaseModel):
    note: str = ""
    criterion_type: str = "exclusion"
    criterion_text: str = ""
    max_new: int = 16
    k: int = 4
    include_nocache: bool = True
    include_speculative: bool = True


class AttnReq(BaseModel):
    L: int = 512
    H: int = 4
    D: int = 64
    block: int = 64
    causal: bool = True


def ensure_ready():
    if eng.state == "error":
        raise HTTPException(503, "The model failed to load: " + str(eng.error))
    if eng.state != "ready":
        raise HTTPException(503, "The model is still loading. Please wait a moment and try again.")


def new_job(kind, meta=None):
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "status": "queued", "progress": 0, "total": 1, "message": "Queued",
           "partial": [], "result": None, "error": None, "meta": meta or {}}
    JOBS[jid] = job
    while len(JOBS) > 40:
        JOBS.pop(next(iter(JOBS)))
    return job


def start(job, fn):
    def run():
        with eng.lock:
            job["status"], job["message"] = "running", "Running"
            try:
                fn(job)
                job["status"], job["message"] = "done", "Done"
            except Exception as ex:
                traceback.print_exc()
                job["status"], job["error"] = "error", f"{type(ex).__name__}: {ex}"
    threading.Thread(target=run, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(APP / "static" / "index.html")


@app.get("/api/status")
def status():
    return eng.status()


@app.get("/api/samples")
def samples():
    out = json.loads((APP / "samples.json").read_text(encoding="utf-8-sig"))
    val = ROOT / "data" / "processed" / "val.jsonl"
    if val.exists():
        groups = {}
        for line in val.read_text(encoding="utf-8-sig").splitlines():
            r = json.loads(line)
            groups.setdefault((r["patient_id"], r["trial_id"]), []).append(r)
        for (pid, tid), recs in list(groups.items())[:3]:
            inc = [r["criterion_text"] for r in recs if r["criterion_type"] == "inclusion"][:4]
            exc = [r["criterion_text"] for r in recs if r["criterion_type"] == "exclusion"][:4]
            crit = "Inclusion criteria:\n" + "\n".join("- " + c.replace("\n", " ") for c in inc) + \
                   "\nExclusion criteria:\n" + "\n".join("- " + c.replace("\n", " ") for c in exc)
            out.append({"name": f"Validation split: patient {pid}, trial {tid}", "note": recs[0]["note"], "criteria": crit})
    return out


@app.post("/api/screen")
def screen(req: ScreenReq):
    ensure_ready()
    if req.mode not in ("cached", "nocache", "speculative"):
        raise HTTPException(400, "Unknown decoding mode.")
    if not 1 <= req.k <= 8:
        raise HTTPException(400, "Speculative k must be between 1 and 8.")
    try:
        sents, numbered, auto = E.prepare_note(req.note)
        crit = E.parse_criteria(req.criteria_text)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    job = new_job("screen", eng.make_meta(sents, numbered, auto, req.mode, req.k))
    job["total"] = len(crit)

    def work(j):
        t0 = time.perf_counter()
        if req.mode == "speculative":
            j["message"] = "Loading draft model"
            eng.ensure_draft()
        for i, (t, c) in enumerate(crit):
            j["message"] = f"Criterion {i + 1} of {len(crit)}"
            j["partial"].append(eng.screen_one(sents, numbered, t, c, req.mode, req.k))
            j["progress"] = i + 1
        j["result"] = E.assemble(j["meta"], j["partial"], time.perf_counter() - t0)
        try:
            R.save(j["result"], j["id"], REPORT_DIR)
        except Exception:
            traceback.print_exc()
    start(job, work)
    return {"job_id": job["id"]}


@app.post("/api/benchmark")
def benchmark(req: BenchReq):
    ensure_ready()
    if req.criterion_type not in ("inclusion", "exclusion"):
        raise HTTPException(400, "Criterion type must be inclusion or exclusion.")
    if not req.criterion_text.strip():
        raise HTTPException(400, "Enter a criterion to benchmark.")
    if len(req.criterion_text) > E.LIMITS["criterion_chars"]:
        raise HTTPException(400, "The criterion is too long.")
    if not 4 <= req.max_new <= 64 or not 1 <= req.k <= 8:
        raise HTTPException(400, "max new tokens must be 4-64 and k 1-8.")
    try:
        E.prepare_note(req.note)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    job = new_job("benchmark")
    start(job, lambda j: eng.benchmark(j, req.note, req.criterion_type, req.criterion_text.strip(), req.max_new,
                                       req.k, req.include_nocache, req.include_speculative))
    return {"job_id": job["id"]}


@app.post("/api/attention")
def attention(req: AttnReq):
    if not (16 <= req.L <= 2048 and 8 <= req.block <= 256 and 1 <= req.H <= 8 and req.D in (16, 32, 64, 128)):
        raise HTTPException(400, "Use L 16-2048, block 8-256, heads 1-8, head dim 16/32/64/128.")
    job = new_job("attention")
    start(job, lambda j: E.attention_demo(j, req.L, req.H, req.D, req.block, req.causal))
    return {"job_id": job["id"]}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job (the server may have restarted).")
    return JOBS[jid]


STORED = ["zeroshot_test_metrics", "zeroshot_hint_test_metrics", "lora_r16_v1_test_metrics", "kv_bench_gpu_fp16",
          "kv_fp32_correctness", "spec_ftdraft_fp16", "spec_basedraft_fp16", "spec_ftdraft_fp32",
          "attention_bench_gpu", "attention_bench_cpu", "attention_tests"]


@app.get("/api/stored")
def stored():
    out = {}
    for n in STORED:
        p = ROOT / "results" / (n + ".json")
        if p.exists():
            out[n] = json.loads(p.read_text(encoding="utf-8-sig"))
    a = out.get("attention_tests")
    if a:
        f64 = [c["max_abs_err"] for c in a["cases"] if c["dtype"].endswith("float64")]
        out["attention_tests"] = {"all_pass": a["all_pass"], "causality_ok": a["causality_ok"], "n_cases": len(a["cases"]),
                                  "max_err_float64": max(f64) if f64 else None}
    return out


@app.get("/api/report/{jid}")
def get_report(jid: str, format: str = "html", download: int = 0):
    if not re.fullmatch(r"[0-9a-f]{10}", jid) or format not in ("html", "json"):
        raise HTTPException(404, "Report not found.")
    p = REPORT_DIR / f"screening_{jid}.{format}"
    if not p.exists():
        raise HTTPException(404, "Report not found. It is written when a screening run finishes.")
    disp = ("attachment" if download else "inline") + f'; filename="{p.name}"'
    return Response(p.read_text(encoding="utf-8"), media_type="text/html" if format == "html" else "application/json",
                    headers={"Content-Disposition": disp})

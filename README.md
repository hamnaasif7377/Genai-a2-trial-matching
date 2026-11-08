# Clinical trial eligibility screening: LoRA fine-tuning, KV cache, tiled attention, speculative decoding

Generative AI, Assignment 2. **Educational screening prototype. Not a clinical enrollment decision.**

A Qwen2.5-1.5B-Instruct model is fine-tuned with LoRA on the TrialGPT Criterion Annotations. For one patient note and one
inclusion/exclusion criterion it returns an eligibility label and the IDs of the supporting patient sentences. The repository
also contains hand-written KV-cached decoding, tiled online-softmax attention, greedy speculative decoding, and a
browser application. `model.generate()` is never used.

## Links
- Report (PDF): `report/main.pdf` (LaTeX source: `report/main.tex`)
- Adapter weights (download): https://drive.google.com/drive/folders/1Q_y-8_If-Pbt11oLkFB0fIlx_YyGqPF6?usp=sharing (folders `lora_r16_v1/final_adapter` and `lora_draft_0p5b_v1/final_adapter`)
- Demo video: **https://youtu.be/cOS3ocKGSzQ**
- Split manifest: `manifests/split_manifest.json`

## Headline results (held-out test split: 8 patients, 206 criterion records, greedy decoding)
| Model | Accuracy | Macro-F1 | Evidence macro-F1 |
|---|---|---|---|
| Qwen2.5-1.5B zero-shot, naive prompt | 0.000 | 0.000 | 0.149 |
| Qwen2.5-1.5B zero-shot, labels listed in prompt | 0.170 | 0.141 | 0.587 |
| **Qwen2.5-1.5B + LoRA (ours)** | **0.733** | **0.540** | **0.835** |

KV cache: token-identical to the uncached reference in float32 (16/16), 24/26 in fp16 (two near-ties); up to 5.6x faster at 1024 input tokens (T4).
Tiled attention: max error about 2e-16 in float64; peak memory 384 MB -> 6 MB at length 2048, but slower than the reference and SDPA.
Speculative decoding: output identical to greedy (30/30 and 15/15), acceptance 0.96, but 0.68-0.72x speed of cached greedy on a T4.
Main weakness: missing information is often predicted as "not excluded"; rare labels are rarely right. See the report.

## Repository layout
```
src/data/        prepare_data.py (split, manifest, leakage checks), prompts.py (prompt, output schema, parser)
src/training/    train_lora.py (resumable LoRA training)
src/eval/        evaluate.py, metrics.py, token_stats.py, visualize.py, error_report.py, make_tables.py
src/inference/   kv_decode.py (uncached + KV-cached decoders), common.py, bench_kv.py
src/attention/   attention.py (reference + tiled online softmax), test_attention.py, bench_attention.py
src/speculative/ spec_decode.py (greedy speculative decoder), bench_spec.py
app/             FastAPI backend (main.py, engine.py, report.py) and browser UI (static/index.html)
data/processed/  train/val/test JSONL (GPT-4 columns removed)
manifests/       split_manifest.json
results/         metrics, predictions, benchmark JSON, logs, figures
report/          IEEE LaTeX paper
```

## Setup (Windows PowerShell; Linux/macOS: same commands with the usual venv activation)
```powershell
git clone https://github.com/hamnaasif7377/Genai-a2-trial-matching.git
cd Genai-a2-trial-matching
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
`requirements-lock.txt` lists the exact versions used.

## Download the adapters
Put them **next to** the repository folder, like this:
```
<parent>\repo\                                   (this repository)
<parent>\adapter\lora_r16_v1\final_adapter\adapter_model.safetensors
<parent>\adapter\lora_draft_0p5b_v1\final_adapter\adapter_model.safetensors
```
Download the Drive folders from the link above, extract, and make sure the `.safetensors` and `adapter_config.json` files sit
directly inside each `final_adapter` folder. Other locations: set `ADAPTER_PATH` and `DRAFT_ADAPTER_PATH`.

## Run the application (no notebook or IDE needed)
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000. The status chip is amber while the model loads (the first start downloads about 3 GB from the
Hugging Face Hub, and about 1 GB more the first time speculative mode is used) and turns green when ready. Keep the window open.
Needs about 8 GB of free RAM (CPU, fp32). Roughly 11 s per criterion on an 8-thread laptop CPU.

Docker alternative (needs about 10 GB available to Docker/WSL):
```powershell
docker compose up --build
```
then open http://localhost:8000. The compose file mounts `../adapter` read-only and writes reports to `./reports`.

Tabs: **Screening** (note + criteria -> per-criterion label, cited sentences, explanation, warnings, overall outcome, HTML/JSON report),
**Decoding benchmark** (live uncached vs KV-cached vs speculative on one criterion), **Experiments** (recorded results and a live
attention check), **About**.

## Reproduce the experiments
1. Data and split (local, seconds): `python src/data/prepare_data.py` (writes `data/processed/*.jsonl`, `manifests/split_manifest.json`; asserts no patient leakage).
2. Token statistics / truncation: `python -m src.eval.token_stats`.
3. GPU steps ran on a free Colab T4 (clone the repo, `pip install -U transformers peft accelerate datasets scikit-learn "torchao>=0.16.0"`, restart):
```bash
python -m src.training.train_lora --out $G/adapters/lora_r16_v1                       # target LoRA (~10 min)
python -m src.training.train_lora --model Qwen/Qwen2.5-0.5B-Instruct --out $G/adapters/lora_draft_0p5b_v1
python -m src.eval.evaluate --split val  --out results/zeroshot_val                     # naive prompt
LABEL_HINT=1 python -m src.eval.evaluate --split val --out results/zeroshot_hint_val    # labels listed
python -m src.eval.evaluate --split val  --adapter $A --out results/lora_r16_v1_val
# test split: run ONCE, after all decisions
python -m src.eval.evaluate --split test --adapter $A --out results/lora_r16_v1_test
python -m src.inference.bench_kv --adapter $A --out results/kv_bench_gpu_fp16           # also --dtype float32 for exact equality
python -m src.speculative.bench_spec --target_adapter $A --draft_adapter $DA --out results/spec_ftdraft_fp16
python -m src.attention.bench_attention
```
4. Attention tests (CPU): `python -m src.attention.test_attention` (112 cases + causality check; writes `results/attention_tests.json`).
5. Figures and tables: `python -m src.eval.visualize ...`, `python -m src.eval.error_report`, `python -m src.eval.make_tables`.

Benchmark logs and raw predictions are in `results/`. Training curves data: `results/logs/log_history.json`; hyperparameters: `results/logs/train_config.json`.

## Evaluation protocol
Patient-grouped split: sorted unique patient IDs, shuffled with `numpy.random.default_rng(42)`, first 37 train, next 8 validation, last 8 test
(603 / 206 / 206 records). GPT-4 columns of the dataset are dropped and never used as input or in metrics. The test split was evaluated once.
Macro-F1 averages over classes present in the gold labels or predictions. Evidence metrics: both-empty = 1, exactly-one-empty = 0.

## Limitations
Small data (53 patients, 8 test patients, one run, no confidence intervals); rare labels and numeric/negated criteria are weak; absence of a mention is
often read as "not excluded" (the app flags decisions without cited evidence); explanations are templates built from cited sentences; the overall outcome is a
hand-written aggregation rule, not a labeled ground truth; tiled attention and speculative decoding are educational and slower than the baselines here.

## AI-use statement
I used an AI assistant (Claude) to generate the first versions of the scripts and a draft of the report, and to help fix library-version errors on Colab. I ran all training, evaluations, benchmarks and tests myself, checked the logs and outputs, and made sure the numbers in the report come from the saved result files. See the appendix of the report.




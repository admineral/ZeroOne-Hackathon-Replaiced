# Report — Industrial AI (Infineon Track)

**Zero One Hack_01 · Learning & Benchmarking Process Logic**

> Can a model learn *real* semiconductor process logic — or does it just memorize
> patterns? We train on **3 known product families** and deliberately prepare the
> model for the **secret 4th family** the jury evaluates on. The entire HPC pipeline
> on the Leonardo cluster is driven from a **click-through dashboard with an AI
> experiment coach** — without a single terminal command.

---

## 1. TL;DR

- **Model:** GPT-style causal Transformer (`TinyCausalTransformer`) over process-step
  tokens. Scalable from ~0.5 M to ~180 M parameters via four one-click presets.
- **Our core trick:** **Random family dropout** (30 %). The family token is randomly
  replaced with `<FAMILY_UNKNOWN>` during training → the model learns to produce
  plausible sequences even without a known family → **OOD generalization to the
  jury's unknown 4th family**.
- **Evaluation:** Rule-aware evaluation against **10 formal process-logic rules**,
  plus an N-gram baseline. Two eval passes (real family / `UNKNOWN`) directly
  demonstrate OOD robustness.
- **Engineering:** A local **FastAPI + React + AI-gateway** dashboard that automates
  the Leonardo Slurm pipeline over SSH: generate → upload → train → monitor live →
  evaluate → submit.
- **Submission:** All three tasks produced (Next-Step, Completion, Anomaly). Anomaly
  detection is rule-based: **387 / 987** sequences flagged invalid.

---

## 2. Problem & Our Approach

Industrial processes are long, strictly ordered step sequences whose meaning depends
on order and intermediate steps. The track provides three semiconductor product
families — **MOSFET, IGBT, IC** — and evaluates on a **fourth, undisclosed family**
(out-of-distribution).

**Our key idea:** A model that only memorizes patterns of the three training
families will fail on the 4th. We therefore force *family-agnostic* learning by
controllably removing the family information during training (see §4.3). This makes
the model fall back on the *process logic* itself instead of the family hint.

---

## 3. Repository Structure

```
zero1hack/
├── industrial-infineon/
│   └── training_data/              # ML pipeline (data, model, eval, submission)
│       ├── generate_sequences.py   # Grammar generator + validator (10 rules)
│       ├── sequence_data.py        # Tokenization, vocab, packing, family tokens
│       ├── train_transformer.py    # Training (DDP, bf16, checkpointing)
│       ├── evaluate_rules.py       # Rule-aware evaluation
│       ├── baseline_ngram.py       # N-gram baseline (next-step)
│       ├── make_submission.py      # Produces the 3 organizer CSVs
│       ├── eval_metrics.py         # Metric definitions + quality thresholds
│       └── *.slurm                 # Slurm jobs for Leonardo
├── dashboard/
│   ├── backend/                    # FastAPI: SSH orchestration of the pipeline
│   ├── frontend/                   # React/Vite: click UI with live charts
│   ├── ai-gateway/                 # Node: AI experiment coach
│   └── dev.sh                      # Starts all 3 local services
└── participant_files/submission/   # Generated prediction CSVs
```

---

## 4. ML Pipeline

### 4.1 Data

- **Three product families** in *long-format* CSVs (`SEQUENCE_ID, STEP`, one row per
  step). Every sequence starts with `RECEIVE WAFER LOT` and ends with `SHIP LOT`.
- **Data generation** via `generate_sequences.py`, which implements the formal
  process grammar from `generation_rules.md` and only produces rule-valid sequences.
- **Scaling experiment:** We massively expanded the dataset
  (`dataset_manifest.json`):

| Family | Generated sequences | Step rows |
|---|---:|---:|
| MOSFET | 100,000 | 12,525,095 |
| IGBT | 100,000 | 14,804,325 |
| IC | 100,000 | 11,514,196 |
| **Total** | **300,000** | **38,843,616** |

> Seed 42, deterministically reproducible.

- **Memmap packing:** For large datasets the corpus is packed into a memmap blob
  (`packed/`). Under DDP all ranks share the same data via the OS page cache
  (RAM ≈ 0 instead of full materialization).

### 4.2 Tokenization & Family Conditioning

- **One process step = one token** (e.g. `"DEPOSIT GATE OXIDE"`). Vocabulary
  ≈ 120 step strings plus special tokens.
- Each sequence is encoded as `[<BOS>, <FAMILY_x>, …steps…, <EOS>]`.
- **Four family tokens:** `<FAMILY_MOSFET>`, `<FAMILY_IGBT>`, `<FAMILY_IC>`, and a
  neutral `<FAMILY_UNKNOWN>` for unknown/missing families.

### 4.3 Random Family Dropout (our OOD mechanism)

During training the real family token is replaced with `<FAMILY_UNKNOWN>` with
probability **`family_dropout = 0.30`** — **re-sampled per example and per epoch**:

```
<BOS> <FAMILY_IC> …steps…   --30%-->   <BOS> <FAMILY_UNKNOWN> …steps…
```

Effect: the model learns to continue plausibly both *with* a known family and
*without* a family prefix. That is exactly the jury's evaluation setting on an
unknown 4th product family.

### 4.4 Model

`TinyCausalTransformer` — GPT-style, causal mask, learned positional embeddings,
`GELU`, final `LayerNorm`. Default configuration and one-click presets:

| Preset | d_model | Layers | Heads | Batch | ≈ Params |
|---|---:|---:|---:|---:|---:|
| Tiny (default) | 128 | 2 | 4 | 32 | ~0.5 M |
| Recommended | 256 | 4 | 8 | 256 | ~3.5 M |
| Scale up | 512 | 6 | 8 | 128 | ~22 M |
| Even bigger | 1024 | 12 | 16 | 256 | ~180 M |

These presets enable the track's required **scaling experiment** (small vs. large,
data vs. model size) with one click.

### 4.5 Training

File: `train_transformer.py`. Default hyperparameters:

- Objective: next-token prediction, `CrossEntropyLoss` (padding ignored, optional
  label smoothing).
- Optimizer: **AdamW**, `lr = 3e-4`, `weight_decay = 0.01`, grad clipping 1.0.
- LR schedule: constant or **linear warmup + cosine decay** (`--lr-schedule cosine`).
- Precision: **bf16 autocast + TF32** on A100 (no GradScaler needed).
- Split: 80 % train / 10 % val / 10 % test (deterministic, seed 42).
- **Multi-GPU:** `DistributedDataParallel` via `torchrun` (one process per GPU),
  metrics all-reduced across ranks.
- **Robust checkpointing:** the best-by-val-loss checkpoint is written atomically
  *during* training; a `SIGTERM`/walltime kill flushes the current best checkpoint
  (marked `interrupted`) so no run is lost.
- **Self-describing checkpoints:** hyperparameters, split ratios, seed,
  `max_sequences`, and the Slurm job id are stored, so evaluation reconstructs the
  identical held-out split (no leakage).
- **Telemetry:** `train_log.csv` (loss/acc/LR/GPU memory per epoch),
  `train_stats.json` (parameter count, peak GPU memory, training time), and a
  `gpu_timeline.csv` (nvidia-smi every 2 s) for the dashboard charts.

---

## 5. Evaluation

### 5.1 Rule-aware Eval (`evaluate_rules.py`)

We evaluate generated sequence completions against **10 formal process-logic rules**
(from `generate_sequences.py` / `generation_rules.md`):

| Rule | Violation |
|---|---|
| `RULE_DEP_NO_CLEAN` | Deposition without a prior clean |
| `RULE_METAL_ETCH_NO_LITHO` | Metal etch without preceding lithography |
| `RULE_ETCH_NO_MASK` | Etch without a mask |
| `RULE_LITHO_LEVEL_SKIP` | Skipped lithography level |
| `RULE_IMPLANT_NO_MASK` | Implant without a mask |
| `RULE_CMP_NO_DEP` | CMP without prior deposition |
| `RULE_PAD_OPEN_BEFORE_DEP` | Pad-open before its deposition |
| `RULE_TEST_BEFORE_PASSIVATION` | Electrical test before passivation |
| `RULE_SHIP_BEFORE_TEST` | Ship before test |
| `RULE_BACKSIDE_BEFORE_PASSIVATION` | Backside step before passivation |

**Measured metrics** (rolled up per source/completion fraction):

- `valid_rate` — share of rule-valid completions.
- `quality_rate` — stricter: valid **and** plausible length (length ratio
  0.8–1.25), at most 2 consecutive repeats, suffix accuracy ≥ 0.5.
- `mean_suffix_acc`, `mean_jaccard`, `mean_len_ratio`, `eos_rate`.

**Three eval rows per run** make the core claim directly visible:

1. `heldout_source` — real held-out recipes (sanity anchor, ≈ 1.0 everywhere).
2. `model_generated` — model **with** the real family.
3. `model_generated_unknown` — model **with `<FAMILY_UNKNOWN>`** → the OOD test: do
   completions stay valid *without* knowing the family?

Completion is greedy at **60 % and 80 %** sequence cut; the held-out split is
reconstructed exactly from the checkpoint metadata.

### 5.2 Baseline (`baseline_ngram.py`)

N-gram model with backoff (orders 1/2/3/5) for next-step prediction, evaluated with
**Top-1/3/5 accuracy and MRR**. Provides the reference point for "does the
Transformer learn more than pure frequency statistics?".

---

## 6. Submission

`make_submission.py` produces the three organizer-ready CSVs from the official eval
inputs into `participant_files/submission/`:

| Task | File | Content | Method |
|---|---|---|---|
| 1 — Next-Step | `predictions_nextstep.csv` | 600 rows, Top-5 ranking | Model |
| 2 — Completion | `predictions_completion.csv` | 600 rows, remaining sequence (60 %/80 %) | Model (greedy) |
| 3 — Anomaly | `predictions_anomaly.csv` | 987 rows, valid/invalid + rule | Rule-based |

- **Task 3** needs no model: the validator flags each sequence and outputs the first
  violated rule as `PREDICTED_RULE`. **Result: 387 / 987 sequences flagged invalid.**
- **Tasks 1 & 2** use the trained checkpoint + vocabulary. Unknown families
  automatically fall back to `<FAMILY_UNKNOWN>` (consistent with training).

---

## 7. Dashboard — UI & Automation

Instead of manual `ssh`, `sbatch`, `scp`, and log tailing: a local web dashboard
that automates the entire pipeline.

### 7.1 Architecture

```
Browser (React) --REST/SSE--> FastAPI (127.0.0.1) --paramiko/SSH--> Leonardo login --sbatch--> A100 GPU job
```

The browser never SSHes itself. `dev.sh` starts three local services:
**FastAPI backend (:8000)**, **AI gateway (:8787)**, **Vite frontend (:5173)**.

### 7.2 UI (React / Vite)

- **Pipeline rail** — click flow Setup → Dataset → Monitor → Evaluate → Submission.
- **Live loss chart** — training loss streams per epoch via SSE.
- **Log drawer** — job `.out`/`.err` tailed live.
- **Results panel** — valid-rate cards + rule-violation table from the eval.
- **Submission card** — tasks + checkpoint selection, one click.
- **Dataset management** + **resource panel** (GPU utilization/power).

### 7.3 Automation (FastAPI + paramiko)

1. Connect over SSH (password only in gitignored `.env`, never in the browser).
2. Generate data on Leonardo — versioned in `datasets/<id>/`.
3. Upload scripts via `sftp`.
4. Check the environment (torch + CUDA).
5. Training: submit `sbatch`, capture the job id.
6. Auto-poll the queue (`squeue`, every 5 s) + resources (`sacct`).
7. Live streams for loss, logs, and GPU timeline (SSE).
8. Evaluate + submission job; prediction CSVs are pulled back automatically.

Run keys: `transformer`, `ngram`, `eval_transformer`, `generate_remote`.

### 7.4 AI Experiment Coach (`ai-gateway`)

Reads the dashboard snapshot (loss, GPU, eval) exclusively via **typed FastAPI
routes** (no SSH/shell) and returns a structured "coach card":

- Verdict (`bad`/`promising`/`good`) + confidence, diagnosis & current bottleneck.
- A full parameter suggestion with a rationale *per setting*.
- A single-change suggestion (one controlled change) + ablation plan.
- **Approval-gated action:** no Leonardo action (data, training, eval) runs without
  explicit approval. Every suggestion is validated against the FastAPI limits first;
  without an API key a deterministic local fallback kicks in.

---

## 8. Reproducibility & How to Run

**ML pipeline (on Leonardo, via pixi):**

```bash
# Expand the data
python training_data/generate_sequences.py --family mosfet --count 2000 --seed 42

# Training (single GPU)
python training_data/train_transformer.py --epochs 20 --lr-schedule cosine

# Multi-GPU (DDP)
DDP=1 GPUS=4 sbatch training_data/run_train_transformer.slurm

# Rule-aware eval (reconstructs the held-out split from the checkpoint)
python training_data/evaluate_rules.py

# Generate submission CSVs
python training_data/make_submission.py --tasks all
```

**Dashboard (local):**

```bash
cd dashboard && ./dev.sh   # FastAPI :8000 · AI gateway :8787 · frontend :5173
```

Determinism via a fixed seed (42) for data generation, split, and training.

---

## 9. Infrastructure (Leonardo / CINECA)

- Partition `boost_usr_prod`, account `EUHPC_D30_031`, reservation `s_tra_ncc`.
- 1× **NVIDIA A100**, 120 GB RAM, 8 CPUs, 45 min walltime (for Tiny/Recommended).
- Environment via **pixi**; DDP optional via `torchrun`.

---

## 10. Design Decisions & Honest Limitations

- **Why family dropout instead of a real 4th family?** We have no access to the
  jury's family. Dropout is a direct training signal for "work without a family
  hint" and thus our best approximation of the OOD condition. The eval pass
  `model_generated_unknown` measures exactly this case.
- **Rule-based anomaly:** Since the 10 rules are fully specified, an exact validator
  beats learning an anomaly classifier (deterministic, 100 % explainable, no false
  negatives on known rule types).
- **Greedy completion:** simple and reproducible; sampling/beam search would be a
  possible next step for diversity.
- **To be filled in:** Concrete final metrics (val loss, top-k, per-family
  valid_rate) are produced per run in `outputs/` and are visible directly in the
  dashboard; they depend on the chosen preset/run.

---

## 11. Mapping to the Judging Criteria

- **Working artifact:** end-to-end pipeline + interactive dashboard, no slideware.
- **Reproducibility:** fixed seeds, self-describing checkpoints, one-click rerun;
  eval reconstructs the exact training split.
- **Honest evaluation:** baseline vs. model, sanity anchor, explicit OOD pass, a
  strict `quality_rate` in addition to `valid_rate`.
- **Visible reasoning:** documented in this report and in the code comments; the AI
  coach makes experiment rationales explicit.

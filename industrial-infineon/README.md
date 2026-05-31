# Industrial AI — Learning & Benchmarking Process Logic

> **Hackathon Track:** Can models learn real process logic from semiconductor fabrication sequences — or do they just memorize patterns?



This repository contains everything you need for the **Industrial AI** hackathon track: task briefings (DE + EN), synthetic training data for three semiconductor product families, a sequence generator/validator, and the full process grammar reference.

---

## 📁 Repository Structure

```
.
├── README.md                              ← You are here
├── Track_industrial.md                    ← Full track briefing (German)
├── Track_industrial_en.md                 ← Full track briefing (English)
│
└── training_data/
    ├── README.md                          ← Data & eval quickstart guide
    ├── generation_rules.md                ← Full grammar, forbidden patterns & eval protocol
    ├── generate_sequences.py              ← Sequence generator & validator (CLI)
    │
    │   ── Reference Sequences (1 per family) ──
    ├── synthetic_mosfet.csv               ← MOSFET: 126-step canonical reference
    ├── syntheticIGBT.csv                  ← IGBT:  151-step canonical reference
    ├── syntheticIC.csv                    ← IC:    107-step canonical reference
    │
    │   ── Enriched Variants (step + description + parameters) ──
    ├── MOSFET_Longdescr.csv               ← MOSFET steps with text descriptions
    ├── IGBT_Longdescr.csv                 ← IGBT steps with text descriptions
    ├── IC_Longdescr.csv                   ← IC steps with text descriptions
    ├── MOSFET_longdescription_parameters.csv  ← + realistic fab-level parameters
    ├── IGBT_longdescription_parameters.csv    ← + realistic fab-level parameters
    ├── IC_longdescription_parameters.csv       ← + realistic fab-level parameters
    │
    │   ── Pre-generated Training Sequences (long format) ──
    ├── MOSFET_variants.csv                ← 1,000 MOSFET sequences (~125k rows)
    ├── IGBT_variants.csv                  ← 1,000 IGBT sequences  (~148k rows)
    ├── IC_variants.csv                    ← 1,000 IC sequences     (~115k rows)
    │
    │   ── Python cache (can be ignored) ──
    └── __pycache__/
```

---

## 🚀 Quick Start

### 1. Read the Track Briefing

Start with the track briefing for the full problem statement, evaluation tasks, and scoring criteria:

- **English:** [`Track_industrial_en.md`](./Track_industrial_en.md)
- **Deutsch:** [`Track_industrial.md`](./Track_industrial.md)

### 2. Explore the Data

Each `_variants.csv` uses **long format** — one step per row:

```csv
SEQUENCE_ID,STEP
seq_0001,RECEIVE WAFER LOT
seq_0001,LOT IDENTIFICATION
seq_0001,INITIAL WAFER INSPECTION
...
seq_0002,RECEIVE WAFER LOT
...
```

Load in Python (no external dependencies):

```python
from pathlib import Path
from training_data.generate_sequences import read_csv_sequences

mosfet_seqs = read_csv_sequences(Path("training_data/MOSFET_variants.csv"))
# → dict[sequence_id, list[step_str]]  (1000 sequences)
```

### 3. Generate More Training Data

The combinatorial space is enormous (MOSFET ~51 billion, IGBT ~13 trillion, IC ~6 billion distinct valid sequences). Generate as many as you need:

```bash
# Generate 2000 additional MOSFET sequences
python training_data/generate_sequences.py --family mosfet --count 2000 --output extra_mosfet.csv --seed 42

# Validate an existing file against all 10 process-logic rules
python training_data/generate_sequences.py --validate training_data/MOSFET_variants.csv

# Estimate combinatorial space for a family
python training_data/generate_sequences.py --family igbt --estimate-only
```

Supported families: `mosfet`, `igbt`, `ic`

### 4. Understand the Grammar & Rules

**[`training_data/generation_rules.md`](./training_data/generation_rules.md)** is the authoritative reference. It documents:

| Section | Content                                                                |
| ------- | ---------------------------------------------------------------------- |
| §1      | Full step vocabulary (~120 distinct step strings across 12 categories) |
| §2      | Formal process grammar with block notation for each family             |
| §3      | **10 forbidden patterns** (process-logic violations used in eval)      |
| §4      | 11 variation axes you can change while keeping sequences valid         |
| §5      | Shared evaluation protocol — metrics, file formats, submission format  |
| §6      | CLI usage examples                                                     |

---

## 🧪 Evaluation Tasks

The organizers distribute two eval input files at the start of the hackathon:

| File                     | Purpose     | Content                                      |
| ------------------------ | ----------- | -------------------------------------------- |
| `eval_input_valid.csv`   | Tasks 1 & 2 | 600 partial sequences at 60%/80% completion  |
| `eval_input_anomaly.csv` | Task 3      | 987 mixed sequences (valid + rule-violating) |

### Task Overview

| #   | Task                     | Input                            | Key Metrics                                                  |
| --- | ------------------------ | -------------------------------- | ------------------------------------------------------------ |
| 1   | **Next-step prediction** | Partial sequence                 | Top-1/3/5 Accuracy, MRR                                      |
| 2   | **Sequence completion**  | Partial sequence (60% or 80%)    | Exact Match, Edit Distance, Token/Block Accuracy             |
| 3   | **Anomaly detection**    | Full sequence (valid or invalid) | Binary Acc, Precision, Recall, F1, ROC-AUC, Rule Attribution |
| 4*  | **OOD generalization**   | Unknown product family           | Performance drop ID → OOD                                    |

*\*Task 4 is evaluated post-submission by organizers on a hidden 4th product family.*

### Self-Evaluation

The scoring script `eval_metrics.py` (provided by organizers) requires no external dependencies:

```bash
python eval_metrics.py --task anomaly --ground-truth <ground_truth.csv> --predictions <your_output.csv>
```

---

## 📊 Data at a Glance

| Family | Reference Steps | Pre-generated Sequences | Combinatorial Space |
| ------ | --------------- | ----------------------- | ------------------- |
| MOSFET | 126             | 1,000 (~125k rows)      | ~51 billion         |
| IGBT   | 151             | 1,000 (~148k rows)      | ~13 trillion        |
| IC     | 107             | 1,000 (~115k rows)      | ~6 billion          |

All sequences follow the same **backbone flow**:

> Logistics → Clean → Family Prep → Oxidation → Litho/Etch/Implant Cycles → ILD → Via → Metal → Passivation → Backside → Final Inspection → Test → Ship

But each family has its own specific blocks, cycle counts, and optional steps.

---

## 🔑 Key Concepts

- **Each step string is one token** (~120 vocabulary items across all families)
- **Sequences always start** with `RECEIVE WAFER LOT` and **end** with `SHIP LOT`
- **Family as context** helps: step names are mostly shared; differences are in which optional blocks appear and cycle counts
- **Process order matters**: several rules enforce ordering constraints (e.g., deposition requires a prior clean; electrical tests must follow passivation)
- **Scaling experiments**: compare models trained on 100 vs. 1,000 vs. 5,000+ sequences — this is an explicit stretch goal

---

## 🔗 Suggested Tech Stack

- Python + PyTorch (or comparable framework for sequence models)
- Transformer-based models, LLMs, or other sequence architectures
- Experiment tracking (e.g., Weights & Biases, MLflow)
- Training on **Leonardo cluster** (GPU quota per team: TBD)

---

## 📋 Useful Links

| Resource                                                                       | Description                                 |
| ------------------------------------------------------------------------------ | ------------------------------------------- |
| [`Track_industrial_en.md`](./Track_industrial_en.md)                           | Full English track briefing                 |
| [`Track_industrial.md`](./Track_industrial.md)                                 | Full German track briefing                  |
| [`training_data/README.md`](./training_data/README.md)                         | Data & eval quickstart                      |
| [`training_data/generation_rules.md`](./training_data/generation_rules.md)     | Grammar, forbidden patterns & eval protocol |
| [`training_data/generate_sequences.py`](./training_data/generate_sequences.py) | Generator & validator CLI                   |

---

## 🎯 Task Levels

| Level                 | Goal                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Level 1**           | Understand data, generate synthetic sequences, build a baseline for next-step prediction / sequence completion |
| **Level 2**           | Train a model, tune it, benchmark baseline vs. trained vs. optimized model                                     |
| **Level 3 / Stretch** | Systematically compare scaling (model size, data volume, compute) and generalize to unseen sequences           |

---

> 🎁 **Bonus (Optional):** Build a small frontend dashboard to visualize your results — training loss curves, metric comparisons (Top-1/3/5 Accuracy, MRR, F1, ROC-AUC across families), baseline-vs-trained side-by-side, anomaly confusion matrices, scaling plots, and before/after prediction examples. Anything from a Streamlit/Gradio quick-app to a full React dashboard counts. This is **not** required but makes your submission stand out!

**Good luck, and may your models learn true process logic — not just surface patterns! 🏭🤖**
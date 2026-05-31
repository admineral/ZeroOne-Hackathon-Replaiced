# Process Sequence Training Data
### Hackathon Track: Prozesslogik lernen und benchmarken

This folder is your complete starter pack for the **Prozesslogik lernen und benchmarken** hackathon track.
It contains all training data, tooling, and reference material you need.
The data covers three semiconductor product families — **MOSFET**, **IGBT**, and **IC** — each represented as ordered sequences of fabrication process steps.

---

## Files

| File | Description |
|---|---|
| `synthetic_mosfet.csv` | Original reference sequence — MOSFET, 126 steps |
| `syntheticIGBT.csv` | Original reference sequence — IGBT, 151 steps |
| `syntheticIC.csv` | Original reference sequence — IC, 107 steps |
| `MOSFET_Longdescr.csv` | MOSFET steps with text descriptions |
| `IGBT_Longdescr.csv` | IGBT steps with text descriptions |
| `IC_Longdescr.csv` | IC steps with text descriptions |
| `MOSFET_longdescription_parameters.csv` | MOSFET steps with descriptions + realistic fab-level parameters |
| `IGBT_longdescription_parameters.csv` | IGBT steps with descriptions + realistic fab-level parameters |
| `IC_longdescription_parameters.csv` | IC steps with descriptions + realistic fab-level parameters |
| `MOSFET_variants.csv` | 1,000 valid MOSFET process sequences (~125 steps each, 125k rows total) |
| `IGBT_variants.csv` | 1,000 valid IGBT process sequences (~148 steps each, 148k rows total) |
| `IC_variants.csv` | 1,000 valid IC process sequences (~115 steps each, 115k rows total) |
| `generate_sequences.py` | Script to generate and validate additional sequences |
| `eval_metrics.py` | Official scoring script — use to self-evaluate your model against all three tasks |
| `generation_rules.md` | Full grammar reference, forbidden-pattern documentation, and eval protocol |

---

## Data Format

Each variant CSV uses **long format** — one step per row:

```
SEQUENCE_ID,STEP
MOSFET_0001,RECEIVE WAFER LOT
MOSFET_0001,LOT IDENTIFICATION
MOSFET_0001,INITIAL WAFER INSPECTION
...
MOSFET_0002,RECEIVE WAFER LOT
...
```

To load all sequences for one family in Python (no dependencies beyond stdlib):
```python
from pathlib import Path
from generate_sequences import read_csv_sequences
seqs = read_csv_sequences(Path("MOSFET_variants.csv"))
# seqs: dict[sequence_id -> list[step_str]]
```

---

## Generating Additional Training Data

The combinatoric space is large (MOSFET ~51 billion, IGBT ~13 trillion, IC ~6 billion distinct valid sequences). You can generate as many additional sequences as you need:

```bash
# Generate 2000 more MOSFET sequences (append to existing data or save separately)
python generate_sequences.py --family mosfet --count 2000 --output my_extra_mosfet.csv --seed 123

# Validate your own generated sequences against all 10 process rules
python generate_sequences.py --validate my_extra_mosfet.csv

# Estimate combinatoric space for a family
python generate_sequences.py --family igbt --estimate-only
```

Supported families: `mosfet`, `igbt`, `ic`

All three families share the same backbone process flow (logistics → clean → device fabrication → ILD/via/metal stack → passivation → test → ship) with family-specific preparation blocks and cycle counts.

---

## Process Grammar & Forbidden Patterns

`generation_rules.md` is the authoritative reference for the data. It documents:

- **Section 1** — Full step vocabulary organised into 12 functional categories
- **Section 2** — Formal process grammar with block notation (`[optional]`, `A|B`, `{m..n}` repetition)
- **Section 3** — All 10 forbidden patterns: which step triggers the violation, what prerequisite is missing, and example violations
- **Section 4** — 11 independent variation axes (things you can change to systematically study model generalisation)
- **Section 5** — Shared evaluation protocol: file formats, metrics, and submission formats
- **Section 6** — CLI usage examples

Understanding the grammar helps interpret why sequences look the way they do and gives insight into what process knowledge a well-trained model should learn.

---

## Evaluation Tasks

The organizers will distribute two eval input files at the start of the hackathon:

| File | Used for | Contents |
|---|---|---|
| `eval_input_valid.csv` | Tasks 1 & 2 | 600 partial sequences — `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE` |
| `eval_input_anomaly.csv` | Task 3 | 987 unlabeled sequences — `EXAMPLE_ID, FAMILY, SEQUENCE` |

Three tasks are scored by the organizers against a fixed ground truth:

| # | Task | Input | Metric(s) |
|---|---|---|---|
| 1 | **Next-step prediction** | `eval_input_valid.csv` — given a partial sequence, rank the 5 most likely next steps | Top-1 Accuracy, Top-3 Accuracy, Top-5 Accuracy, MRR |
| 2 | **Sequence completion** | `eval_input_valid.csv` — given a partial sequence (60% or 80% complete), predict all remaining steps | Exact Match Rate, Normalized Edit Distance, Token Accuracy, Block-level Accuracy |
| 3 | **Anomaly detection** | `eval_input_anomaly.csv` — given a full sequence, determine if it is valid or contains a process-rule violation | Binary Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Rule Attribution Accuracy |

Submission file formats for all three tasks are documented in `generation_rules.md` (Section 5).

Generate organizer-ready prediction files with:

```bash
# Rule-based Task 3 only; does not require a model checkpoint.
python make_submission.py --tasks anomaly

# All tasks; Tasks 1 & 2 require a local trained checkpoint + vocab.
python make_submission.py --tasks all \
  --checkpoint ../outputs/transformer/transformer_model.pt \
  --vocab ../outputs/transformer/vocab.json
```

Outputs are written to `../../participant_files/submission/`:
`predictions_nextstep.csv`, `predictions_completion.csv`, and
`predictions_anomaly.csv`.

You can self-evaluate your model locally using the provided `eval_metrics.py` script (no external dependencies):

```bash
# Example: score your anomaly detection output
python eval_metrics.py --task anomaly --ground-truth <ground_truth.csv> --predictions <your_output.csv>
```

---

## Tips for Training

- **Tokenisation**: Each step string is one token (e.g., `"DEPOSIT GATE OXIDE"` as a single unit). The vocabulary is ~120 distinct step strings across all families.
- **Sequence length**: MOSFET ~125 steps, IGBT ~148 steps, IC ~115 steps. Sequences always start with `RECEIVE WAFER LOT` and end with `SHIP LOT`.
- **Family as context**: Step names are mostly shared across families; the key differences are which optional blocks appear and how many lithography cycles run. Including family as a conditioning token is likely helpful.
- **Process order matters**: Several rules are about ordering (e.g., deposition requires a prior clean; electrical test must come after passivation). Models that learn positional constraints will outperform pure frequency-based approaches.
- **Scaling experiment**: Compare models trained on 100 vs. 1,000 vs. 5,000+ sequences to observe scaling effects — this is explicitly one of the track's stretch goals.

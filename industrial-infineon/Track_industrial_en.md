# Track: Industrial AI
## Use Case Title: Learning and Benchmarking Process Logic

**Challenge Owner:** TBD
**Mentor(s):** TBD
**Difficulty Level:** Advanced to Expert
**Estimated Scope:** Yes, the case is realistically achievable within 36 hours because work is done with synthetic data, model selection is open, and even a reduced scope with data generation, a training run, and a robust evaluation constitutes a meaningful deliverable.

---

### 1. Problem Statement (3–5 sentences)
Many industrial processes can be described as long sequences of steps whose meaning depends heavily on order, intermediate steps, and process logic. In semiconductor manufacturing as well, products are created through complex process routes in which materials are deposited, patterned, modified, and removed again. The hackathon abstracts this problem and investigates how well models can learn such process sequences, predict next steps, and generalize to modified or unseen workflows. The key question is whether a model merely reproduces known patterns or builds a transferable understanding of the underlying process logic.

### 2. Why This Matters (Business Context)
- The challenge lies in the fact that while industrial process workflows are highly structured, this structure is difficult to model robustly and compare without proper training and benchmarking.
- This primarily benefits teams from process development, ML engineering, and research who want to systematically investigate which model and data strategies work for process-like sequences.
- The connection to **European AI Sovereignty** lies in the focus on own data generation, training on the cluster, reproducible evaluation pipelines, and the comparison of different model sizes. The emphasis is on building robust infrastructure and model competency rather than a pure API or wrapper approach.

### 3. Expected Outcome / Definition of Done
- **Minimum Viable Result:** A reproducible end-to-end workflow with synthetic data generation, at least one trained model, a baseline-vs.-post-training comparison, and a clearly documented benchmark for process sequences.
- **Stretch Goals:** Comparison of multiple model sizes or architectures, analysis of scaling effects, generalization to unseen or modified sequences, inclusion of optional process parameters, and a small demonstrator.
- **Learning Objectives for Participants:** Setting up and training sequence models on cluster infrastructure, generating synthetic training data, systematic benchmarking, analyzing scaling effects, and properly evaluating before/after behavior.
- **Format:** Training run with eval report; optionally complemented by a demo, prototype, or demonstrator.
- **Demonstrator Stretch Goal:** A small before/after demo that shows sample outputs of a baseline model and a trained model side by side, e.g., for next-step prediction, sequence completion, or the detection of atypical process steps.

### 4. Model Specification
- **Type:** Freely selectable. Particularly suitable are transformer-based sequence models, LLMs, other sequence models, or hybrid approaches.
- **Base Model:** Freely selectable, e.g., training from scratch on synthetic data or using an open base model as a starting point.
- **Model Sizes:** One or more sizes are explicitly encouraged to enable comparison of small-vs.-large setups and scaling effects.
- **Training Methods:** Freely selectable. Obvious choices include next-step prediction, sequence completion, SFT, as well as RL-based approaches like GRPO or other training objectives suited to the model.
- **Constraints:** Focus on reproducible training and evaluation setups on cluster infrastructure and on an open, transparent stack rather than black-box API solutions.

### 5. Task Structure (Levels)
- **Level 1:** Understand the initial data, generate additional synthetic data, and set up a robust baseline for next-step prediction or sequence completion.
- **Level 2:** Train a model, then specifically further tune or improve it, and make the difference between baseline, trained model, and subsequently optimized model visible with a custom benchmark for process understanding.
- **Level 3 / Stretch:** Systematically analyze scaling effects, e.g., by comparing model sizes, compute time, and data volume, as well as their impact on performance and generalization. Optionally, different architectures or process parameters can also be included.

### 6. Data & Resources
- **Datasets:** Data is available for three product families: IC, IGBT, and MOSFET. Per family, there is a Long-Description variant with `STEP` and `DESCRIPTION`, a variant with additional `REALISTIC FAB-LEVEL PARAMETERS`, and a synthetic sequence file.
- **Data Format:** CSV files with process-related step sequences. Depending on the file, the data contains only `STEP` or additionally textual descriptions and realistic fab-level parameters per process step. The pre-generated training sequences are provided in **long format** (`SEQUENCE_ID, STEP`; one row per step).
- **Data Volume:** In the `training_data/` folder, **1,000 pre-generated, validated sequences** per product family are available (3,000 sequences total, each approx. 115–150 steps). Additionally, the original nine reference CSV files with around 1,100 data rows are available in the main directory. The combinatorial space of valid sequences is very large (MOSFET ~51 billion, IGBT ~13 billion, IC ~6 billion distinct sequences), so teams can generate any number of additional training sequences using the included script `training_data/generate_sequences.py`. The underlying process grammar, all validation rules, and the eval protocol specification are documented in `training_data/generation_rules.md`.
- **APIs/Systems:** No mandatory APIs specified; the provided CSV sample data and the cluster environment are primarily relevant.
- **Compute:** Training on the Leonardo cluster is explicitly intended. GPU quota per team: TBD.
- **NDAs / Data Protection:** Data protection and clearance for the provided data are resolved; use within the framework of the hackathon is permitted.

### 7. Evaluation & Benchmarking
- **Eval Setup:** A shared, fixed eval set is available, distributed by the organizers. It consists of two subsets:
  - **Next-Step and Completion Tasks** (`eval_input_valid.csv`): 600 entries – 100 held-out sequences per family, each truncated at 60% and 80% completion.
  - **Anomaly Detection** (`eval_input_anomaly.csv`): 987 mixed sequences – 387 with deliberately injected process rule violations (labeled across 10 rule types) and 600 valid sequences, shuffled and unlabeled.
- **Three Submission Tasks (teams submit results):**

  | #   | Task                        | Input                                        | Metric(s)                                                                                      |
  | --- | --------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
  | 1   | **Next-Step Prediction**    | Partial sequence                             | Top-1 Accuracy, Top-3 Accuracy, Top-5 Accuracy, MRR                                            |
  | 2   | **Sequence Completion**     | Partial sequence (60% or 80%)                | Exact Match Rate, Normalized Edit Distance, Token Accuracy, Block-level Accuracy                |
  | 3   | **Anomaly Detection**       | Complete sequence (with/without rule violation) | Binary Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Rule Attribution Accuracy |

- **Additional Generalization Reporting (by organizers only, after submission):**

  | #   | Task                    | Input                                            | Metric(s)                                                       |
  | --- | ----------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
  | 4   | **OOD Generalization**  | ID vs. OOD split on unknown product family        | Performance drop ID → OOD (per main metric from Tasks 1–3)        |

  Teams do not submit separate results for Task 4. The organizers apply the submitted models to the OOD dataset after submission and calculate the performance drop.

- **Scoring:** The evaluation script `eval_metrics.py` is ready to use for all three tasks and requires no external dependencies. It provides a detailed report per task with breakdowns by family and truncation point.
- **Generalization (Task 4):** Generalization ability is additionally evaluated on a fourth product family not included in the training and not disclosed to participants. This family is unknown to participants and is used exclusively by the organizers for post-submission evaluation. The performance drop (ID → OOD) across all main metrics is assessed.
- **Test Frequency:** Automated evaluation intervals during training are recommended to make learning progress, overfitting, and scaling effects visible. Teams define the specific frequency themselves.
- **Inference Stack:** A UI is optional. For live tests, a simple inference workflow or notebook-based demonstrator is sufficient. A direct side-by-side comparison of baseline output and trained model output on identical inputs is particularly interesting.
- **Visualization:** At minimum, loss curves, performance metrics over time, and comparative visualizations between baseline and trained model are expected; additional visualizations are welcome.
- **Comparability:** All teams work with the same eval set and the same metrics; results are therefore directly comparable.

**Example Demonstrator Outputs:**

- **Baseline Model:** Input: `RECEIVE WAFER LOT -> LOT IDENTIFICATION -> INITIAL WAFER INSPECTION -> ?` Output: `ETCH` or a generic, contextually implausible next step.
- **Trained Model:** Input: `RECEIVE WAFER LOT -> LOT IDENTIFICATION -> INITIAL WAFER INSPECTION -> ?` Output: `MEASURE THICKNESS` or `MEASURE INITIAL THICKNESS` with higher plausibility in the context of the process sequence.
- **Baseline Model:** An incomplete sequence with a missing cleaning step is completed without recognizable process logic.
- **Trained Model:** Recognizes that a plausible preparation or cleaning step is missing before a patterning or deposition step and completes the sequence accordingly.

### 8. Technical Notes
- Suggested Tech Stack: Python, PyTorch or comparable frameworks for sequence models, complemented by proper experiment tracking and evaluation scripts.
- Known Pitfalls: Quality and distribution of synthetic data, fair comparison between models, meaningful generalization tests, proper checkpointing on cluster infrastructure, and a benchmark that measures more than mere memorization.
- Known Baseline: The provided sample data serves as a starting point; a specific internal model baseline is not evident from this briefing and should be added if needed.

### 9. Evaluation Criteria (track-specific)
- Technical depth and traceable model and data decisions
- Quality of the training and benchmark setup on real infrastructure
- Reproducibility and clarity of evaluation
- Expressiveness of the comparison between baseline, trained model, and optional scaling variants
- Quality of demo, visualization, and result presentation

### 10. Contact & Support During the Event
- On-site Mentor: Simeon
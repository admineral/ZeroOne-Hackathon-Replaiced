# Report — Industrial AI (Infineon Track)

**Zero One Hack_01 · Lernen & Benchmarken von Prozesslogik**

> Kann ein Modell *echte* Halbleiter-Prozesslogik lernen — oder merkt es sich nur
> Muster? Wir trainieren auf **3 bekannten Produktfamilien** und rüsten das Modell
> gezielt für die **geheime 4. Familie**, mit der die Jury bewertet. Die komplette
> HPC-Pipeline auf dem Leonardo-Cluster ist über ein **Klick-Dashboard mit
> AI-Experiment-Coach** steuerbar — ohne ein einziges Terminal-Kommando.

---

## 1. TL;DR

- **Modell:** GPT-artiger, kausaler Transformer (`TinyCausalTransformer`) auf
  Prozess-Schritt-Tokens. Skalierbar von ~0,5 M bis ~180 M Parameter über vier
  Klick-Presets.
- **Unser Kern-Kniff:** **Random Family-Dropout** (30 %). Das Familien-Token wird
  im Training zufällig durch `<FAMILY_UNKNOWN>` ersetzt → das Modell lernt, auch
  ohne bekannte Familie plausible Sequenzen zu erzeugen → **OOD-Generalisierung auf
  die unbekannte 4. Familie der Jury**.
- **Evaluation:** Regelbewusste Eval gegen **10 formale Prozesslogik-Regeln**, plus
  N-Gramm-Baseline. Zwei Eval-Pässe (mit echter Familie / mit `UNKNOWN`) belegen die
  OOD-Robustheit direkt.
- **Engineering:** Lokales **FastAPI + React + AI-Gateway**-Dashboard, das die
  Leonardo-Slurm-Pipeline per SSH automatisiert: generieren → hochladen →
  trainieren → live überwachen → evaluieren → einreichen.
- **Submission:** Alle drei Tasks erzeugt (Next-Step, Completion, Anomaly). Die
  Anomalie-Erkennung ist regelbasiert: **387 / 987** Sequenzen als ungültig erkannt.

---

## 2. Problemstellung & unser Ansatz

Industrielle Prozesse sind lange, streng geordnete Schrittfolgen, deren Bedeutung
von Reihenfolge und Zwischenschritten abhängt. Der Track stellt drei
Halbleiter-Produktfamilien bereit — **MOSFET, IGBT, IC** — und bewertet auf einer
**vierten, nicht offengelegten Familie** (Out-of-Distribution).

**Unsere Leitidee:** Ein Modell, das nur Muster der drei Trainingsfamilien
auswendig lernt, scheitert an der 4. Familie. Wir erzwingen daher
*familien-agnostisches* Lernen, indem wir die Familieninformation beim Training
kontrolliert „wegnehmen" (siehe §4.3). So muss das Modell auf die *Prozesslogik*
selbst zurückgreifen statt auf den Familien-Hinweis.

---

## 3. Repository-Struktur

```
zero1hack/
├── industrial-infineon/
│   └── training_data/              # ML-Pipeline (Daten, Modell, Eval, Submission)
│       ├── generate_sequences.py   # Grammatik-Generator + Validator (10 Regeln)
│       ├── sequence_data.py        # Tokenisierung, Vokabular, Packing, Family-Tokens
│       ├── train_transformer.py    # Training (DDP, bf16, Checkpointing)
│       ├── evaluate_rules.py       # Regelbewusste Evaluation
│       ├── baseline_ngram.py       # N-Gramm-Baseline (Next-Step)
│       ├── make_submission.py      # Erzeugt die 3 Organizer-CSVs
│       ├── eval_metrics.py         # Metrik-Definitionen + Qualitäts-Schwellen
│       └── *.slurm                 # Slurm-Jobs für Leonardo
├── dashboard/
│   ├── backend/                    # FastAPI: SSH-Orchestrierung der Pipeline
│   ├── frontend/                   # React/Vite: Klick-UI mit Live-Charts
│   ├── ai-gateway/                 # Node: AI Experiment Coach
│   └── dev.sh                      # Startet alle 3 lokalen Dienste
└── participant_files/submission/   # Erzeugte Vorhersage-CSVs
```

---

## 4. ML-Pipeline

### 4.1 Daten

- **Drei Produktfamilien** in *Long-Format*-CSVs (`SEQUENCE_ID, STEP`, eine Zeile
  je Schritt). Jede Sequenz beginnt mit `RECEIVE WAFER LOT` und endet mit
  `SHIP LOT`.
- **Datengenerierung** über `generate_sequences.py`, das die formale Prozess-
  Grammatik aus `generation_rules.md` umsetzt und nur regel-valide Sequenzen
  produziert.
- **Skalierungsexperiment:** Wir haben den Datensatz massiv erweitert
  (`dataset_manifest.json`):

| Familie | Generierte Sequenzen | Step-Zeilen |
|---|---:|---:|
| MOSFET | 100.000 | 12.525.095 |
| IGBT | 100.000 | 14.804.325 |
| IC | 100.000 | 11.514.196 |
| **Gesamt** | **300.000** | **38.843.616** |

> Seed 42, deterministisch reproduzierbar.

- **Memmap-Packing:** Für große Datensätze wird der Korpus in ein
  Memmap-Blob gepackt (`packed/`). Unter DDP teilen sich alle Ranks dieselben
  Daten über den OS-Page-Cache (RAM ≈ 0 statt voller Materialisierung).

### 4.2 Tokenisierung & Familien-Konditionierung

- **Ein Prozess-Schritt = ein Token** (z. B. `"DEPOSIT GATE OXIDE"`). Vokabular
  ≈ 120 Schritt-Strings plus Spezial-Tokens.
- Jede Sequenz wird kodiert als
  `[<BOS>, <FAMILY_x>, …Schritte…, <EOS>]`.
- **Vier Familien-Tokens:** `<FAMILY_MOSFET>`, `<FAMILY_IGBT>`, `<FAMILY_IC>` und
  ein neutrales `<FAMILY_UNKNOWN>` für unbekannte/fehlende Familien.

### 4.3 Random Family-Dropout (unser OOD-Mechanismus)

Beim Training wird das echte Familien-Token mit Wahrscheinlichkeit
**`family_dropout = 0.30`** durch `<FAMILY_UNKNOWN>` ersetzt — **pro Beispiel und
pro Epoche neu gewürfelt**:

```
<BOS> <FAMILY_IC> …Schritte…   --30%-->   <BOS> <FAMILY_UNKNOWN> …Schritte…
```

Effekt: Das Modell lernt, sowohl *mit* bekannter Familie als auch *ohne* Familien-
Prefix plausibel weiterzuschreiben. Genau das ist die Situation der Jury-Bewertung
auf einer unbekannten 4. Produktfamilie.

### 4.4 Modell

`TinyCausalTransformer` — GPT-Stil, kausale Maske, gelernte Positions-Embeddings,
`GELU`, finaler `LayerNorm`. Default-Konfiguration und Klick-Presets:

| Preset | d_model | Layer | Heads | Batch | ≈ Parameter |
|---|---:|---:|---:|---:|---:|
| Tiny (Default) | 128 | 2 | 4 | 32 | ~0,5 M |
| Recommended | 256 | 4 | 8 | 256 | ~3,5 M |
| Scale up | 512 | 6 | 8 | 128 | ~22 M |
| Even bigger | 1024 | 12 | 16 | 256 | ~180 M |

Diese Presets ermöglichen das vom Track geforderte **Skalierungs-Experiment**
(klein vs. groß, Daten- vs. Modellgröße) per Klick.

### 4.5 Training

Datei: `train_transformer.py`. Default-Hyperparameter:

- Objective: Next-Token-Prediction, `CrossEntropyLoss` (Padding ignoriert,
  optional Label-Smoothing).
- Optimizer: **AdamW**, `lr = 3e-4`, `weight_decay = 0.01`, Grad-Clipping 1.0.
- LR-Schedule: konstant oder **Linear-Warmup + Cosine-Decay** (`--lr-schedule cosine`).
- Präzision: **bf16-Autocast + TF32** auf A100 (kein GradScaler nötig).
- Split: 80 % Train / 10 % Val / 10 % Test (deterministisch, Seed 42).
- **Multi-GPU:** `DistributedDataParallel` via `torchrun` (1 Prozess pro GPU),
  Metriken werden über Ranks all-reduced.
- **Robustes Checkpointing:** „best-by-val-loss" wird *während* des Trainings
  atomar gespeichert; ein `SIGTERM`/Walltime-Kill flusht den aktuellen besten
  Checkpoint (markiert als `interrupted`), damit kein Lauf verloren geht.
- **Selbstbeschreibende Checkpoints:** Hyperparameter, Split-Ratios, Seed,
  `max_sequences` und Slurm-Job-ID werden mitgespeichert, sodass die Evaluation den
  identischen Held-Out-Split rekonstruiert (kein Leakage).
- **Telemetrie:** `train_log.csv` (Loss/Acc/LR/GPU-Memory je Epoche),
  `train_stats.json` (Parameterzahl, Peak-GPU-Memory, Trainingszeit) und ein
  `gpu_timeline.csv` (nvidia-smi alle 2 s) für die Dashboard-Charts.

---

## 5. Evaluation

### 5.1 Regelbewusste Eval (`evaluate_rules.py`)

Wir bewerten generierte Sequenz-Vervollständigungen gegen **10 formale
Prozesslogik-Regeln** (aus `generate_sequences.py` / `generation_rules.md`):

| Regel | Verletzung |
|---|---|
| `RULE_DEP_NO_CLEAN` | Deposition ohne vorherige Reinigung |
| `RULE_METAL_ETCH_NO_LITHO` | Metall-Ätzung ohne vorausgehende Lithografie |
| `RULE_ETCH_NO_MASK` | Ätzen ohne Maske |
| `RULE_LITHO_LEVEL_SKIP` | Übersprungenes Lithografie-Level |
| `RULE_IMPLANT_NO_MASK` | Implantation ohne Maske |
| `RULE_CMP_NO_DEP` | CMP ohne vorherige Deposition |
| `RULE_PAD_OPEN_BEFORE_DEP` | Pad-Open vor zugehöriger Deposition |
| `RULE_TEST_BEFORE_PASSIVATION` | Elektr. Test vor Passivierung |
| `RULE_SHIP_BEFORE_TEST` | Versand vor Test |
| `RULE_BACKSIDE_BEFORE_PASSIVATION` | Backside-Schritt vor Passivierung |

**Gemessene Kennzahlen** (gerollt je Quelle/Completion-Fraction):

- `valid_rate` — Anteil regel-valider Vervollständigungen.
- `quality_rate` — strenger: valid **und** plausible Länge (Längenverhältnis
  0,8–1,25), max. 2 aufeinanderfolgende Wiederholungen, Suffix-Accuracy ≥ 0,5.
- `mean_suffix_acc`, `mean_jaccard`, `mean_len_ratio`, `eos_rate`.

**Drei Eval-Zeilen pro Lauf** machen die Kernaussage direkt sichtbar:

1. `heldout_source` — echte Held-Out-Rezepte (Sanity-Anker, ≈ 1,0 überall).
2. `model_generated` — Modell **mit** echter Familie.
3. `model_generated_unknown` — Modell **mit `<FAMILY_UNKNOWN>`** → der OOD-Test:
   bleiben die Vervollständigungen valide, *ohne* dass die Familie bekannt ist?

Vervollständigt wird greedy bei **60 % und 80 %** Sequenz-Cut; der Held-Out-Split
wird aus den Checkpoint-Metadaten exakt rekonstruiert.

### 5.2 Baseline (`baseline_ngram.py`)

N-Gramm-Modell mit Backoff (Ordnungen 1/2/3/5) für Next-Step-Prediction, bewertet
mit **Top-1/3/5-Accuracy und MRR**. Liefert den Vergleichspunkt „lernt der
Transformer mehr als reine Häufigkeitsstatistik?".

---

## 6. Submission

`make_submission.py` erzeugt die drei organizer-fertigen CSVs aus den offiziellen
Eval-Inputs nach `participant_files/submission/`:

| Task | Datei | Inhalt | Verfahren |
|---|---|---|---|
| 1 — Next-Step | `predictions_nextstep.csv` | 600 Zeilen, Top-5-Ranking | Modell |
| 2 — Completion | `predictions_completion.csv` | 600 Zeilen, Rest-Sequenz (60 %/80 %) | Modell (greedy) |
| 3 — Anomaly | `predictions_anomaly.csv` | 987 Zeilen, gültig/ungültig + Regel | Regelbasiert |

- **Task 3** braucht kein Modell: der Validator markiert jede Sequenz und gibt die
  zuerst verletzte Regel als `PREDICTED_RULE` aus. **Ergebnis: 387 / 987 Sequenzen
  als ungültig erkannt.**
- **Tasks 1 & 2** nutzen den trainierten Checkpoint + Vokabular. Unbekannte Familien
  fallen automatisch auf `<FAMILY_UNKNOWN>` zurück (konsistent mit dem Training).

---

## 7. Dashboard — UI & Automatisierung

Statt manuell `ssh`, `sbatch`, `scp` und Log-Tailing: ein lokales Web-Dashboard,
das die komplette Pipeline automatisiert.

### 7.1 Architektur

```
Browser (React) --REST/SSE--> FastAPI (127.0.0.1) --paramiko/SSH--> Leonardo Login --sbatch--> A100 GPU-Job
```

Der Browser SSHt nie selbst. `dev.sh` startet drei lokale Dienste:
**FastAPI-Backend (:8000)**, **AI-Gateway (:8787)**, **Vite-Frontend (:5173)**.

### 7.2 UI (React / Vite)

- **Pipeline-Rail** — Klick-Flow Setup → Dataset → Monitor → Evaluate → Submission.
- **Live-Loss-Chart** — Trainings-Loss streamt epochenweise per SSE.
- **Log-Drawer** — Job-`.out`/`.err` werden live mitgelesen.
- **Results-Panel** — Valid-Rate-Karten + Regelverletzungs-Tabelle aus der Eval.
- **Submission-Card** — Tasks + Checkpoint-Auswahl, ein Klick.
- **Dataset-Verwaltung** + **Resource-Panel** (GPU-Auslastung/Power).

### 7.3 Automatisierung (FastAPI + paramiko)

1. Verbinden über SSH (Passwort nur in gitignored `.env`, nie im Browser).
2. Daten auf Leonardo generieren — versioniert in `datasets/<id>/`.
3. Skripte via `sftp` hochladen.
4. Umgebung prüfen (torch + CUDA).
5. Training: `sbatch` absetzen, Job-ID erfassen.
6. Queue automatisch pollen (`squeue`, alle 5 s) + Ressourcen (`sacct`).
7. Live-Streams für Loss, Logs und GPU-Timeline (SSE).
8. Evaluieren + Submission-Job; die Vorhersage-CSVs werden automatisch
   zurückgeholt.

Run-Keys: `transformer`, `ngram`, `eval_transformer`, `generate_remote`.

### 7.4 AI Experiment Coach (`ai-gateway`)

Liest den Dashboard-Snapshot (Loss, GPU, Eval) ausschließlich über **typisierte
FastAPI-Routen** (kein SSH/Shell) und liefert eine strukturierte „Coach-Card":

- Verdict (`bad`/`promising`/`good`) + Konfidenz, Diagnose & aktueller Bottleneck.
- Vollständiger Parameter-Vorschlag mit Begründung *pro Einstellung*.
- Single-Change-Vorschlag (eine kontrollierte Änderung) + Ablations-Plan.
- **Approval-gated Action:** keine Leonardo-Aktion (Daten, Training, Eval) läuft
  ohne explizite Bestätigung. Jeder Vorschlag wird vorab gegen die FastAPI-Limits
  validiert; ohne API-Key greift ein deterministischer lokaler Fallback.

---

## 8. Reproduzierbarkeit & Ausführung

**ML-Pipeline (auf Leonardo, via pixi):**

```bash
# Daten erweitern
python training_data/generate_sequences.py --family mosfet --count 2000 --seed 42

# Training (Single-GPU)
python training_data/train_transformer.py --epochs 20 --lr-schedule cosine

# Multi-GPU (DDP)
DDP=1 GPUS=4 sbatch training_data/run_train_transformer.slurm

# Regelbewusste Eval (rekonstruiert den Held-Out-Split aus dem Checkpoint)
python training_data/evaluate_rules.py

# Submission-CSVs erzeugen
python training_data/make_submission.py --tasks all
```

**Dashboard (lokal):**

```bash
cd dashboard && ./dev.sh   # FastAPI :8000 · AI-Gateway :8787 · Frontend :5173
```

Determinismus durch festen Seed (42) für Datengenerierung, Split und Training.

---

## 9. Infrastruktur (Leonardo / CINECA)

- Partition `boost_usr_prod`, Account `EUHPC_D30_031`, Reservation `s_tra_ncc`.
- 1× **NVIDIA A100**, 120 GB RAM, 8 CPUs, 45 min Walltime (für Tiny/Recommended).
- Umgebung über **pixi**; DDP optional über `torchrun`.

---

## 10. Designentscheidungen & ehrliche Grenzen

- **Warum Family-Dropout statt einer echten 4. Familie?** Wir haben keinen Zugriff
  auf die Jury-Familie. Dropout ist ein direktes Trainingssignal für „arbeite ohne
  Familien-Hinweis" und damit unsere beste Annäherung an die OOD-Bedingung. Der
  Eval-Pass `model_generated_unknown` misst genau diesen Fall.
- **Anomalie regelbasiert:** Da die 10 Regeln vollständig spezifiziert sind, ist
  ein exakter Validator dem Lernen einer Anomalie-Klassifikation überlegen
  (deterministisch, 100 % erklärbar, keine False Negatives auf bekannten
  Regeltypen).
- **Greedy-Completion:** einfach und reproduzierbar; Sampling/Beam-Search wären ein
  möglicher nächster Schritt für Diversität.
- **Offen einzutragen:** Konkrete Endmetriken (Val-Loss, Top-k, valid_rate je
  Familie) werden pro Lauf in `outputs/` erzeugt und sind direkt im Dashboard
  sichtbar; sie hängen vom gewählten Preset/Lauf ab.

---

## 11. Bezug zu den Bewertungskriterien

- **Lauffähiges Artefakt:** End-to-End-Pipeline + interaktives Dashboard, keine
  Slideware.
- **Reproduzierbarkeit:** feste Seeds, selbstbeschreibende Checkpoints, ein-Klick-
  Wiederholung; Eval rekonstruiert exakt den Trainings-Split.
- **Ehrliche Evaluation:** Baseline vs. Modell, Sanity-Anker, expliziter OOD-Pass,
  strenge `quality_rate` zusätzlich zur `valid_rate`.
- **Nachvollziehbare Entscheidungen:** dokumentiert in diesem Report und in den
  Code-Kommentaren; der AI-Coach macht Experiment-Begründungen explizit.

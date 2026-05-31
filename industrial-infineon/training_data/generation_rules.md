# Process Sequence Generation Rules

**Hackathon Track: Prozesslogik lernen und benchmarken**

This document defines:
1. The **process grammar** for each product family (MOSFET, IGBT, IC) — what makes a sequence valid.
2. The **variation axes** — what teams are allowed to change when generating new sequences.
3. The **10 forbidden patterns** — explicit process-logic violations used in the held-out eval set.
4. The **shared eval protocol** — metrics, eval-set format, and scoring.

Use this document to:
- Generate your own training data (`generate_sequences.py` implements this grammar).
- Understand what "process logic" means, so your model can be tested on it.
- Validate sequences your model generates or completes.

---

## 1. Step Vocabulary

Steps are organized into functional categories. All step names use the uppercase string forms from the CSV files.

### 1.1 Logistics & Inspection

| Category | Steps |
|---|---|
| **Lot logistics** | RECEIVE WAFER LOT, LOT IDENTIFICATION, LOT RELEASE, FINAL LOT RELEASE, SHIP LOT |
| **Incoming inspection** | INITIAL WAFER INSPECTION, PRE CLEAN INSPECTION, INITIAL WAFER INSPECTION |
| **Measurement — geometry** | MEASURE THICKNESS, MEASURE INITIAL THICKNESS, MEASURE INITIAL GEOMETRY, MEASURE GEOMETRY |
| **Measurement — surface** | MEASURE SURFACE PARTICLES, MEASURE SURFACE DEFECTS, MEASURE BACKSIDE ROUGHNESS |
| **Final inspection** | FINAL CLEAN, FINAL THICKNESS MEASURE, FINAL GEOMETRY CHECK, FINAL PARTICLE INSPECTION, FINAL OXIDE CHECK, FINAL CD INSPECTION |

### 1.2 Cleaning Steps

| Step | Notes |
|---|---|
| PRE CLEAN WAFER / WAFER CLEAN PRE PROCESS | Initial pre-clean, mandatory before first oxidation |
| BACKSIDE CLEAN | Clean wafer backside |
| FRONTSIDE CLEAN | Clean wafer frontside |
| WET CLEAN RCA1 / RCA CLEAN 1 | Alkaline RCA clean (organic residue removal) |
| WET CLEAN RCA2 / RCA CLEAN 2 | Acidic RCA clean (metallic contamination removal) |
| HF DIP | Native oxide strip; hydrogen-terminates surface |
| DRY WAFER / DRY WAFER BACKSIDE | Spin dry after wet processing |
| CLEAN AFTER ETCH | Post-etch clean; mandatory after any etch before next deposition |
| CLEAN AFTER OXIDE ETCH / CLEAN AFTER WINDOW ETCH / CLEAN AFTER FIELD ETCH / CLEAN AFTER VIA ETCH / CLEAN AFTER METAL ETCH / CLEAN AFTER POLY ETCH | Variant post-etch cleans |
| CLEAN PAD OPENING | Post-etch clean for pad window |
| BACKSIDE ETCH CLEAN | Post-backside-etch clean |
| BACKSIDE RINSE | Rinse after backside etch clean |
| FRONTSIDE CLEAN FINAL | Final frontside clean before test |
| BACKSIDE CLEAN FINAL | Final backside clean before test |

> **Rule anchor:** Any deposition step (thermal oxidation, CVD, sputtering, epitaxial deposition) **must** be preceded by a cleaning step within the same processing block. See RULE_DEP_NO_CLEAN (Section 3).

### 1.3 Thermal & Deposition Steps

| Step | Notes |
|---|---|
| THERMAL OXIDATION | Grows gate or pad oxide; requires clean surface |
| GATE OXIDE PREP / GATE OXIDE GROWTH | MOSFET-specific gate oxidation sequence |
| DEPOSIT PAD OXIDE | IC-specific pad oxide |
| ANNEAL OXIDE | Post-deposition anneal |
| DEPOSIT POLYSILICON | Poly gate or resistor deposition |
| POLYSILICON ANNEAL / ANNEAL POLYSILICON | Post-poly anneal |
| DEPOSIT SPACER DIELECTRIC | MOSFET LDD spacer |
| DEPOSIT FIELD OXIDE | IGBT field dielectric |
| DEPOSIT GATE OXIDE OR DIELECTRIC | IGBT gate dielectric |
| DEPOSIT INTERLAYER DIELECTRIC / DEPOSIT INTERLEVEL DIELECTRIC | Pre-CMP ILD |
| DENSIFY DIELECTRIC / DENSIFY OXIDE | ILD densification anneal |
| DEPOSIT BARRIER METAL | Diffusion barrier before via fill |
| DEPOSIT METAL SEED | Seed layer for via fill |
| DEPOSIT METAL 1 / DEPOSIT TOP METAL | Interconnect metal layer |
| DEPOSIT BACKSIDE METAL | Backside contact metallization |
| DEPOSIT TUNGSTEN SEED | IC-specific tungsten via seed |
| DEPOSIT PASSIVATION / DEPOSIT PASSIVATION LAYER | Passivation dielectric |
| DEPOSIT BACKSIDE PROTECTION | IC-specific backside protection |
| EPITAXIAL DEPOSITION | MOSFET epitaxial layer growth |

### 1.4 Lithography Steps (form a required sequential block)

| Step | Notes |
|---|---|
| SPIN COAT PHOTORESIST | Must be first in every litho block |
| SOFT BAKE | Follows SPIN COAT |
| ALIGN MASK LEVEL N | Aligns mask for level N (N = 1, 2, 3, …) |
| EXPOSE LITHO LEVEL N | Exposes pattern for level N |
| POST EXPOSE BAKE | Optional; IC uses it; must follow EXPOSE and precede DEVELOP |
| DEVELOP PHOTORESIST | Develops pattern |
| INSPECT PATTERN LEVEL N / PATTERN INSPECTION LEVEL N / POLY PATTERN INSPECTION / VIA INSPECTION / METAL PATTERN INSPECTION / P BODY WINDOW INSPECTION / FIELD PATTERN INSPECTION / VIA OPENING INSPECTION | Pattern quality check; must follow DEVELOP |
| HARD BAKE | Optional; hardens resist before etch; must follow pattern inspection |

### 1.5 Etch Steps

| Step | Notes |
|---|---|
| OXIDE ETCH / OXIDE ETCH DRY | Etch gate/field/pad oxide |
| POLYSILICON ETCH / POLYSILICON ETCH DRY | Etch poly gate |
| ANISOTROPIC ETCH SPACER | MOSFET LDD spacer etch |
| ETCH SILICON OR OXIDE WINDOW | IGBT body window etch |
| FIELD OXIDE ETCH | IGBT field oxide etch |
| VIA ETCH / VIA ETCH THROUGH DIELECTRIC / DIELECTRIC ETCH VIA | Contact via etch |
| METAL ETCH / METAL ETCH DRY | Pattern metal layer |
| PASSIVATION ETCH PAD OPENING / PASSIVATION ETCH | Open bond pads in passivation |
| ETCH WET BACKSIDE | IC backside wet etch |
| BACKSIDE ETCH CLEAN | Backside etch clean (part of backside block) |

### 1.6 Strip Steps

| Step | Notes |
|---|---|
| STRIP PHOTORESIST / STRIP RESIST / STRIP RESIST LEVEL N | Photoresist removal; always follows etch |
| OXIDE STRIP / HF DIP | Oxide removal (part of pre-process clean) |

### 1.7 Implant & Diffusion Steps

| Step | Notes |
|---|---|
| IMPLANT WELL | MOSFET well implant |
| IMPLANT SOURCE DRAIN / IMPLANT SOURCE REGION | Source/drain dopant implant |
| IMPLANT LDD | MOSFET lightly-doped drain implant |
| IMPLANT P BODY | IGBT p-body implant |
| IMPLANT N BUFFER | IGBT n-buffer layer implant |
| IMPLANT CHANNEL STOP | IGBT channel stop implant |
| IMPLANT DRAIN / CATHODE REGION | IGBT drain/cathode implant |
| IMPLANT N-TYPE | IC n-well/source implant |
| DRIVE IN DIFFUSION | Diffusion anneal after implant |
| RAPID THERMAL ANNEAL | RTA to activate dopants; follows implant blocks |
| LIGHT ANNEAL | MOSFET alternative to full RTA for source/drain |

### 1.8 CMP & Planarization

| Step | Notes |
|---|---|
| CMP DIELECTRIC / CMP INTERLAYER DIELECTRIC | Planarize ILD after deposition |
| CMP METAL / CMP VIA FILL | Planarize metal or via fill |

> **Rule anchor:** A CMP step **must** be preceded by a deposition step in the same block. See RULE_CMP_NO_DEP (Section 3).

### 1.9 Via Fill

| Step | Notes |
|---|---|
| FILL VIA METAL / FILL VIA TUNGSTEN | Via metal fill |

### 1.10 Measurement (in-process)

| Step | Notes |
|---|---|
| MEASURE OXIDE THICKNESS / MEASURE GATE OXIDE THICKNESS | Post-oxidation |
| MEASURE EPITAXY THICKNESS | Post-epitaxy |
| MEASURE RESISTIVITY | Epitaxial layer quality |
| MEASURE FILM THICKNESS / MEASURE DIELECTRIC THICKNESS | Post-deposition |
| MEASURE PLANARITY / MEASURE SURFACE PLANARITY | Post-CMP |
| MEASURE POLY THICKNESS / MEASURE GATE CD / MEASURE GATE OXIDE THICKNESS | Post-poly |
| MEASURE OPENING CD / MEASURE WINDOW CD / MEASURE CD LEVEL N / MEASURE LINE WIDTH | Post-etch |
| MEASURE VIA CD / MEASURE VIA RESISTANCE | Post-via etch/fill |
| MEASURE CONTACT RESISTANCE / MEASURE BACKSIDE CONTACT | Post-metallization |
| MEASURE SHEET RESISTANCE | Post-implant/anneal |
| MEASURE JUNCTION DEPTH / MEASURE JUNCTION PROFILE | Post-implant |
| MEASURE PASSIVATION THICKNESS / MEASURE PASSIVATION QUALITY | Post-passivation |
| MEASURE PAD OPENING | Post-pad etch |
| MEASURE SPACER WIDTH | MOSFET spacer etch |
| MEASURE DEVICE PARAMETER | IGBT mid-process quality gate |
| MEASURE WAFER THICKNESS | Post-backside grind |
| MEASURE BACKSIDE ROUGHNESS | Post-backside grind (IC) |
| MEASURE OXIDE QUALITY | Post-dielectric anneal |
| MEASURE BACKSIDE CONTACT | Post-backside metallization |

### 1.11 Electrical Tests

| Step | Notes |
|---|---|
| PARAMETRIC TEST / ELECTRICAL PARAMETRIC TEST / FINAL ELECTRICAL TEST PREP | Parametric characterization |
| THRESHOLD VOLTAGE TEST | MOSFET-specific |
| LEAKAGE TEST | All families |
| BREAKDOWN VOLTAGE TEST | IGBT-specific |
| SWITCHING TEST | All families |
| WAFER SORT TEST | Mandatory; must precede SHIP LOT |
| YIELD ANALYSIS | Post sort analysis |

### 1.12 Substrate Preparation (Family-Specific)

| Step | Notes |
|---|---|
| SUBSTRATE CHECK | MOSFET; verifies substrate before epitaxy |
| EPITAXY PREP / EPITAXIAL LAYER PREP | Pre-epitaxy preparation |
| EPITAXIAL DEPOSITION | MOSFET epitaxial layer growth |
| MEASURE EPITAXY THICKNESS | Post-epitaxy measurement |
| EPITAXY ANNEAL | MOSFET post-epitaxy anneal |
| WAFER SURFACE CLEAN | MOSFET; clean after epitaxy |
| EPITAXIAL WAFER CHECK / EPITAXIAL REWORK CHECK | IGBT; verify epitaxial substrate |
| GRINDING WAFER BACKSIDE | IC; early backside thinning |
| ETCH WET BACKSIDE | IC; backside wet etch after grind |
| RINSE WET WAFER_EDGE | IC; edge rinse |
| DRY WAFER BACKSIDE | IC; dry after backside grind |
| SURFACE PREP FOR DEPOSITION | IC; surface prep before first deposition |
| PRE ANNEAL CHECK | Pre-anneal quality verification |
| GATE OXIDE PREP | MOSFET; gate oxide preparation |
| BACKSIDE THINNING CHECK | IC; post-backside-thin check |
| PACKAGE PREPARATION | IC-specific final step before SHIP LOT |

---

## 2. Process Grammar

### 2.1 Notation

```
[STEP]         = optional step (present or absent with some probability)
STEP_A | STEP_B = choose one (both are valid alternates for the same role)
<BLOCK_NAME>   = reference to a named block defined below
(BLOCK){N}     = repeat BLOCK exactly N times
(BLOCK){3..6}  = repeat BLOCK between 3 and 6 times (inclusive)
→              = must appear in this order
```

### 2.2 Shared Backbone (all families)

```
<PREFIX>
→ <INITIAL_MEASUREMENTS>
→ <PRE_PROCESS_CLEAN>
→ <FAMILY_SPECIFIC_PREP>          ← content differs per family
→ <FIRST_OXIDATION>
→ <PROCESS_CYCLES>{3..6}          ← repeated litho–etch–implant cycles
→ <ILD_BLOCK>
→ <VIA_BLOCK>
→ <METAL_BLOCK>
→ <PASSIVATION_BLOCK>
→ <BACKSIDE_BLOCK>
→ <FINAL_INSPECTION>
→ <TEST_SUITE>
→ <SUFFIX>
```

### 2.3 Block Definitions

#### PREFIX (mandatory, fixed order)
```
RECEIVE WAFER LOT
→ LOT IDENTIFICATION
→ INITIAL WAFER INSPECTION | PRE CLEAN INSPECTION
```

#### INITIAL_MEASUREMENTS (optional individually)
```
[MEASURE THICKNESS | MEASURE INITIAL THICKNESS | MEASURE INITIAL GEOMETRY]
→ [MEASURE SURFACE PARTICLES | MEASURE SURFACE DEFECTS]
```

#### PRE_PROCESS_CLEAN (mandatory; order within block is fixed)
```
PRE CLEAN WAFER | WAFER CLEAN PRE PROCESS
→ [BACKSIDE CLEAN]
→ [FRONTSIDE CLEAN]
→ WET CLEAN RCA1 | RCA CLEAN 1
→ WET CLEAN RCA2 | RCA CLEAN 2
→ HF DIP
→ [DRY WAFER | DRY WAFER BACKSIDE]
```

#### FAMILY_SPECIFIC_PREP

**MOSFET:**
```
SUBSTRATE CHECK
→ EPITAXY PREP
→ EPITAXIAL DEPOSITION
→ MEASURE EPITAXY THICKNESS
→ MEASURE RESISTIVITY
→ EPITAXY ANNEAL
→ WAFER SURFACE CLEAN
```

**IGBT:**
```
EPITAXIAL WAFER CHECK
→ MEASURE EPITAXY THICKNESS
→ MEASURE RESISTIVITY
→ [EPITAXIAL REWORK CHECK]
→ EPITAXIAL LAYER PREP
```

**IC:**
```
WAFER CLEAN PRE-GRIND
→ GRINDING WAFER BACKSIDE
→ MEASURE GEOMETRY | MEASURE INITIAL GEOMETRY
→ ETCH WET BACKSIDE
→ RINSE WET WAFER_EDGE
→ DRY WAFER BACKSIDE
→ BACKSIDE CLEAN
→ MEASURE BACKSIDE ROUGHNESS
```

#### FIRST_OXIDATION
```
THERMAL OXIDATION
→ [GATE OXIDE PREP]                  ← MOSFET only
→ [GATE OXIDE GROWTH]                ← MOSFET only
→ [SURFACE PREP FOR DEPOSITION]      ← IC only
→ [DEPOSIT PAD OXIDE]                ← IC only
→ [ANNEAL OXIDE]
→ MEASURE OXIDE THICKNESS | MEASURE FILM THICKNESS
→ [WET CLEAN RCA1 | RCA CLEAN 1]     ← IC only: post-oxidation clean
→ [WET CLEAN RCA2 | RCA CLEAN 2]
→ [HF DIP]
→ [OXIDE STRIP]
```

#### LITHO_CYCLE_TEMPLATE (used inside PROCESS_CYCLES)
```
SPIN COAT PHOTORESIST
→ SOFT BAKE
→ ALIGN MASK LEVEL {N}
→ EXPOSE LITHO LEVEL {N}
→ [POST EXPOSE BAKE]                 ← IC-style; optional for other families
→ DEVELOP PHOTORESIST
→ <PATTERN_INSPECTION>
→ [HARD BAKE]                        ← optional; hardens resist before etch
```

Where `<PATTERN_INSPECTION>` is one of:
```
INSPECT PATTERN LEVEL N | PATTERN INSPECTION LEVEL N | POLY PATTERN INSPECTION
| VIA INSPECTION | METAL PATTERN INSPECTION | P BODY WINDOW INSPECTION
| FIELD PATTERN INSPECTION | VIA OPENING INSPECTION
```

#### PROCESS_CYCLES (repeating units, varies per family)

Each cycle follows the pattern:
```
[DEPOSIT or GROW something]          ← e.g., THERMAL OXIDATION, DEPOSIT POLYSILICON
→ [MEASURE after deposit]
→ [PRE_CLEAN before litho]           ← optional intermediate clean
→ <LITHO_CYCLE_TEMPLATE>
→ <ETCH_STEP>                        ← appropriate etch for what was deposited
→ STRIP PHOTORESIST | STRIP RESIST
→ CLEAN AFTER ETCH                   ← mandatory post-etch clean
→ [MEASURE_CD]                       ← measure opening/line width
→ [IMPLANT_BLOCK]                    ← if this cycle includes doping
→ [ANNEAL_BLOCK]                     ← anneal after implant
→ [MEASURE after implant/anneal]
```

**MOSFET cycle overview (4 litho levels):**
```
Cycle 1: Well implant cycle    → OXIDE ETCH → IMPLANT WELL → DRIVE IN DIFFUSION → RTA
Cycle 2: Poly gate cycle       → POLYSILICON ETCH → IMPLANT SOURCE DRAIN → LIGHT ANNEAL
         (includes spacer sub-block: DEPOSIT SPACER DIELECTRIC → ANISOTROPIC ETCH SPACER → IMPLANT LDD → RTA)
Cycle 3: Via cycle             → VIA ETCH → BARRIER + FILL VIA METAL + CMP
Cycle 4: Metal cycle           → METAL ETCH
```

**IGBT cycle overview (6 litho levels):**
```
Cycle 1: P body implant cycle   → OXIDE ETCH DRY → IMPLANT P BODY → DRIVE IN DIFFUSION → RTA
Cycle 2: N buffer implant cycle → WINDOW ETCH → IMPLANT N BUFFER → RTA
Cycle 3: Field oxide cycle      → FIELD OXIDE ETCH → IMPLANT SOURCE/DRAIN/CATHODE REGION → RTA
Cycle 4: Poly gate cycle        → POLYSILICON ETCH DRY → IMPLANT CHANNEL STOP → RTA
Cycle 5: Via cycle              → VIA ETCH → BARRIER + FILL VIA METAL + CMP
Cycle 6: Metal cycle            → METAL ETCH DRY
```

**IC cycle overview (4 litho levels):**
```
Cycle 1: Pad oxide / STI cycle  → OXIDE ETCH DRY (with POST EXPOSE BAKE + HARD BAKE)
Cycle 2: Poly gate cycle        → POLYSILICON ETCH DRY → IMPLANT N-TYPE → RTA
Cycle 3: Via cycle              → VIA ETCH → BARRIER + TUNGSTEN SEED + FILL VIA TUNGSTEN + CMP
Cycle 4: Metal cycle            → METAL ETCH DRY
```

#### ILD_BLOCK (between last implant/poly cycle and via cycle)
```
DEPOSIT INTERLAYER DIELECTRIC | DEPOSIT INTERLEVEL DIELECTRIC
→ DENSIFY DIELECTRIC | DENSIFY OXIDE
→ MEASURE FILM THICKNESS | MEASURE DIELECTRIC THICKNESS
→ CMP DIELECTRIC | CMP INTERLAYER DIELECTRIC
→ MEASURE PLANARITY | MEASURE SURFACE PLANARITY
```

#### VIA_BLOCK (after via litho cycle)
```
DEPOSIT BARRIER METAL
→ DEPOSIT METAL SEED | DEPOSIT TUNGSTEN SEED
→ FILL VIA METAL | FILL VIA TUNGSTEN
→ CMP METAL | CMP VIA FILL
→ MEASURE CONTACT RESISTANCE | MEASURE VIA RESISTANCE
```

#### METAL_BLOCK (after metal litho cycle)
```
DEPOSIT METAL 1 | DEPOSIT TOP METAL
→ ANNEAL METAL 1 | ANNEAL METAL
→ [MEASURE METAL THICKNESS]
```

#### PASSIVATION_BLOCK
```
DEPOSIT PASSIVATION | DEPOSIT PASSIVATION LAYER
→ CURE PASSIVATION
→ MEASURE PASSIVATION THICKNESS | MEASURE PASSIVATION QUALITY
→ OPEN PAD WINDOW | OPEN BOND PAD WINDOW
→ PAD WINDOW LITHO | OPEN PAD WINDOW LITHO
→ DEVELOP PHOTORESIST | DEVELOP PAD WINDOW
→ PASSIVATION ETCH PAD OPENING | PASSIVATION ETCH
→ STRIP RESIST
→ CLEAN PAD OPENING
→ MEASURE PAD OPENING
```

#### BACKSIDE_BLOCK
```
[BACKSIDE THINNING CHECK]           ← IC only
→ BACKSIDE CLEAN | BACKSIDE CLEAN FINAL
→ [DEPOSIT BACKSIDE PROTECTION]     ← IC only
→ BACKSIDE GRIND                    ← MOSFET/IGBT; happens here; IC did it early
→ MEASURE THICKNESS | MEASURE WAFER THICKNESS
→ BACKSIDE ETCH CLEAN
→ BACKSIDE RINSE
→ BACKSIDE DRY
→ BACKSIDE METALLIZATION PREP
→ DEPOSIT BACKSIDE METAL
→ BACKSIDE ANNEAL
→ MEASURE BACKSIDE CONTACT
```

> **Note for IC:** The physical backside grind is done early in FAMILY_SPECIFIC_PREP. The BACKSIDE_BLOCK for IC covers final backside clean, protection, and anneal only.

#### FINAL_INSPECTION
```
FINAL CLEAN
→ FINAL THICKNESS MEASURE
→ FINAL GEOMETRY CHECK
→ [FINAL OXIDE CHECK]
→ [FINAL CD INSPECTION]
→ FINAL PARTICLE INSPECTION
→ [FRONTSIDE CLEAN FINAL]
→ [FINAL ELECTRICAL TEST PREP]
```

#### TEST_SUITE (mandatory steps; order matters)
```
PARAMETRIC TEST | ELECTRICAL PARAMETRIC TEST
→ LEAKAGE TEST
→ THRESHOLD VOLTAGE TEST | BREAKDOWN VOLTAGE TEST   ← family-specific
→ SWITCHING TEST
→ WAFER SORT TEST
→ YIELD ANALYSIS
```

#### SUFFIX (mandatory, fixed order)
```
LOT RELEASE | FINAL LOT RELEASE
→ [PACKAGE PREPARATION]             ← IC only
→ SHIP LOT
```

---

## 3. Forbidden Patterns

These 10 rules define **process-logic violations**. A sequence that breaks any of these rules is invalid, regardless of whether individual steps appear in the vocabulary.

They are used in the held-out evaluation set (`eval_set_forbidden.csv`) to test whether a model has learned process logic or only step-order statistics.

The checker function is implemented in `generate_sequences.py` as `validate_sequence(steps) → list[Violation]`.

---

### RULE_DEP_NO_CLEAN
**Any deposition step must be preceded by a cleaning step within the same block (within N=12 steps before it).**

Deposition steps (triggers): `THERMAL OXIDATION`, `GATE OXIDE GROWTH`, `DEPOSIT PAD OXIDE`, `EPITAXIAL DEPOSITION`, `DEPOSIT POLYSILICON`, `DEPOSIT SPACER DIELECTRIC`, `DEPOSIT FIELD OXIDE`, `DEPOSIT GATE OXIDE OR DIELECTRIC`, `DEPOSIT INTERLAYER DIELECTRIC`, `DEPOSIT INTERLEVEL DIELECTRIC`, `DEPOSIT BARRIER METAL`, `DEPOSIT METAL SEED`, `DEPOSIT METAL 1`, `DEPOSIT TOP METAL`, `DEPOSIT BACKSIDE METAL`, `DEPOSIT TUNGSTEN SEED`, `DEPOSIT PASSIVATION`, `DEPOSIT PASSIVATION LAYER`, `DEPOSIT BACKSIDE PROTECTION`

Required preceding (any one of):
- *Wet/chemical cleans:* `PRE CLEAN WAFER`, `WAFER CLEAN PRE PROCESS`, `WAFER SURFACE CLEAN`, `RCA CLEAN 1`, `RCA CLEAN 2`, `WET CLEAN RCA1`, `WET CLEAN RCA2`, `HF DIP`, `OXIDE STRIP`, `SURFACE PREP FOR DEPOSITION`, `FRONTSIDE CLEAN`, `BACKSIDE CLEAN`, `FRONTSIDE CLEAN FINAL`, `BACKSIDE CLEAN FINAL`, `WAFER CLEAN PRE-GRIND`, `DRY WAFER`, `DRY WAFER BACKSIDE`
- *Post-etch cleans:* `CLEAN AFTER ETCH`, `CLEAN AFTER OXIDE ETCH`, `CLEAN AFTER POLY ETCH`, `CLEAN AFTER VIA ETCH`, `CLEAN AFTER METAL ETCH`, `CLEAN AFTER WINDOW ETCH`, `CLEAN AFTER FIELD ETCH`, `CLEAN PAD OPENING`, `BACKSIDE ETCH CLEAN`, `BACKSIDE RINSE`
- *Thermal steps (create a clean/passivated surface):* `THERMAL OXIDATION`, `GATE OXIDE PREP`, `RAPID THERMAL ANNEAL`, `EPITAXY ANNEAL`, `ANNEAL OXIDE`

**Violation example:** `... → ETCH VIA → DEPOSIT BARRIER METAL` (missing `CLEAN AFTER VIA ETCH` between them)

---

### RULE_METAL_ETCH_NO_LITHO
**A metal etch step must be preceded by a complete litho sequence (EXPOSE LITHO + DEVELOP PHOTORESIST) within the current metal block (within N=15 steps before it).**

Trigger steps: `METAL ETCH`, `METAL ETCH DRY`

Required preceding (both must appear): `EXPOSE LITHO LEVEL {N}` and `DEVELOP PHOTORESIST` (or `DEVELOP PAD WINDOW`)

**Violation example:** `... → DEPOSIT METAL 1 → ANNEAL METAL 1 → METAL ETCH → ...` (no litho between metal deposition and metal etch)

---

### RULE_ETCH_NO_MASK
**Any etch step (except backside etch clean) must be preceded by a lithography develop step within N=12 steps.**

Trigger steps: `OXIDE ETCH`, `OXIDE ETCH DRY`, `POLYSILICON ETCH`, `POLYSILICON ETCH DRY`, `ETCH SILICON OR OXIDE WINDOW`, `FIELD OXIDE ETCH`, `VIA ETCH`, `VIA ETCH THROUGH DIELECTRIC`, `DIELECTRIC ETCH VIA`, `METAL ETCH`, `METAL ETCH DRY`, `PASSIVATION ETCH PAD OPENING`, `PASSIVATION ETCH`

> **Note:** `ANISOTROPIC ETCH SPACER` is deliberately excluded — it is a blanket (non-patterned) etch and does not require a prior lithography step.

Required preceding: `DEVELOP PHOTORESIST` or `DEVELOP PAD WINDOW`

**Violation example:** `... → MEASURE OXIDE THICKNESS → OXIDE ETCH → ...` (no litho sequence before the etch)

---

### RULE_LITHO_LEVEL_SKIP
**Litho levels must be sequential: ALIGN MASK LEVEL N+1 must not appear before ALIGN MASK LEVEL N is complete (i.e., followed by EXPOSE, DEVELOP, ETCH, STRIP).**

Rule: If ALIGN MASK LEVEL 3 appears in the sequence, ALIGN MASK LEVEL 2 must appear earlier and have a corresponding DEVELOP step before ALIGN MASK LEVEL 3.

**Violation example:** Sequence contains `ALIGN MASK LEVEL 3` before `ALIGN MASK LEVEL 2`

---

### RULE_IMPLANT_NO_MASK
**Any implant step must be preceded by an oxide etch or litho develop step within N=15 steps (the implant region must be opened before implanting).**

Trigger steps: `IMPLANT WELL`, `IMPLANT SOURCE DRAIN`, `IMPLANT SOURCE REGION`, `IMPLANT LDD`, `IMPLANT P BODY`, `IMPLANT N BUFFER`, `IMPLANT CHANNEL STOP`, `IMPLANT DRAIN / CATHODE REGION`, `IMPLANT N-TYPE`

Required preceding (any one): `OXIDE ETCH`, `OXIDE ETCH DRY`, `ETCH SILICON OR OXIDE WINDOW`, `DEVELOP PHOTORESIST`

**Violation example:** `... → MEASURE JUNCTION DEPTH → IMPLANT LDD → ...` (implanting without an open oxide window)

---

### RULE_CMP_NO_DEP
**A CMP step must be preceded by a deposition step within the same block (within N=6 steps before it).**

Trigger steps: `CMP DIELECTRIC`, `CMP INTERLAYER DIELECTRIC`, `CMP METAL`, `CMP VIA FILL`

Required preceding: any deposition step from RULE_DEP_NO_CLEAN list, or `FILL VIA METAL`, `FILL VIA TUNGSTEN`

**Violation example:** `... → MEASURE PLANARITY → CMP METAL → ...` (CMP without preceding fill/deposition)

---

### RULE_PAD_OPEN_BEFORE_DEP
**The pad window opening sequence (PAD WINDOW LITHO / OPEN PAD WINDOW) must appear after DEPOSIT PASSIVATION and CURE PASSIVATION.**

Rule: `OPEN PAD WINDOW` or `OPEN BOND PAD WINDOW` or `PAD WINDOW LITHO` must have both `DEPOSIT PASSIVATION` and `CURE PASSIVATION` appearing earlier in the sequence.

**Violation example:** `... → OPEN PAD WINDOW → DEPOSIT PASSIVATION → CURE PASSIVATION → ...`

---

### RULE_TEST_BEFORE_PASSIVATION
**Electrical test steps must appear after CURE PASSIVATION.**

Trigger steps: `PARAMETRIC TEST`, `ELECTRICAL PARAMETRIC TEST`, `THRESHOLD VOLTAGE TEST`, `BREAKDOWN VOLTAGE TEST`, `LEAKAGE TEST`, `SWITCHING TEST`

Rule: All of these must appear after `CURE PASSIVATION` in the sequence.

**Violation example:** `... → DEPOSIT PASSIVATION → PARAMETRIC TEST → CURE PASSIVATION → ...`

---

### RULE_SHIP_BEFORE_TEST
**SHIP LOT must appear after WAFER SORT TEST.**

Rule: `SHIP LOT` must not appear at any position before `WAFER SORT TEST`.

**Violation example:** `... → YIELD ANALYSIS → SHIP LOT → WAFER SORT TEST → ...`

---

### RULE_BACKSIDE_BEFORE_PASSIVATION
**DEPOSIT BACKSIDE METAL must appear after CURE PASSIVATION (frontside must be protected before backside metallization).**

Rule: `DEPOSIT BACKSIDE METAL` must appear after `CURE PASSIVATION` in the sequence.

**Violation example:** `... → DEPOSIT BACKSIDE METAL → BACKSIDE ANNEAL → DEPOSIT PASSIVATION → CURE PASSIVATION → ...`

---

## 4. Variation Axes

When generating new sequences, the following dimensions can be varied while keeping the sequence valid:

| Axis | Options | Notes |
|---|---|---|
| **Number of litho cycles** | 3, 4, 5, or 6 | Must be ≥ 3; each cycle includes at least the litho template + etch |
| **POST EXPOSE BAKE** | present / absent | Can be included in any litho cycle for any family |
| **HARD BAKE** | present / absent | Can be included after pattern inspection in any litho cycle |
| **Intermediate clean cycle** | present / absent | An extra RCA1 + RCA2 + HF DIP block may be inserted between major process blocks |
| **Extra measurements** | per step: present / absent | Any individual measurement step can be omitted or added at valid positions |
| **DRY WAFER after HF DIP** | present / absent | Optional spin dry |
| **EPITAXIAL REWORK CHECK** | present / absent | IGBT only |
| **PRE ANNEAL CHECK** | present / absent | Any family; appears before RAPID THERMAL ANNEAL |
| **Second metal layer** | present / absent | Additional DEPOSIT METAL + litho + ETCH cycle |
| **CMP METAL after via fill** | present / absent | Via CMP is common but not always required |
| **Step name synonyms** | interchangeable | E.g., STRIP PHOTORESIST vs STRIP RESIST; RCA CLEAN 1 vs WET CLEAN RCA1 |

**Fixed (not variable):**
- Block order (prefix → clean → prep → cycles → passivation → backside → test → suffix)
- The 10 ordering constraints (Section 3)
- Family-specific mandatory blocks (MOSFET epitaxy, IGBT dual-implant, IC backside grind)
- TEST_SUITE internal order

---

## 5. Shared Evaluation Protocol

### 5.1 Eval Input Files (distributed by organizers)

**`eval_input_valid.csv`** — partial sequences for next-step prediction and sequence completion:
```
EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
```
- `PARTIAL_SEQUENCE`: pipe-separated steps up to the cut point (e.g., `"RECEIVE WAFER LOT|LOT IDENTIFICATION|..."`)
- `COMPLETION_FRACTION`: 0.6 or 0.8 — indicates how far through the full sequence the partial cuts off
- 600 rows total: 100 sequences × 3 families × 2 cut points

**`eval_input_anomaly.csv`** — unlabeled sequences for anomaly detection:
```
EXAMPLE_ID, FAMILY, SEQUENCE
```
- `SEQUENCE`: pipe-separated full sequence — some are valid, some contain a process-rule violation
- 987 rows total: shuffled mix of valid and invalid sequences, no labels

### 5.2 Core Metrics

| Task | Metric(s) |
|---|---|
| **Next-step prediction** | Top-1 Accuracy, Top-3 Accuracy, Top-5 Accuracy, MRR |
| **Sequence completion** | Exact Match Rate, Normalized Edit Distance (lower = better), Token Accuracy, Block-level Accuracy |
| **Anomaly detection** | Binary Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Rule Attribution Accuracy (among detected violations) |
| **Generalization (reporting)** | Performance-Drop ID -> OOD (for each primary metric) |

### 5.3 Submission Format

**Task 1 — Next-step prediction** (one row per `eval_input_valid.csv` example):
```
EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
valid_0001, MEASURE THICKNESS, INSPECT WAFER, CLEAN SURFACE, MEASURE GEOMETRY, HF DIP
```

**Task 2 — Sequence completion** (one row per `eval_input_valid.csv` example):
```
EXAMPLE_ID, PREDICTED_SEQUENCE
valid_0001, MEASURE THICKNESS|CLEAN SURFACE|DEPOSIT GATE OXIDE|PATTERN GATE|...
```
Predict only the steps **after** the cut point — do not repeat the partial sequence given in `PARTIAL_SEQUENCE`.

**Task 3 — Anomaly detection** (one row per `eval_input_anomaly.csv` example):
```
EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
valid_0001,  1, 0.95,
forbidden_0042, 0, 0.08, RULE_DEP_NO_CLEAN
```
- `IS_VALID`: **1** = valid, **0** = invalid (required)
- `SCORE`: probability that the sequence is valid, range [0.0, 1.0] (optional; used for AUC)
- `PREDICTED_RULE`: rule ID if invalid (optional; used for Rule Attribution Accuracy)

Submit all three prediction files to the organizers for scoring.

---

## 6. Using the Generation Script

```bash
# Generate 500 MOSFET variants
python generate_sequences.py --family mosfet --count 500 --output MOSFET_variants.csv --seed 42

# Generate IGBT variants (let script decide count from combinatorics)
python generate_sequences.py --family igbt --output IGBT_variants.csv --seed 42

# Validate an existing sequence file
python generate_sequences.py --validate mysequences.csv --family mosfet

# Print combinatoric estimate without generating
python generate_sequences.py --family ic --estimate-only
```

Output CSV format (matches existing `synthetic_*.csv` files, extended with SEQUENCE_ID):
```
SEQUENCE_ID, STEP
seq_001, RECEIVE WAFER LOT
seq_001, LOT IDENTIFICATION
...
seq_002, RECEIVE WAFER LOT
...
```

---

## 7. Reference Sequences

The original single-sequence reference files (do not modify):

| File | Family | Steps |
|---|---|---|
| `synthetic_mosfet.csv` | MOSFET | 126 |
| `syntheticIGBT.csv` | IGBT | 151 |
| `syntheticIC.csv` | IC | 107 |

These are the canonical baseline sequences. Generated variants should have sequences of comparable length (±20%) and the same overall block structure.

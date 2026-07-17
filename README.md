# Predicting Early Discontinuation — CDISC Pilot 01

This solution predicts whether a **randomized** patient will discontinue a clinical
trial before completing it, from source data organized to **CDISC SDTM** standards.
It uses **baseline-only** information: one command validates the input, engineers
leakage-safe features, splits 70/30, trains several models across **binary and
time-to-event** outcomes, and writes a combined report.

> **Design priority: rigor over a headline metric.** This dataset makes it
> trivially easy to "predict" dropout using information that exists only *because*
> a patient dropped out. Preventing that leakage is the core of the work — so the
> honest baseline-only accuracy is modest, and that is the point.

---

## Key results (recommended run — clinical mode, seed 52)

| | |
|---|---|
| **Cohort** | 306 subjects → 254 randomized → **247 modeled** (137 discontinued / 110 completed) |
| **Split** | 172 development / 75 holdout (stratified on the outcome) |
| **Binary champion** | **Gradient Boosting** — holdout ROC-AUC **0.70**, PR-AUC 0.72 (Logistic Regression 0.66) |
| **Time-to-event** | Active-arm dropout hazard **≈ 2.1–2.2× placebo** (Cox, p < 0.001, c-index 0.64, PH holds); log-rank p = 4.8×10⁻⁷ |
| **Competing risks** | **Adverse events are the leading cause** of dropout (38% cumulative incidence) |

**Key insight.** Treatment arm dominates every model. The active xanomeline arms
discontinue at roughly **twice** the placebo rate (Xan_Lo 69.5%, Xan_Hi 66.7% vs.
Pbo 31.0%), they leave **sooner** (median 77–106 days vs. not reached for placebo),
and they leave mainly because of **adverse events**. Five independent models —
binary, survival and competing-risks — converge on the same conclusion, so the
models are recovering known clinical reality rather than an artifact. Practically,
active-arm patients are the group where proactive tolerability management would
retain the most participants, and they can be flagged at randomization.

---

## Repository structure

```
run_pipeline.py                   # single entry point: validate → 5 stages → report
sdtm_reader.py                    # 1  SDTM (.xpt) → clean CSVs   (+ define.xml gate)
build_features_cdiscpilot01.py    # 2  baseline features + 70/30 stratified split
model_cdiscpilot01.py             # 3  binary models: train, tune, evaluate, explain
survival_cdiscpilot01.py          # 4  Kaplan–Meier, log-rank, Cox+PH, competing risks
                                  # 5  combined Word report (inside run_pipeline.py)
requirements.txt
PROMPT.md                         # conversational "AI function" prompt (front-end)
README.md                         # this file
cdiscpilot01_out_v2/              # (generated) csv/ features/ model/ survival/ + report.docx
```

---

## Quick start

```bash
pip install -r requirements.txt

# Minimum settings — runs end to end on defaults (downloads the data + define.xml):
python run_pipeline.py --study "CDISC Pilot 01" --fetch-reference --report both

# Recommended settings — explicit control of every key parameter;
# reproduces the results in this README:
python run_pipeline.py --study "CDISC Pilot 01" --fetch-reference --outdir cdiscpilot01_out_v2 --target-mode clinical --landmark-day 1 --death event --report both --seed 52

# Your own CDISC-Pilot-01-format study:
python run_pipeline.py --study "My Study" --input-dir ./my_sdtm --report both
```

> **Note on `--seed`.** The built-in default is `42`; the recommended command uses
> `52`, which reproduces the numbers reported here. The seed only affects the
> split-dependent (binary) results — the survival analyses use all 247 patients and
> are seed-independent.

**Input contract.** The `--input-dir` folder must contain `define.xml` **and**
every dataset it declares (`.xpt`/`.csv`). Otherwise the run stops immediately
with: `User's input does not meet CDISC SDTM requirements that results in
termination of further analysis`.

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--study` | *(required)* | Label for the run |
| `--input-dir` / `--fetch-reference` | *(one required)* | Local SDTM folder / download the reference study |
| `--target-mode` | `clinical` | `clinical` drops administrative sponsor-terminations; `any` keeps all non-completions |
| `--landmark-day` | `1` | Baseline window (study day ≤ N) — the leakage guard |
| `--death` | `event` | Treat DEATH as an event, or `exclude` |
| `--report` | `both` | `files`, `word`, or `both` |
| `--outdir` / `--seed` | `cdiscpilot01_out_v2` / `42` | Output folder / reproducibility |

---

## Methodology (the reviewer's four questions)

| Question | Answer |
|---|---|
| Baseline-only or time-varying **features**? | **Baseline-only** for every model — enforced by a randomization landmark (day ≤ 1) |
| Cutoff relative to completion? | **None, and none needed** — a completion-relative cutoff is only required with time-varying features; the landmark is *start*-relative, not completion-relative |
| Binary or time-to-event **outcome**? | **Both.** Binary (Logistic Regression, Gradient Boosting) is primary; time-to-event (Kaplan–Meier, Cox, competing risks) adds *when*, *how fast*, and *why* |
| Explainable? | **Yes** — LR odds ratios, permutation importance, Cox hazard ratios, cause-specific incidence (+ SHAP if installed) |

> "Time-to-event" describes the **outcome**, not the features. This project uses
> time-varying features nowhere — including in the survival models.

**Leakage policy.** Features use only information known at randomization:
- **Used:** DM demographics + *planned* arm; MH history; baseline VS/LB/QS (ADAS-Cog `ACTOT`, NPI-X `NPTOT`).
- **Excluded from features:** AE (the dropout mechanism), EX/SV/CM/SE, SUPP\*, *actual* arm, and end-of-participation dates — all only knowable during/after the trial. DS is used **solely to define the outcome** (and event timing), never as a predictor.
- Imputation and scaling are fit on **development data only**, then applied to the holdout; the holdout is scored exactly once, after model selection.

Every decision is recorded to `features/feature_manifest.json`.

---

## Models at a glance

| Model | Dependent variable | Predictors | Objective |
|---|---|---|---|
| Logistic Regression | Binary (discontinued 1 / completed 0) | 66 baseline features | Interpretable risk score — *who* |
| Gradient Boosting (champion) | Binary | 66 baseline features | Non-linear risk score — *who* |
| Kaplan–Meier (+ log-rank) | Survival curve S(t) | Arm (grouping) | *When* dropout happens, by arm |
| Cox proportional hazards | Hazard (instantaneous risk) | Arm, AGE, SEX, ADAS-Cog, NPI-X (baseline) | *How much* each factor changes risk |
| Competing-risks (Aalen–Johansen) | Cause-specific cumulative incidence | Cause grouping | *Why* patients drop out |

---

## Outputs (`cdiscpilot01_out_v2/`)

- `csv/` — one clean CSV per SDTM domain; `subjects.csv`; `patient_long.csv`
- `features/` — `model_matrix.csv`, `dev.csv`, `holdout.csv`, `feature_manifest.json`
- `model/` — `metrics.json`, `evaluation_curves.png`, importances, predictions, saved models
- `survival/` — `km_by_arm.png`, `cif_by_cause.png`, `cox_summary.csv`, `cox_ph_check.txt`, `survival_report.json`
- `<study>_report.docx` — combined Word report (binary + survival), with `--report word`/`both`

---

## Two ways to run it

**a) Directly** — run `run_pipeline.py` in any code editor or terminal, using the
parameters above.

**b) As an AI function** — paste `PROMPT.md` into any code-capable AI assistant
(GitHub Copilot Chat, a Copilot Studio agent, or an OpenAI Custom GPT with Code
Interpreter). The prompt turns the assistant into a user-friendly front-end: it
collects the run parameters, builds and executes the `run_pipeline.py` command, and
reports the output back in the chat — using only real numbers from the run. The
Python pipeline stays the fixed, validated backend; the assistant is only the
interface.

> Tested in GitHub Copilot Chat in VS Code, where local files and network access are
> available so `--fetch-reference` works. An OpenAI Custom GPT sandbox has no
> internet, so upload the SDTM files and use `--input-dir` instead.

---

## Environment

Python 3.11 (works on 3.9+). **Required:** `pandas`, `numpy`, `scikit-learn`,
`scipy`, `matplotlib`, `joblib`, `python-docx`, `lifelines` (Cox + PH test).
**Optional:** `shap` (per-patient attribution), `pyreadstat` (faster XPORT).

## Data source

CDISC SDTM Pilot 01, published by PHUSE:
<https://github.com/phuse-org/phuse-scripts/tree/master/data/sdtm/cdiscpilot01>
(`--fetch-reference` downloads it automatically). Any study following the same SDTM
structure can be analyzed with `--input-dir`.

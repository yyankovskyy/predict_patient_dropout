#!/usr/bin/env python3
"""
build_features_cdiscpilot01.py   (v2 -- hardened)
=================================================
Step 2 of the discontinuation-prediction task. Consumes the tidy CSVs written by
`read_cdiscpilot01_sdtm.py` and produces a leakage-safe, subject-level modeling
matrix plus a 70/30 stratified development/holdout split, and records every
methodology decision as an artifact.

==============================================================================
METHODOLOGY (answers the questions posed in the task)
==============================================================================
* Baseline-only vs. time-varying:
    BASELINE-ONLY. Every feature is knowable at randomization. No
    post-randomization / time-varying covariate enters the matrix.

* Cutoff relative to completion (only relevant if time-varying were used):
    Not applicable, because we use no post-baseline data. Instead we enforce a
    RANDOMIZATION LANDMARK: findings (VS/LB/QS) are taken only from records with
    study day <= --landmark-day (default 1). This makes a completion-relative
    cutoff unnecessary -- there is simply nothing downstream of baseline to leak.

* Binary vs. time-to-event outcome:
    PRIMARY = BINARY (discontinued=1 vs completed=0). We ALSO emit a
    time-to-event outcome (TIME_DAYS, EVENT) computed from the disposition date,
    so a Cox / survival sensitivity analysis (with completion as censoring) can
    be run without re-deriving anything. Outcome timing legitimately comes from
    DS; it is used only as a label, never as a feature.

* Explainability:
    The matrix is built for intrinsically interpretable models (logistic
    regression -> odds ratios) and for post-hoc attribution (SHAP on a gradient
    boosting model -> per-subject and global feature attribution). Categorical
    encodings are kept human-readable to support this.

==============================================================================
LEAKAGE POLICY  (the crux of the task)
==============================================================================
The outcome is "did the patient discontinue?". Anything recorded because of, or
after, that event trivially predicts it. Features are therefore restricted to
information knowable at randomization:

  USED (baseline / pre-randomization only):
    DM   - demographics + PLANNED ARM (assigned at randomization)         [safe]
    MH   - medical history = pre-existing conditions (landmark-guarded)   [safe]
    VS   - vital signs,   study day <= landmark (baseline value)          [safe]
    LB   - labs,          study day <= landmark (baseline value)          [safe]
    QS   - ACTOT/CIBIC/NPTOT, study day <= landmark                       [safe]

  EXCLUDED (encode or post-date the outcome):
    DS                 - IS the outcome (used only for label + TTE timing)
    AE                 - AE-driven dropout is the outcome mechanism here
    EX                 - exposure duration is shorter for drop-outs
    SV                 - number of visits attended leaks dropout
    CM, SE             - accumulate post-randomization
    SUPPDM             - contains completion / population flags
    DM ACTARM/ACTARMCD - ACTUAL arm is realized over time (e.g. "Not
                         Treated") -> outcome-adjacent; only PLANNED arm kept
    DM RF*ENDTC/DTH*   - encode WHEN/WHY participation ended -> hard-banned

Imputation/scaling are intentionally deferred to the modeling step so they can
be fit on DEVELOPMENT data only (avoiding train/holdout leakage).

USAGE
  python build_features_cdiscpilot01.py
  python build_features_cdiscpilot01.py --target-mode any        # any non-completion = 1
  python build_features_cdiscpilot01.py --target-mode clinical   # default (see below)
  python build_features_cdiscpilot01.py --death exclude          # drop DEATH subjects
  python build_features_cdiscpilot01.py --landmark-day 1

Requires: pandas, numpy, scikit-learn
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ADMIN_DISPOSITIONS = {"STUDY TERMINATED BY SPONSOR"}  # not patient/clinical-driven

# PLANNED arm only (ACTARM/ACTARMCD deliberately excluded -- see LEAKAGE POLICY)
DM_SAFE = ["USUBJID", "SITEID", "COUNTRY", "AGE", "AGEU", "SEX", "RACE",
           "ETHNIC", "ARM", "ARMCD"]
DM_BANNED = ["RFENDTC", "RFPENDTC", "RFXENDTC", "RFXSTDTC", "DTHDTC", "DTHFL",
             "RFICDTC", "ACTARM", "ACTARMCD"]

QS_ENDPOINTS = ["ACTOT", "CIBIC", "NPTOT"]  # ADAS-Cog, CIBIC+, NPI-X


# --------------------------------------------------------------------------
def load(indir: Path, name: str) -> pd.DataFrame | None:
    p = indir / "csv" / f"{name.upper()}.csv"
    if not p.exists():
        print(f"  [warn] {p} not found", file=sys.stderr)
        return None
    return pd.read_csv(p, dtype=str, keep_default_na=True)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def study_day(start_iso: pd.Series, event_iso: pd.Series) -> pd.Series:
    """SDTM study-day convention: (event - start) days, +1 when event >= start."""
    s = pd.to_datetime(start_iso, errors="coerce")
    e = pd.to_datetime(event_iso, errors="coerce")
    d = (e - s).dt.days
    return d.where(d < 0, d + 1)


def baseline_findings(df, testcd, valcol, dycol, landmark, tests=None, prefix=""):
    """One row per subject: BASELINE value of each test = last measurement on/
    before the landmark day, pivoted wide."""
    if df is None:
        return None
    need = {"USUBJID", testcd, valcol, dycol}
    if not need.issubset(df.columns):
        print(f"  [warn] {prefix or valcol}: missing {need - set(df.columns)}",
              file=sys.stderr)
        return None
    d = df[["USUBJID", testcd, valcol, dycol]].copy()
    if tests is not None:
        d = d[d[testcd].isin(tests)]
    d[valcol] = to_num(d[valcol])
    d[dycol] = to_num(d[dycol])
    d = d[d[dycol] <= landmark].dropna(subset=[valcol])   # baseline window
    if d.empty:
        return None
    d = d.sort_values(["USUBJID", testcd, dycol])
    d = d.groupby(["USUBJID", testcd], as_index=False).last()  # closest to landmark
    wide = d.pivot(index="USUBJID", columns=testcd, values=valcol)
    wide.columns = [f"{prefix}{c}" for c in wide.columns]
    return wide.reset_index()


def mh_features(mh, landmark):
    """Pre-existing medical history -> counts + top-body-system flags.
    Landmark-guarded: any MH record with study day > landmark is dropped."""
    if mh is None or "USUBJID" not in mh.columns:
        return None
    mh = mh.copy()
    if "MHDY" in mh.columns:
        dy = to_num(mh["MHDY"])
        before = dy.isna() | (dy <= landmark)   # undated history kept; future dropped
        dropped = int((~before).sum())
        if dropped:
            print(f"  MH: dropped {dropped} records with study day > {landmark}")
        mh = mh[before]
    g = mh.groupby("USUBJID")
    feats = pd.DataFrame({"MH_N_RECORDS": g.size()})
    if "MHDECOD" in mh.columns:
        feats["MH_N_CONDITIONS"] = g["MHDECOD"].nunique()
    if "MHBODSYS" in mh.columns:
        feats["MH_N_BODSYS"] = g["MHBODSYS"].nunique()
        for sysname in mh["MHBODSYS"].value_counts().head(6).index:
            col = "MH_BODSYS_" + "".join(
                ch if ch.isalnum() else "_" for ch in str(sysname))[:24].upper()
            hits = mh[mh["MHBODSYS"] == sysname].groupby("USUBJID").size()
            feats[col] = (feats.index.to_series().map(hits).fillna(0) > 0).astype(int)
    return feats.reset_index()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="cdiscpilot01_out_v2")
    ap.add_argument("--outdir", default="cdiscpilot01_out_v2/features")
    ap.add_argument("--landmark-day", type=int, default=1)
    ap.add_argument("--target-mode", choices=["clinical", "any"], default="clinical")
    ap.add_argument("--death", choices=["event", "exclude"], default="event")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"study": "cdiscpilot01", "config": vars(args).copy()}

    # ---- cohort -------------------------------------------------------------
    subjects = pd.read_csv(indir / "subjects.csv", dtype=str, keep_default_na=True)
    subjects["RANDOMIZED"] = subjects["RANDOMIZED"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    coh = subjects[subjects["RANDOMIZED"]].copy()
    manifest["n_randomized"] = len(coh)
    print(f"Randomized subjects: {len(coh)}")

    # ---- ISSUE 2: randomized-but-no-disposition ----------------------------
    disp_raw = coh["DISPOSITION"]
    no_disp = disp_raw.isna() | (disp_raw.astype(str).str.strip().isin(["", "nan", "NaN"]))
    manifest["n_no_disposition_excluded"] = int(no_disp.sum())
    if no_disp.any():
        ids = coh.loc[no_disp, "USUBJID"].tolist()
        print(f"  [issue-2] {no_disp.sum()} randomized subject(s) have NO disposition "
              f"record -> EXCLUDED (cannot be labeled): {ids}")
        coh = coh[~no_disp].copy()
    else:
        print("  [issue-2] all randomized subjects have a disposition record -> none dropped")

    disp = coh["DISPOSITION"].astype(str).str.upper().str.strip()

    # ---- target-mode exclusions --------------------------------------------
    if args.target_mode == "clinical":
        adm = disp.isin({d.upper() for d in ADMIN_DISPOSITIONS})
        print(f"  [clinical] dropping {int(adm.sum())} administrative "
              f"(STUDY TERMINATED BY SPONSOR) subject(s)")
        coh, disp = coh[~adm].copy(), disp[~adm]
    if args.death == "exclude":
        dth = disp == "DEATH"
        print(f"  dropping {int(dth.sum())} DEATH subject(s)")
        coh, disp = coh[~dth].copy(), disp[~dth]

    # ---- outcomes: binary (primary) + time-to-event (sensitivity) ----------
    coh["TARGET"] = (disp != "COMPLETED").astype(int)     # binary primary outcome
    coh["EVENT"] = coh["TARGET"]                          # TTE event indicator
    coh["TIME_DAYS"] = np.nan
    ds = load(indir, "ds")
    if ds is not None and {"USUBJID", "DSSTDTC"}.issubset(ds.columns) \
            and "RFSTDTC" in coh.columns:
        de = ds.copy()
        if "DSCAT" in de.columns:
            de = de[de["DSCAT"].astype(str).str.upper() == "DISPOSITION EVENT"]
        sort_cols = [c for c in ["USUBJID", "DSSTDTC", "DSSEQ"] if c in de.columns]
        de = de.sort_values(sort_cols).groupby("USUBJID", as_index=False).last()
        dmap = dict(zip(de["USUBJID"], de["DSSTDTC"]))
        coh["_DISP_DTC"] = coh["USUBJID"].map(dmap)
        coh["TIME_DAYS"] = study_day(coh["RFSTDTC"], coh["_DISP_DTC"])
        coh = coh.drop(columns=["_DISP_DTC"])
        print(f"  TTE outcome computed (median time on study "
              f"{coh['TIME_DAYS'].median():.0f} days; "
              f"{coh['TIME_DAYS'].isna().sum()} missing)")

    manifest["n_modeling_cohort"] = len(coh)
    manifest["target_balance"] = coh["TARGET"].value_counts().sort_index().to_dict()
    print(f"Modeling cohort: {len(coh)}  |  binary target balance: "
          f"{manifest['target_balance']}")

    # ---- baseline features (leakage-safe) ----------------------------------
    lm = args.landmark_day
    dm = load(indir, "dm")
    dm_safe = dm[[c for c in DM_SAFE if c in dm.columns]].copy()
    dm_safe["AGE"] = to_num(dm_safe.get("AGE"))
    banned_present = [c for c in DM_BANNED if c in dm.columns]
    print(f"  DM: kept {list(dm_safe.columns)}")
    print(f"  DM: EXCLUDED (leakage) {banned_present}")

    mh = mh_features(load(indir, "mh"), lm)
    vs = baseline_findings(load(indir, "vs"), "VSTESTCD", "VSSTRESN", "VSDY", lm, prefix="VS_")
    lb = baseline_findings(load(indir, "lb"), "LBTESTCD", "LBSTRESN", "LBDY", lm, prefix="LB_")
    qs = baseline_findings(load(indir, "qs"), "QSTESTCD", "QSSTRESN", "QSDY", lm,
                           tests=QS_ENDPOINTS, prefix="QSBL_")

    # ---- assemble one row per subject --------------------------------------
    keep = ["USUBJID", "TARGET", "EVENT", "TIME_DAYS"]
    X = coh[keep].merge(dm_safe, on="USUBJID", how="left")
    for block in (mh, vs, lb, qs):
        if block is not None:
            X = X.merge(block, on="USUBJID", how="left")

    for ep in QS_ENDPOINTS:                       # CIBIC has no true baseline
        col = f"QSBL_{ep}"
        if col in X.columns and X[col].notna().sum() == 0:
            print(f"  [note] {col} has no baseline value -> dropped "
                  f"(expected for CIBIC, a change-from-baseline instrument)")
            X = X.drop(columns=[col])

    outcome_cols = ["TARGET", "EVENT", "TIME_DAYS"]
    feature_cols = [c for c in X.columns if c not in (["USUBJID"] + outcome_cols)]
    manifest["n_features"] = len(feature_cols)
    manifest["feature_columns"] = feature_cols
    manifest["outcome_columns"] = outcome_cols
    manifest["outcome_primary"] = "binary (TARGET)"
    manifest["outcome_also_provided"] = "time-to-event (EVENT, TIME_DAYS)"
    manifest["information_type"] = "baseline-only"
    manifest["landmark_day"] = lm
    manifest["dm_features_used"] = [c for c in dm_safe.columns if c != "USUBJID"]
    manifest["dm_excluded_leakage"] = banned_present
    print(f"\nFeature matrix: {X.shape[0]} subjects x {len(feature_cols)} features "
          f"(+ {len(outcome_cols)} outcome cols)")

    miss = X[feature_cols].isna().mean().sort_values(ascending=False)
    print("Top missingness (fraction):")
    print(miss.head(8).round(3).to_string())

    # ---- 70/30 stratified split --------------------------------------------
    dev, hold = train_test_split(X, test_size=args.test_size, random_state=args.seed,
                                 stratify=X["TARGET"])
    X.to_csv(outdir / "model_matrix.csv", index=False)
    dev.to_csv(outdir / "dev.csv", index=False)
    hold.to_csv(outdir / "holdout.csv", index=False)
    manifest["split"] = {"dev_n": len(dev), "holdout_n": len(hold),
                         "dev_pos_rate": round(float(dev["TARGET"].mean()), 4),
                         "holdout_pos_rate": round(float(hold["TARGET"].mean()), 4),
                         "seed": args.seed}
    (outdir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    def bal(d):
        vc = d["TARGET"].value_counts().sort_index()
        return f"n={len(d)}  0={vc.get(0,0)}  1={vc.get(1,0)}  pos_rate={d['TARGET'].mean():.3f}"
    print("\n=== 70/30 stratified split ===")
    print("Development :", bal(dev))
    print("Holdout     :", bal(hold))

    print("\n=== METHODOLOGY (recorded to feature_manifest.json) ===")
    print(" information type : baseline-only (randomization landmark, day <= "
          f"{lm}; no time-varying covariates)")
    print(" completion cutoff: not applicable (no post-baseline features to leak)")
    print(" outcome          : primary = BINARY; also provided = time-to-event "
          "(completion censored)")
    print(" explainability   : logistic-regression odds ratios + SHAP on GBM")
    print(f"\nWrote:\n  {outdir/'model_matrix.csv'}\n  {outdir/'dev.csv'}"
          f"\n  {outdir/'holdout.csv'}\n  {outdir/'feature_manifest.json'}")
    print("\nReminder: fit imputation/scaling on dev only, then apply to holdout.")


if __name__ == "__main__":
    main()

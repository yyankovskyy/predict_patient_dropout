#!/usr/bin/env python3
"""
survival_cdiscpilot01.py   (time-to-event stage)
================================================
Survival / time-to-event analysis of study discontinuation, complementing the
binary classifiers. All predictors are BASELINE-ONLY; only the OUTCOME changes
from a yes/no label to "time until dropout, with completers censored."

Produces (first-class outputs):
  1. Kaplan-Meier (KM)      - the retention curve overall and by treatment arm,
                              with median time-in-study and a log-rank test.
  2. Cox proportional hazards - hazard ratios (with 95% CI and p) for baseline
                              covariates, concordance, and a PH-assumption check.
  3. Competing-risks CIF    - cause-specific cumulative incidence (AE, withdrawal,
                              death, ...), because "any-cause" dropout lumps
                              competing reasons that a single KM curve blurs.

KM, the log-rank test, and the cumulative-incidence functions are implemented in
numpy (always available). Cox regression and its diagnostics use `lifelines`
(a note is printed if it is not installed).

Inputs : cdiscpilot01_out_v2/features/model_matrix.csv  (TIME_DAYS, EVENT, arm, covariates)
         cdiscpilot01_out_v2/subjects.csv                (DISPOSITION = cause)
Outputs: cdiscpilot01_out_v2/survival/  (plots, tables, survival_report.json)

Requires: pandas, numpy, matplotlib   |   Optional: lifelines (Cox + PH test)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Cox covariates (baseline-only, parsimonious for ~137 events)
COX_NUM = ["AGE", "QSBL_ACTOT", "QSBL_NPTOT"]   # numeric covariates
COX_CAT = ["SEX"]                                # categorical covariates (one-hot)


# --------------------------------------------------------------------------
# Kaplan-Meier (numpy)
# --------------------------------------------------------------------------
def km_estimate(time, event):
    """Return step arrays (t, S(t), n_at_risk, d_events) and the median."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time)
    time, event = time[order], event[order]
    uniq = np.unique(time[event == 1])          # distinct event times
    ts, surv, at_risk, events = [], [], [], []
    S = 1.0
    for t in uniq:
        n = int((time >= t).sum())              # at risk just before t
        d = int(((time == t) & (event == 1)).sum())
        if n > 0:
            S *= (1 - d / n)
        ts.append(t); surv.append(S); at_risk.append(n); events.append(d)
    ts = np.array(ts); surv = np.array(surv)
    # median = first time S(t) <= 0.5
    median = float(ts[surv <= 0.5][0]) if np.any(surv <= 0.5) else np.nan
    return ts, surv, np.array(at_risk), np.array(events), median


def logrank_test(time, event, group):
    """k-sample log-rank test. Returns (chi2, dof, p)."""
    from scipy.stats import chi2 as chi2dist  # scipy is a sklearn dependency
    time = np.asarray(time, float); event = np.asarray(event, int)
    group = np.asarray(group)
    groups = list(pd.unique(group))
    k = len(groups)
    if k < 2:
        return np.nan, 0, np.nan
    event_times = np.unique(time[event == 1])
    O_E = np.zeros(k)
    V = np.zeros((k, k))
    for t in event_times:
        n = int((time >= t).sum())
        d = int(((time == t) & (event == 1)).sum())
        if n <= 1:
            continue
        n_g = np.array([int(((time >= t) & (group == g)).sum())
                        for g in groups], float)
        d_g = np.array([int(((time == t) & (event == 1) & (group == g)).sum())
                        for g in groups], float)
        E_g = d * n_g / n
        O_E += (d_g - E_g)
        f = d * (n - d) / (n - 1) / n
        for i in range(k):
            for j in range(k):
                V[i, j] += f * (n_g[i] * ((i == j) - n_g[j] / n))
    # drop last group to avoid singular covariance
    z = O_E[:-1]
    Vr = V[:-1, :-1]
    try:
        stat = float(z @ np.linalg.solve(Vr, z))
    except np.linalg.LinAlgError:
        stat = float(z @ np.linalg.pinv(Vr) @ z)
    dof = k - 1
    p = float(chi2dist.sf(stat, dof))
    return stat, dof, p


def cif_by_cause(time, event, cause, causes):
    """Aalen-Johansen cause-specific cumulative incidence for each cause."""
    time = np.asarray(time, float); event = np.asarray(event, int)
    cause = np.asarray(cause, dtype=object)
    uniq = np.unique(time[event == 1])
    S_prev = 1.0
    grid, cif = [], {c: [] for c in causes}
    running = {c: 0.0 for c in causes}
    for t in uniq:
        n = int((time >= t).sum())
        d_all = int(((time == t) & (event == 1)).sum())
        for c in causes:
            d_c = int(((time == t) & (event == 1) & (cause == c)).sum())
            running[c] += S_prev * (d_c / n) if n > 0 else 0.0
            cif[c].append(running[c])
        grid.append(t)
        if n > 0:
            S_prev *= (1 - d_all / n)
    return np.array(grid), {c: np.array(v) for c, v in cif.items()}


# --------------------------------------------------------------------------
def run_survival(indir="cdiscpilot01_out_v2/features",
                 subjects="cdiscpilot01_out_v2/subjects.csv",
                 outdir="cdiscpilot01_out_v2/survival",
                 arm_col="ARMCD"):
    indir = Path(indir); outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mm = pd.read_csv(indir / "model_matrix.csv")
    if "TIME_DAYS" not in mm.columns or "EVENT" not in mm.columns:
        raise SystemExit("model_matrix.csv lacks TIME_DAYS/EVENT; re-run features step.")
    mm = mm.dropna(subset=["TIME_DAYS", "EVENT"]).copy()
    mm["TIME_DAYS"] = pd.to_numeric(mm["TIME_DAYS"], errors="coerce")
    mm = mm[mm["TIME_DAYS"] > 0]
    if arm_col not in mm.columns:
        arm_col = "ARM" if "ARM" in mm.columns else None

    report = {"n": int(len(mm)), "events": int(mm["EVENT"].sum()),
              "arm_col": arm_col}

    # ---- KM overall + by arm + log-rank ------------------------------------
    ts, surv, atrisk, ev, med = km_estimate(mm["TIME_DAYS"], mm["EVENT"])
    report["median_time_overall"] = med
    print(f"KM overall: n={len(mm)}, events={int(mm['EVENT'].sum())}, "
          f"median time-in-study = {med:.0f} days"
          if not np.isnan(med) else
          f"KM overall: median not reached")

    by_arm = {}
    if arm_col:
        for a, g in mm.groupby(arm_col):
            _, s_a, _, _, med_a = km_estimate(g["TIME_DAYS"], g["EVENT"])
            by_arm[str(a)] = {"n": int(len(g)),
                              "events": int(g["EVENT"].sum()),
                              "median": None if np.isnan(med_a) else med_a}
        stat, dof, p = logrank_test(mm["TIME_DAYS"], mm["EVENT"], mm[arm_col])
        report["logrank"] = {"chi2": round(stat, 3), "dof": dof, "p_value": p}
        report["km_by_arm"] = by_arm
        print(f"Log-rank across {arm_col}: chi2={stat:.2f} (df={dof}), "
              f"p={p:.2e}")

    # ---- KM plot -----------------------------------------------------------
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.step(np.r_[0, ts], np.r_[1, surv], where="post",
                color="black", lw=2, label="Overall")
        if arm_col:
            for a, g in mm.groupby(arm_col):
                t_a, s_a, _, _, _ = km_estimate(g["TIME_DAYS"], g["EVENT"])
                ax.step(np.r_[0, t_a], np.r_[1, s_a], where="post",
                        lw=1.4, label=str(a))
        ax.set_xlabel("Days since randomization")
        ax.set_ylabel("Probability still in study  S(t)")
        ax.set_title("Kaplan-Meier retention")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(outdir / "km_by_arm.png", dpi=140); plt.close()

    # ---- Competing-risks CIF by cause --------------------------------------
    cif_summary = {}
    subj_p = Path(subjects)
    if subj_p.exists() and "USUBJID" in mm.columns:
        subj = pd.read_csv(subj_p)[["USUBJID", "DISPOSITION"]]
        d = mm.merge(subj, on="USUBJID", how="left")
        cause = d["DISPOSITION"].astype(str).str.upper().where(d["EVENT"] == 1, "CENSORED")
        top = cause[d["EVENT"] == 1].value_counts()
        keep = list(top.head(4).index)
        cause_grp = cause.where(cause.isin(keep + ["CENSORED"]), "OTHER")
        causes = keep + (["OTHER"] if (cause_grp == "OTHER").any() else [])
        grid, cif = cif_by_cause(d["TIME_DAYS"], d["EVENT"], cause_grp, causes)
        for c in causes:
            cif_summary[c] = round(float(cif[c][-1]), 3) if len(cif[c]) else 0.0
        report["cif_final"] = cif_summary
        pd.DataFrame({"cause": list(cif_summary), "cif_at_end": list(cif_summary.values())}) \
            .to_csv(outdir / "cif_summary.csv", index=False)
        print("Cumulative incidence at end of study, by cause:",
              {c: f"{v:.1%}" for c, v in cif_summary.items()})
        if HAVE_MPL:
            fig, ax = plt.subplots(figsize=(7, 4.4))
            for c in causes:
                ax.step(np.r_[0, grid], np.r_[0, cif[c]], where="post", lw=1.6,
                        label=c.title())
            ax.set_xlabel("Days since randomization")
            ax.set_ylabel("Cumulative incidence")
            ax.set_title("Cause-specific cumulative incidence (competing risks)")
            ax.legend(fontsize=8)
            plt.tight_layout(); plt.savefig(outdir / "cif_by_cause.png", dpi=140); plt.close()

    # ---- Cox proportional hazards (lifelines) ------------------------------
    report["cox"] = None
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
        num = [c for c in COX_NUM if c in mm.columns]
        df = mm[["TIME_DAYS", "EVENT"] + num].copy()
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop numeric covariates that are too sparse at baseline (e.g. baseline
        # QS often missing) instead of dropping every patient who lacks them.
        MAX_MISSING = 0.30
        dropped = [c for c in num if df[c].isna().mean() > MAX_MISSING]
        if dropped:
            df = df.drop(columns=dropped)
            print(f"[Cox] dropped sparse covariate(s) (> {int(MAX_MISSING*100)}% "
                  f"missing): {dropped}")
        # Add categorical covariates (e.g. SEX) as one-hot dummies, if populated.
        for c in COX_CAT:
            if c in mm.columns and mm[c].isna().mean() <= MAX_MISSING \
                    and mm[c].nunique(dropna=True) >= 2:
                df = pd.concat([df, pd.get_dummies(mm[c], prefix=c, drop_first=True)],
                               axis=1)
        # Drop near-constant covariates (no variation -> unidentifiable).
        const = [c for c in df.columns
                 if c not in ("TIME_DAYS", "EVENT") and df[c].nunique(dropna=True) < 2]
        if const:
            df = df.drop(columns=const)
            print(f"[Cox] dropped constant covariate(s): {const}")
        if arm_col:  # add treatment arm as dummies (reference dropped)
            df = pd.concat([df, pd.get_dummies(mm[arm_col], prefix="arm",
                                               drop_first=True)], axis=1)
        df = df.dropna()

        # Guard: never hand lifelines a degenerate frame (this is what caused the
        # ZeroDivisionError: an empty table with 0 rows).
        n_rows = len(df)
        n_events = int(df["EVENT"].sum()) if n_rows else 0
        n_covars = df.shape[1] - 2
        if n_rows < 20 or n_events < 10 or n_covars < 1:
            print(f"[note] Cox skipped: insufficient data after cleaning "
                  f"(rows={n_rows}, events={n_events}, covariates={n_covars}). "
                  f"KM, log-rank and competing-risks results above are unaffected.")
            report["cox"] = {"skipped": "insufficient data after cleaning",
                             "rows": n_rows, "events": n_events, "covariates": n_covars}
        else:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df, duration_col="TIME_DAYS", event_col="EVENT")
            summ = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                                "exp(coef) upper 95%", "p"]].copy()
            summ.columns = ["log_HR", "HR", "HR_low95", "HR_high95", "p"]
            summ.round(4).to_csv(outdir / "cox_summary.csv")
            report["cox"] = {
                "n": n_rows, "events": n_events,
                "concordance": round(float(cph.concordance_index_), 4),
                "hazard_ratios": {k: round(float(v), 3) for k, v in summ["HR"].items()},
                "p_values": {k: float(v) for k, v in summ["p"].items()},
            }
            print(f"\nCox model: n={n_rows}, events={n_events}, "
                  f"concordance (c-index) = {cph.concordance_index_:.3f}")
            print(summ.round(3).to_string())
            # PH assumption check
            try:
                ph = proportional_hazard_test(cph, df, time_transform="rank")
                ph_p = {k: float(v) for k, v in ph.summary["p"].items()}
                report["cox"]["ph_test_p"] = ph_p
                violated = [k for k, v in ph_p.items() if v < 0.05]
                report["cox"]["ph_violations"] = violated
                with open(outdir / "cox_ph_check.txt", "w") as f:
                    f.write(ph.summary.to_string())
                print("PH assumption: " +
                      ("holds for all covariates (no p<0.05)." if not violated
                       else f"possible violation for {violated} (p<0.05)."))
            except Exception as e:
                print(f"[note] PH check skipped ({type(e).__name__}).")
    except ImportError:
        print("\n[note] lifelines not installed -> Cox model + PH check skipped. "
              "KM, log-rank, and competing-risks outputs are still produced. "
              "Install with: pip install lifelines")
    except Exception as e:
        # Any other lifelines/numerical failure must not kill the pipeline.
        print(f"\n[note] Cox model skipped ({type(e).__name__}: {e}). "
              f"KM, log-rank, and competing-risks results above are unaffected.")
        report["cox"] = {"skipped": f"{type(e).__name__}: {e}"}

    (outdir / "survival_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSurvival artifacts written to {outdir}/")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="cdiscpilot01_out_v2/features")
    ap.add_argument("--subjects", default="cdiscpilot01_out_v2/subjects.csv")
    ap.add_argument("--outdir", default="cdiscpilot01_out_v2/survival")
    ap.add_argument("--arm-col", default="ARMCD")
    args = ap.parse_args()
    run_survival(args.indir, args.subjects, args.outdir, args.arm_col)


if __name__ == "__main__":
    main()

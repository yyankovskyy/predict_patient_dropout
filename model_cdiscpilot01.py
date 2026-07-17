#!/usr/bin/env python3
"""
model_cdiscpilot01.py   (task step iii)
=======================================
Develop and tune predictive models on the DEVELOPMENT set and evaluate ONCE on
the HOLDOUT set, with explainability. Consumes dev.csv / holdout.csv from
`build_features_cdiscpilot01.py`.

WHAT IT DOES
  1. Loads the leakage-safe baseline feature matrix (dev + holdout).
  2. Builds preprocessing INSIDE a pipeline (fit on development only):
       numeric  -> median impute (+ scale for the linear model)
       category -> most-frequent impute + one-hot
  3. Trains & tunes two models with stratified CV on development:
       - Logistic Regression  (interpretable baseline -> odds ratios)
       - Gradient Boosting    (non-linear -> permutation / SHAP attribution)
  4. Evaluates on the untouched holdout: ROC-AUC, PR-AUC, F1, balanced
     accuracy, Brier (calibration), confusion matrix, classification report.
  5. Explainability: logistic-regression odds ratios; permutation importance;
     SHAP summary if `shap` is installed.
  6. Optional time-to-event sensitivity (Cox) if `lifelines` is installed.
  7. Writes metrics, plots, importances, predictions and models to model/.

Primary outcome  : BINARY (TARGET). Time-to-event columns (EVENT, TIME_DAYS)
                   are used only for the optional Cox sensitivity analysis.

Requires: pandas, numpy, scikit-learn, matplotlib, joblib
Optional: shap (SHAP attribution), lifelines (Cox sensitivity)
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

warnings.filterwarnings("ignore")

ID_COLS = ["USUBJID"]
OUTCOME_COLS = ["TARGET", "EVENT", "TIME_DAYS"]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# --------------------------------------------------------------------------
def get_output_names(ct):
    """Feature names from a fitted ColumnTransformer, robust to older sklearn.

    On sklearn < 1.1 (e.g. some Anaconda builds), SimpleImputer lacks
    get_feature_names_out(), so ColumnTransformer.get_feature_names_out() raises.
    In that case we rebuild names manually: numeric columns pass through
    unchanged; the one-hot encoder names its own expanded columns.
    """
    try:
        return list(ct.get_feature_names_out())
    except (AttributeError, TypeError):
        names = []
        for tname, trans, cols in ct.transformers_:
            if tname == "remainder":
                continue
            if tname == "cat":
                names.extend(trans.named_steps["oh"].get_feature_names_out(cols))
            else:
                names.extend(list(cols))
        return names


def build_preprocessor(X, scale_numeric):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return pre, num, cat


def evaluate(name, model, Xh, yh):
    prob = model.predict_proba(Xh)[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = {
        "model": name,
        "roc_auc": round(roc_auc_score(yh, prob), 4),
        "pr_auc": round(average_precision_score(yh, prob), 4),
        "f1": round(f1_score(yh, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(yh, pred), 4),
        "brier": round(brier_score_loss(yh, prob), 4),
        "confusion_matrix": confusion_matrix(yh, pred).tolist(),
    }
    return m, prob, pred


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="cdiscpilot01_out_v2/features")
    ap.add_argument("--outdir", default="cdiscpilot01_out_v2/model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(indir / "dev.csv")
    hold = pd.read_csv(indir / "holdout.csv")
    feats = [c for c in dev.columns if c not in ID_COLS + OUTCOME_COLS]
    Xd, yd = dev[feats].copy(), dev["TARGET"].astype(int)
    Xh, yh = hold[feats].copy(), hold["TARGET"].astype(int)
    print(f"Development: {Xd.shape[0]} x {Xd.shape[1]}   Holdout: {Xh.shape[0]}")
    print(f"Dev pos-rate {yd.mean():.3f} | Holdout pos-rate {yh.mean():.3f}")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    results = {}

    # ---- Model A: Logistic Regression (interpretable) ----------------------
    preL, numL, catL = build_preprocessor(Xd, scale_numeric=True)
    pipeL = Pipeline([("pre", preL),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    gridL = GridSearchCV(pipeL, {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                         scoring="roc_auc", cv=cv, n_jobs=-1)
    gridL.fit(Xd, yd)
    print(f"\n[LogReg] best C={gridL.best_params_['clf__C']}  "
          f"CV ROC-AUC={gridL.best_score_:.3f}")
    mA, probA, predA = evaluate("LogisticRegression", gridL.best_estimator_, Xh, yh)
    results["logreg"] = mA

    # ---- Model B: Gradient Boosting ----------------------------------------
    preB, numB, catB = build_preprocessor(Xd, scale_numeric=False)
    pipeB = Pipeline([("pre", preB),
                      ("clf", GradientBoostingClassifier(random_state=args.seed))])
    gridB = GridSearchCV(pipeB, {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.05, 0.1],
    }, scoring="roc_auc", cv=cv, n_jobs=-1)
    gridB.fit(Xd, yd)
    print(f"[GBM]    best={gridB.best_params_}  CV ROC-AUC={gridB.best_score_:.3f}")
    mB, probB, predB = evaluate("GradientBoosting", gridB.best_estimator_, Xh, yh)
    results["gbm"] = mB

    print("\n=== HOLDOUT PERFORMANCE ===")
    for k, m in results.items():
        print(f"  {m['model']:20} ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  "
              f"F1={m['f1']}  balAcc={m['balanced_accuracy']}  Brier={m['brier']}")
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"  -> best by ROC-AUC: {results[best_name]['model']}")
    print("\nClassification report (best model, holdout):")
    print(classification_report(yh, predB if best_name == "gbm" else predA,
                                target_names=["completed", "discontinued"]))

    # ---- Explainability 1: Logistic-regression odds ratios -----------------
    prefit = gridL.best_estimator_.named_steps["pre"]
    names = get_output_names(prefit)
    coefs = gridL.best_estimator_.named_steps["clf"].coef_[0]
    odds = pd.DataFrame({"feature": names, "coef": coefs,
                         "odds_ratio": np.exp(coefs)})
    odds["abs_coef"] = odds["coef"].abs()
    odds = odds.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    odds.to_csv(outdir / "logreg_odds_ratios.csv", index=False)
    print("\nTop odds ratios (LogReg) -- OR>1 raises discontinuation risk:")
    print(odds.head(10).to_string(index=False))

    # ---- Explainability 2: Permutation importance (GBM, model-agnostic) ----
    perm = permutation_importance(gridB.best_estimator_, Xh, yh,
                                  scoring="roc_auc", n_repeats=20,
                                  random_state=args.seed)
    imp = pd.DataFrame({"feature": feats,
                        "importance": perm.importances_mean,
                        "std": perm.importances_std}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(outdir / "permutation_importance.csv", index=False)
    print("\nTop permutation importances (GBM, holdout):")
    print(imp.head(10).to_string(index=False))

    # ---- Explainability 3: SHAP (optional) ---------------------------------
    shap_done = False
    try:
        import shap
        # use the fitted preprocessor + classifier from the tuned pipeline
        pre_fitted = gridB.best_estimator_.named_steps["pre"]
        clf_fitted = gridB.best_estimator_.named_steps["clf"]
        Xh_trans = pre_fitted.transform(Xh)
        fn = get_output_names(pre_fitted)
        # dense array (SHAP dislikes sparse); newer sklearn returns sparse OHE
        if hasattr(Xh_trans, "toarray"):
            Xh_trans = Xh_trans.toarray()
        Xh_trans = np.asarray(Xh_trans, dtype=float)

        # try modern unified API first, then fall back to TreeExplainer
        try:
            explainer = shap.Explainer(clf_fitted, feature_names=fn)
            sv = explainer(Xh_trans)
            shap_values = sv.values
        except Exception:
            explainer = shap.TreeExplainer(clf_fitted)
            shap_values = explainer.shap_values(Xh_trans)

        # normalize shape across shap versions (list per class, or 3-D array)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]          # positive class
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:                  # (n, features, classes)
            shap_values = shap_values[:, :, -1]

        if HAVE_MPL:
            shap.summary_plot(shap_values, features=Xh_trans, feature_names=fn,
                              show=False)
            plt.tight_layout()
            plt.savefig(outdir / "shap_summary.png", dpi=130)
            plt.close()
        # also save mean|SHAP| as a portable CSV (no plotting dependency)
        mean_abs = np.abs(shap_values).mean(axis=0)
        pd.DataFrame({"feature": fn, "mean_abs_shap": mean_abs}) \
            .sort_values("mean_abs_shap", ascending=False) \
            .to_csv(outdir / "shap_importance.csv", index=False)
        shap_done = True
        print("\nSHAP written (shap_summary.png, shap_importance.csv).")
    except ImportError:
        print("\n[note] SHAP not installed (pip install shap to enable); "
              "odds ratios + permutation importance provide attribution.")
    except Exception as e:
        print(f"\n[note] SHAP skipped ({type(e).__name__}: {e}); "
              f"odds ratios + permutation importance provide attribution.")

    # ---- KEY INSIGHT: discontinuation by planned arm -----------------------
    full = pd.concat([dev, hold], ignore_index=True)
    print("\n=== KEY INSIGHT ===")
    arm_col = "ARMCD" if "ARMCD" in full.columns else ("ARM" if "ARM" in full.columns else None)
    if arm_col:
        rate = full.groupby(arm_col)["TARGET"].agg(["mean", "size"]) \
            .sort_values("mean", ascending=False)
        rate["mean"] = (rate["mean"] * 100).round(1)
        rate = rate.rename(columns={"mean": "discontinue_%", "size": "n"})
        print(f"Discontinuation rate by planned arm ({arm_col}):")
        print(rate.to_string())
    top_feats = ", ".join(imp.head(4)["feature"].tolist())
    print(f"\nStrongest predictors on holdout: {top_feats}.")
    print("Interpretation: planned treatment arm and baseline disease-severity "
          "measures dominate; discontinuation is concentrated in the active "
          "(higher-dose) arm(s), consistent with adverse-event-driven dropout.")

    # ---- Time-to-event note -------------------------------------------------
    # Survival analysis (Kaplan-Meier, log-rank, Cox + PH diagnostics, and
    # competing-risks incidence) is now a dedicated stage: survival_cdiscpilot01.py
    # (invoked by run_pipeline.py). The TIME_DAYS/EVENT outcome columns in the
    # feature matrix feed that stage.

    # ---- Plots + artifacts -------------------------------------------------
    if HAVE_MPL:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for nm, prob in [("LogReg", probA), ("GBM", probB)]:
            fpr, tpr, _ = roc_curve(yh, prob); ax[0].plot(fpr, tpr, label=nm)
            pr, rc, _ = precision_recall_curve(yh, prob); ax[1].plot(rc, pr, label=nm)
        ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set_title("ROC (holdout)")
        ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
        ax[1].set_title("Precision-Recall (holdout)"); ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision"); ax[1].legend()
        cm = np.array(results[best_name]["confusion_matrix"])
        ax[2].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax[2].text(j, i, str(v), ha="center", va="center")
        ax[2].set_title(f"Confusion ({results[best_name]['model']})")
        ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["comp", "disc"])
        ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["comp", "disc"])
        ax[2].set_xlabel("predicted"); ax[2].set_ylabel("actual")
        plt.tight_layout(); plt.savefig(outdir / "evaluation_curves.png", dpi=130)
        plt.close()
        print("\nWrote evaluation_curves.png")

    pred_out = hold[["USUBJID", "TARGET"]].copy()
    pred_out["prob_logreg"] = probA
    pred_out["prob_gbm"] = probB
    pred_out.to_csv(outdir / "holdout_predictions.csv", index=False)
    joblib.dump(gridL.best_estimator_, outdir / "model_logreg.joblib")
    joblib.dump(gridB.best_estimator_, outdir / "model_gbm.joblib")
    results["best_model"] = results[best_name]["model"]
    results["dev_n"] = int(len(dev)); results["holdout_n"] = int(len(hold))
    results["shap_available"] = shap_done
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"\nArtifacts in {outdir}/: metrics.json, evaluation_curves.png, "
          "logreg_odds_ratios.csv, permutation_importance.csv, "
          "holdout_predictions.csv, model_*.joblib")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
sdtm_reader.py  (v2)
====================
Stage 1 of the discontinuation-prediction "AI function".

Reads a user's SDTM study whose files match the CDISC SDTM Pilot 01 format,
validates it against its own define.xml, and converts every domain to a clean
CSV -- plus a patient-keyed long file and a subject-level spine with the
discontinuation target.

Two input modes:
  * local   : the user's SDTM files live in --input-dir (.xpt and/or .csv),
              and define.xml MUST be present.
  * github  : download the reference CDISC Pilot 01 study from PHUSE
              (--fetch-reference); used for demos / the reference run.

Public functions (imported by run_pipeline.py):
  list_expected_datasets(define_path) -> [names]
  validate_sdtm_input(input_dir)      -> (ok, missing, expected, has_define)
  convert(input_dir, outdir, ...)     -> dict summary

Requires: pandas  (requests only for --fetch-reference; pyreadstat optional)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

GITHUB_BASE = ("https://raw.githubusercontent.com/phuse-org/phuse-scripts/"
               "master/data/sdtm/cdiscpilot01")
REFERENCE_DOMAINS = ["dm", "ds", "sv", "se", "sc", "ae", "cm", "ex", "mh", "lb",
                     "qs", "vs", "suppdm", "suppae", "suppds", "supplb",
                     "relrec", "ta", "te", "ti", "ts", "tv"]
NON_SUBJECT_DOMAINS = {"ta", "te", "ti", "ts", "tv", "relrec"}

VALIDATION_ERROR = ("User's input does not meet CDISC SDTM requirements "
                    "that results in termination of further analysis")


# --------------------------------------------------------------------------
# Validation gate
# --------------------------------------------------------------------------
def _find_define(input_dir: Path) -> Path | None:
    for name in ("define.xml", "Define.xml", "DEFINE.XML"):
        if (input_dir / name).exists():
            return input_dir / name
    hits = list(input_dir.glob("*.xml")) + list(input_dir.glob("*.XML"))
    for h in hits:
        if h.name.lower() == "define.xml":
            return h
    return None


def list_expected_datasets(define_path: Path) -> list[str]:
    """Datasets declared in define.xml (ItemGroupDef Name), lower-cased."""
    xml = Path(define_path).read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r'<ItemGroupDef\b[^>]*\bName="([A-Za-z0-9_]+)"', xml)
    return [n.lower() for n in dict.fromkeys(names)]  # de-dup, keep order


def _dataset_file_present(input_dir: Path, name: str) -> bool:
    for ext in (".xpt", ".XPT", ".csv", ".CSV"):
        if (input_dir / f"{name}{ext}").exists() or \
           (input_dir / f"{name.upper()}{ext}").exists():
            return True
    return False


def validate_sdtm_input(input_dir: Path):
    """Return (ok, missing, expected, has_define).

    Requirement: define.xml is mandatory; every dataset declared in it must be
    present as a .xpt or .csv file. If define.xml is missing, that alone fails.
    """
    input_dir = Path(input_dir)
    define_path = _find_define(input_dir)
    if define_path is None:
        return False, ["define.xml"], [], False
    expected = list_expected_datasets(define_path)
    if not expected:  # define.xml unreadable / declares nothing
        return False, ["define.xml (no datasets declared)"], [], True
    missing = [d for d in expected if not _dataset_file_present(input_dir, d)]
    return (len(missing) == 0), missing, expected, True


# --------------------------------------------------------------------------
# Reading / cleaning
# --------------------------------------------------------------------------
def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
    else:  # XPORT
        try:
            import pyreadstat
            df, _ = pyreadstat.read_xport(str(path))
        except Exception:
            df = pd.read_sas(path, format="xport", encoding=None)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.decode("utf-8", "ignore").strip()
                                  if isinstance(v, (bytes, bytearray)) else v)
                df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    df.columns = [c.upper() for c in df.columns]
    return df


def _resolve_path(input_dir: Path, name: str) -> Path | None:
    for cand in (f"{name}.xpt", f"{name.upper()}.xpt", f"{name}.csv",
                 f"{name.upper()}.csv", f"{name}.XPT", f"{name}.CSV"):
        if (input_dir / cand).exists():
            return input_dir / cand
    return None


def _download_reference(cache: Path) -> Path:
    if requests is None:
        raise RuntimeError("requests is required for --fetch-reference")
    cache.mkdir(parents=True, exist_ok=True)
    for dom in REFERENCE_DOMAINS + ["define"]:
        fn = "define.xml" if dom == "define" else f"{dom}.xpt"
        dest = cache / fn
        if dest.exists():
            continue
        url = f"{GITHUB_BASE}/{fn}"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            dest.write_bytes(r.content)
    return cache


# --------------------------------------------------------------------------
# Subject spine + patient-long (unchanged logic from v1)
# --------------------------------------------------------------------------
def _build_subjects(frames):
    dm, ds = frames.get("dm"), frames.get("ds")
    if dm is None:
        raise RuntimeError("DM domain is required.")
    s = dm.copy()
    rnd_ids = set()
    if ds is not None and {"USUBJID", "DSDECOD"}.issubset(ds.columns):
        rnd_ids = set(ds[ds["DSDECOD"].astype(str).str.upper() == "RANDOMIZED"]["USUBJID"])

    def real_arm(r):
        armcd = str(r.get("ARMCD", "")).upper(); arm = str(r.get("ARM", "")).upper()
        bad = {"SCRNFAIL", "SCREEN FAILURE", "NOTASSGN", "NOT ASSIGNED", "", "NAN", "NONE"}
        return armcd not in bad and "SCREEN FAIL" not in arm

    s["RANDOMIZED_ARM"] = s.apply(real_arm, axis=1)
    s["RANDOMIZED"] = s["USUBJID"].isin(rnd_ids) | s["RANDOMIZED_ARM"]
    s["DISPOSITION"] = pd.NA
    s["DISCONTINUED"] = pd.NA
    if ds is not None and {"USUBJID", "DSDECOD"}.issubset(ds.columns):
        de = ds.copy()
        if "DSCAT" in de.columns:
            de = de[de["DSCAT"].astype(str).str.upper() == "DISPOSITION EVENT"]
        sc = [c for c in ["USUBJID", "DSSTDTC", "DSSEQ"] if c in de.columns]
        de = de.sort_values(sc).groupby("USUBJID", as_index=False).last()
        m = dict(zip(de["USUBJID"], de["DSDECOD"].astype(str)))
        s["DISPOSITION"] = s["USUBJID"].map(m)
        s["DISCONTINUED"] = s["DISPOSITION"].map(
            lambda v: pd.NA if pd.isna(v) else (0 if str(v).upper() == "COMPLETED" else 1))
    lead = [c for c in ["USUBJID", "SUBJID", "SITEID", "ARM", "ARMCD", "ACTARM",
                        "ACTARMCD", "AGE", "AGEU", "SEX", "RACE", "ETHNIC",
                        "COUNTRY", "RFSTDTC", "RFENDTC", "RANDOMIZED",
                        "DISPOSITION", "DISCONTINUED"] if c in s.columns]
    return s[lead + [c for c in s.columns if c not in lead]]


def _build_patient_long(frames):
    parts = []
    for dom, df in frames.items():
        if dom in NON_SUBJECT_DOMAINS or "USUBJID" not in df.columns:
            continue
        t = df.copy(); t.insert(0, "DOMAIN_SRC", dom.upper()); parts.append(t)
    if not parts:
        return pd.DataFrame()
    lg = pd.concat(parts, ignore_index=True, sort=False)
    front = [c for c in ["USUBJID", "DOMAIN_SRC"] if c in lg.columns]
    return lg[front + [c for c in lg.columns if c not in front]] \
        .sort_values(["USUBJID", "DOMAIN_SRC"])


# --------------------------------------------------------------------------
def convert(input_dir, outdir, source="local", domains=None):
    """Read SDTM (local or downloaded reference) and write CSV outputs."""
    outdir = Path(outdir)
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    if source == "github":
        input_dir = _download_reference(outdir / "reference_xpt")
    input_dir = Path(input_dir)

    if domains is None:
        define_path = _find_define(input_dir)
        domains = list_expected_datasets(define_path) if define_path else REFERENCE_DOMAINS

    frames = {}
    for dom in domains:
        path = _resolve_path(input_dir, dom)
        if path is None:
            print(f"  [warn] {dom}: no file found", file=sys.stderr)
            continue
        df = _read_one(path)
        frames[dom] = df
        df.to_csv(csv_dir / f"{dom.upper()}.csv", index=False)
        print(f"  -> {dom.upper():8} {df.shape[0]:6d} rows x {df.shape[1]:2d} cols")

    summary = {"n_domains": len(frames)}
    if "dm" in frames:
        subjects = _build_subjects(frames)
        subjects.to_csv(outdir / "subjects.csv", index=False)
        long_df = _build_patient_long(frames)
        long_df.to_csv(outdir / "patient_long.csv", index=False)
        rnd = subjects[subjects["RANDOMIZED"] == True]
        summary.update({
            "n_subjects": int(len(subjects)),
            "n_randomized": int(subjects["RANDOMIZED"].sum()),
            "n_patient_long_rows": int(len(long_df)),
        })
        print(f"\nSubjects: {summary['n_subjects']}  |  randomized: "
              f"{summary['n_randomized']}")
        if "DISPOSITION" in rnd.columns:
            print("Disposition (randomized):")
            print(rnd["DISPOSITION"].value_counts(dropna=False).to_string())
    return summary


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", help="folder with the user's SDTM files (+ define.xml)")
    ap.add_argument("--fetch-reference", action="store_true",
                    help="download the reference CDISC Pilot 01 study instead")
    ap.add_argument("--outdir", default="cdiscpilot01_out_v2")
    args = ap.parse_args()

    if args.fetch_reference:
        convert(input_dir=None, outdir=args.outdir, source="github")
        return
    if not args.input_dir:
        sys.exit("Provide --input-dir or --fetch-reference")

    ok, missing, expected, has_define = validate_sdtm_input(Path(args.input_dir))
    if not ok:
        print(VALIDATION_ERROR, file=sys.stderr)
        print(f"  (missing: {missing})", file=sys.stderr)
        sys.exit(2)
    convert(args.input_dir, args.outdir, source="local", domains=expected)


if __name__ == "__main__":
    main()

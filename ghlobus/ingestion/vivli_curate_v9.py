"""
vivli_curate_v9.py

Build Vivli FAMLI3 exam metadata and merge it onto the preprocessed prototype.

Vivli delivers C3_SR.csv in already-wide form, unlike the long structured-report
format consumed by tablify_sr_v9.py. This module keeps the v9 curation shape
while adding a Vivli-specific bridge for the wide SR/CRF tables.
"""
import argparse
import os
import zipfile
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from ghlobus.utilities.constants import META_DIR


MISSING_STRINGS = {"", " ", "nan", "NaN", "None", "none", "NULL", "null"}


def _clean_string(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(object).where(~series.astype(str).str.strip().isin(MISSING_STRINGS), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _first_present(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col]
        present = vals.notna() & (vals.astype(str).str.strip() != "")
        out.loc[present & out.isna()] = vals.loc[present & out.isna()]
    return out


def _mean_present(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    return pd.concat([_numeric(df[col]) for col in available], axis=1).mean(axis=1, skipna=True)


def read_vivli_csv(source: str, csv_name: str) -> pd.DataFrame:
    """Read csv_name from either a directory/file path or a Vivli structured zip."""
    if source.endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            matches = [name for name in zf.namelist() if os.path.basename(name) == csv_name]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one {csv_name} in {source}, found {len(matches)}."
                )
            with zf.open(matches[0]) as fh:
                return pd.read_csv(fh, dtype=str).fillna("")

    path = source
    if os.path.isdir(source):
        path = os.path.join(source, csv_name)
    return pd.read_csv(path, dtype=str).fillna("")


def normalize_lie(value) -> float:
    """Map Vivli us_lie to FP binary label: 0 cephalic, 1 non-cephalic."""
    text = _clean_string(value).lower()
    if text in MISSING_STRINGS or text in {"98", "99", "-99", "4", "variable / na"}:
        return np.nan
    if text in {"1", "cephalic"}:
        return 0
    if text in {"2", "3", "5", "breech", "transverse", "oblique"}:
        return 1
    return np.nan


def normalize_nof(us_twin, multiple_gest) -> float:
    """Derive number of fetuses from CRF/SR twin fields."""
    twin = _numeric(pd.Series([us_twin])).iloc[0]
    multiple = _numeric(pd.Series([multiple_gest])).iloc[0]

    if pd.notna(twin) and twin in {1, 2, 3}:
        return twin
    if pd.notna(multiple):
        if multiple > 0:
            return 2
        return 1
    return np.nan


def build_exam_metadata(crf_df: pd.DataFrame, sr_df: pd.DataFrame) -> pd.DataFrame:
    """Create one exam-level metadata table from Vivli C3_CRF and C3_SR."""
    crf = crf_df.rename(columns={c: c.lower() for c in crf_df.columns}).copy()
    sr = sr_df.rename(columns={c: c.lower() for c in sr_df.columns}).copy()

    crf["StudyID"] = crf["pidscan"].map(_clean_string)
    crf["PID_crf"] = crf["pid"].map(_clean_string)
    sr["StudyID"] = sr["pidscan"].map(_clean_string)
    sr["PID_sr"] = sr["pid"].map(_clean_string)

    crf_features = pd.DataFrame({
        "StudyID": crf["StudyID"],
        "PID_crf": crf["PID_crf"],
        "ega_crf": _numeric(crf.get("ega", pd.Series(index=crf.index, dtype=object))),
        "us_twin": _numeric(crf.get("us_twin", pd.Series(index=crf.index, dtype=object))),
        "us_lie": crf.get("us_lie", pd.Series("", index=crf.index)),
        "EFW_crf": _numeric(crf.get("us_efw", pd.Series(index=crf.index, dtype=object))),
        "CRL_crf": _mean_present(crf, ["us_crla", "us_crlb"]),
        "BPD_crf": _mean_present(crf, ["us_bpa", "us_bpb"]),
        "HC_crf": _mean_present(crf, ["us_hca", "us_hcb"]),
        "AC_crf": _mean_present(crf, ["us_aca", "us_acb"]),
        "FL_crf": _mean_present(crf, ["us_fla", "us_flb"]),
    })

    sr_features = pd.DataFrame({
        "StudyID": sr["StudyID"],
        "PID_sr": sr["PID_sr"],
        "ega_sr": _numeric(sr.get("ega", pd.Series(index=sr.index, dtype=object))),
        "multiple_gest": _numeric(sr.get("multiple_gest", pd.Series(index=sr.index, dtype=object))),
        "CRL_sr": _numeric(sr.get("mean_crl", pd.Series(index=sr.index, dtype=object))),
        "BPD_sr": _numeric(sr.get("mean_bpd", pd.Series(index=sr.index, dtype=object))),
        "HC_sr": _numeric(sr.get("mean_hc", pd.Series(index=sr.index, dtype=object))),
        "AC_sr": _numeric(sr.get("mean_ac", pd.Series(index=sr.index, dtype=object))),
        "FL_sr": _numeric(sr.get("mean_fl", pd.Series(index=sr.index, dtype=object))),
    })

    crf_features = crf_features[crf_features["StudyID"] != ""].drop_duplicates("StudyID")
    sr_features = sr_features[sr_features["StudyID"] != ""].drop_duplicates("StudyID")
    exam = sr_features.merge(crf_features, on="StudyID", how="outer")

    exam["PID"] = _first_present(exam, ["PID_sr", "PID_crf"])
    exam["GA"] = _first_present(exam, ["ega_sr", "ega_crf"]).astype(float)
    exam["ega"] = exam["GA"]
    exam["CRL"] = _first_present(exam, ["CRL_sr", "CRL_crf"]).astype(float)
    exam["BPD"] = _first_present(exam, ["BPD_sr", "BPD_crf"]).astype(float)
    exam["HC"] = _first_present(exam, ["HC_sr", "HC_crf"]).astype(float)
    exam["AC"] = _first_present(exam, ["AC_sr", "AC_crf"]).astype(float)
    exam["FL"] = _first_present(exam, ["FL_sr", "FL_crf"]).astype(float)
    exam["EFW"] = exam["EFW_crf"].astype(float)
    exam["NOF"] = [
        normalize_nof(us_twin, multiple)
        for us_twin, multiple in zip(exam.get("us_twin", []), exam.get("multiple_gest", []))
    ]
    exam["lie"] = exam["us_lie"].apply(normalize_lie)

    keep_cols = [
        "PID",
        "StudyID",
        "GA",
        "ega",
        "CRL",
        "BPD",
        "HC",
        "AC",
        "FL",
        "EFW",
        "NOF",
        "lie",
        "us_twin",
        "multiple_gest",
        "us_lie",
        "ega_sr",
        "ega_crf",
    ]
    return exam[keep_cols].sort_values(["PID", "StudyID"]).reset_index(drop=True)


def build_curated_prototype(prototype_df: pd.DataFrame, exam_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Vivli exam metadata onto the preprocessed prototype."""
    prototype = prototype_df.copy()
    exam = exam_df.drop(columns=["PID"], errors="ignore").copy()
    curated = prototype.merge(exam, on="StudyID", how="left", suffixes=("", "_exam"))
    for col in ["GA", "ega", "CRL", "BPD", "HC", "AC", "FL", "EFW", "NOF", "lie"]:
        if col in curated.columns:
            curated[col] = pd.to_numeric(curated[col], errors="coerce")
    return curated


def _source_path(info: dict, key: str) -> str:
    path = info["input"][key]
    if os.path.isabs(path):
        return path
    root = info["input"].get("root_dir", "")
    return os.path.join(root, path)


def run_exam_mode(info: dict) -> None:
    crf_df = read_vivli_csv(_source_path(info, "structured_source"), info["input"]["crf_file"])
    sr_df = read_vivli_csv(_source_path(info, "structured_source"), info["input"]["sr_file"])
    exam = build_exam_metadata(crf_df, sr_df)
    out_dir = os.path.join(info["output"]["out_dir"], META_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, info["output"]["out_file"])
    exam.to_csv(out_path, index=False)
    print(f"Saved {exam.shape[0]} Vivli exam rows to {out_path}.")


def run_prototype_mode(info: dict) -> None:
    in_dir = os.path.join(info["input"]["in_dir"], META_DIR)
    prototype_path = os.path.join(in_dir, info["input"]["prototype_file"])
    exam_path = os.path.join(in_dir, info["input"]["exam_file"])
    prototype = pd.read_csv(prototype_path, dtype=str).fillna("")
    exam = pd.read_csv(exam_path)
    curated = build_curated_prototype(prototype, exam)
    out_dir = os.path.join(info["output"]["out_dir"], META_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, info["output"]["out_file"])
    curated.to_csv(out_path, index=False)
    print(f"Saved {curated.shape[0]} Vivli curated prototype rows to {out_path}.")


def main():
    parser = argparse.ArgumentParser(description="Curate Vivli FAMLI3 metadata for v9 tasks.")
    parser.add_argument("--yaml", required=True, type=str)
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        info = yaml.safe_load(f)

    mode = info.get("mode")
    if mode == "exam":
        run_exam_mode(info)
    elif mode == "prototype":
        run_prototype_mode(info)
    else:
        raise ValueError("mode must be one of: exam, prototype")


if __name__ == "__main__":
    main()

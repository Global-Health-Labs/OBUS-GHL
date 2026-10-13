"""
vivli_task_split_v9.py

Create patient-level train/val/test spreadsheets for Vivli task datasets.
"""
import argparse
import json
import os
import random
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from ghlobus.utilities.constants import GOOD_VIDEO_MSG, META_DIR
from ghlobus.utilities.normalization import BIOMETRY_LOG_STATS, LGA_MEAN, LGA_STD
from ghlobus.utilities.paths import output_file_base_name
from ghlobus.utilities.sweep_utils import VERTICAL_PLUS_TRANSVERSE, VERTICAL_SWEEP_TAGS


def _numeric(series: pd.Series) -> pd.Series:
    missing = {"", "nan", "NaN", "None", "none"}
    cleaned = series.astype(object).where(~series.astype(str).str.strip().isin(missing), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _take_evenly_spaced(values: list, count: int) -> list:
    if count <= 0:
        return []
    if count >= len(values):
        return list(values)
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[pos] for pos in positions]


def _split_pid_values(pid_values: pd.Series,
                      train_fraction: float,
                      val_fraction: float,
                      continuous: bool,
                      seed: int) -> Dict[str, set]:
    pid_values = pid_values.dropna().copy()

    if not continuous:
        rng = random.Random(seed)
        train, val, test = set(), set(), set()
        for _, group in pid_values.groupby(pid_values):
            values = list(group.index)
            rng.shuffle(values)
            n_train = int(len(values) * train_fraction)
            n_val = int(len(values) * val_fraction)
            train.update(values[:n_train])
            val.update(values[n_train:n_train + n_val])
            test.update(values[n_train + n_val:])
        return {"train": train, "val": val, "test": test}

    ordered = list(pid_values.sort_values().index)
    n_total = len(ordered)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_test = n_total - n_train - n_val
    test = set(_take_evenly_spaced(ordered, n_test))
    remaining = [pid for pid in ordered if pid not in test]
    val = set(_take_evenly_spaced(remaining, n_val))
    train = set(pid for pid in remaining if pid not in val)
    return {"train": train, "val": val, "test": test}


def _filter_tags(df: pd.DataFrame, directive: Optional[str]) -> pd.DataFrame:
    if directive is None:
        return df
    if directive == "vertical":
        return df[df["tag"].isin(VERTICAL_SWEEP_TAGS)]
    if directive == "transverse":
        return df[df["tag"].isin(VERTICAL_PLUS_TRANSVERSE)]
    raise ValueError(f"Unknown tag filter directive: {directive}")


def _add_outpath(df: pd.DataFrame, root_dir: str, data_dir: str, project: str) -> pd.DataFrame:
    df = df.copy()
    df["outpath"] = df.apply(
        lambda row: os.path.join(
            root_dir,
            data_dir,
            project,
            row["exam_dir"],
            f"{output_file_base_name(row['filename'])}.pt",
        ),
        axis=1,
    )
    return df


def _write_splits(df: pd.DataFrame,
                  split_pids: Dict[str, set],
                  out_path: str,
                  tag_filters: dict,
                  log_norm_cols: Iterable[str]) -> None:
    os.makedirs(out_path, exist_ok=True)
    log_norm_cols = list(log_norm_cols)
    split_frames = {
        name: _filter_tags(df[df["PID"].isin(split_pids[name])].copy(), tag_filters.get(name))
        for name in ["train", "val", "test"]
    }
    fixed_stats = {**BIOMETRY_LOG_STATS, "GA": (LGA_MEAN, LGA_STD)}
    norm_stats = {}
    # Validate the entire label contract before publishing any split CSV.
    for col in log_norm_cols:
        for name, frame in split_frames.items():
            values = frame[col].astype(float)
            if not (np.isfinite(values) & (values > 0)).all():
                raise ValueError(f"{name}: {col} requires finite, positive labels for log normalization.")
        train_values = np.log(split_frames["train"][col].astype(float))
        if train_values.empty:
            raise ValueError(f"No training labels available for {col} after filtering.")
        mean, std = fixed_stats.get(col, (train_values.mean(), train_values.std()))
        if pd.isna(std) or std == 0:
            std = 1.0
        norm_stats[col] = (mean, std)

    for split_name, split_df in split_frames.items():
        for col in log_norm_cols:
            log_col = f"log_{col}"
            z_col = f"z_log_{col.lower()}"
            values = np.log(split_df[col].astype(float))
            mean, std = norm_stats[col]
            normalized = (values - mean) / std
            # Existing EFW configs consume standardized log_AC/log_FL/log_HC/log_BPD.
            split_df[f"raw_log_{col}"] = values
            split_df[log_col] = normalized if col in BIOMETRY_LOG_STATS else values
            split_df[z_col] = normalized
        if "GA" in log_norm_cols:
            # Compatibility names required by the existing GA datamodule/callbacks.
            # These contain Vivli GA; the alias does not assert BOE provenance.
            split_df["ga_boe"] = split_df["GA"]
            split_df["log_ga_boe"] = split_df["raw_log_GA"]
        split_df.sort_values(["PID", "exam_dir", "filename"]).to_csv(
            os.path.join(out_path, f"{split_name}.csv"),
            index=False,
        )
        print(f"Wrote {split_df.shape[0]} rows to {split_name}.csv.")

    metadata = {
        col: {"mean": float(mean), "std": float(std),
              "source": ("legacy_v9" if col in BIOMETRY_LOG_STATS else
                         "legacy_v4" if col == "GA" else "train"),
              "normalized_column": f"log_{col}" if col in BIOMETRY_LOG_STATS else f"z_log_{col.lower()}"}
        for col, (mean, std) in norm_stats.items()
    }
    with open(os.path.join(out_path, "normalization.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Create Vivli patient-level task splits.")
    parser.add_argument("--yaml", required=True, type=str)
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        info = yaml.safe_load(f)

    seed = info.get("seed", 42)
    np.random.seed(seed)
    random.seed(seed)

    in_dir = os.path.join(info["input"]["root_dir"], META_DIR)
    df = pd.read_csv(os.path.join(in_dir, info["input"]["file"]))
    df = df[df["fail_reason"] == GOOD_VIDEO_MSG].copy()
    df = _add_outpath(
        df,
        root_dir=info["input"]["root_dir"],
        data_dir=info["input"].get("data_dir", "data"),
        project=info["input"].get("project", "VIVLI_FAMLI3"),
    )

    task = info["task"]["name"]
    if task == "GA":
        df["GA"] = _numeric(df["GA"])
        df = df[df["GA"].notna() & (df["GA"] > 0)].copy()
        patient_values = df.groupby("PID")["GA"].mean()
        continuous = True
        log_norm_cols = ["GA"]
    elif task == "FP":
        df["lie"] = _numeric(df["lie"])
        df = df[df["lie"].isin([0, 1])].copy()
        patient_values = df.groupby("PID")["lie"].max()
        continuous = False
        log_norm_cols = []
    elif task == "EFW":
        df["EFW"] = _numeric(df["EFW"])
        df = df[df["EFW"].notna() & (df["EFW"] > 0)].copy()
        patient_values = df.groupby("PID")["EFW"].max()
        continuous = True
        log_norm_cols = [c for c in ["AC", "HC", "BPD", "FL", "EFW", "EFW_hadlock", "GA", "GA_hadlock"] if c in df]
    elif task == "TWIN":
        df["TWIN"] = _numeric(df["TWIN"])
        df = df[df["TWIN"].isin([0, 1])].copy()
        patient_values = df.groupby("PID")["TWIN"].max()
        continuous = False
        log_norm_cols = []
    else:
        raise ValueError(f"Unsupported task: {task}")

    split_pids = _split_pid_values(
        patient_values,
        train_fraction=info["splits"]["train"]["fraction"],
        val_fraction=info["splits"]["val"]["fraction"],
        continuous=continuous,
        seed=seed,
    )
    out_path = os.path.join(
        info["output"]["root_dir"],
        info["output"]["folder"],
        info["output"]["distribution"],
    )
    tag_filters = {k: v.get("tags") for k, v in info["splits"].items()}
    _write_splits(df, split_pids, out_path, tag_filters, log_norm_cols)


if __name__ == "__main__":
    main()

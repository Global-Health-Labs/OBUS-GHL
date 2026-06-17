"""
vivli_task_split_v9.py

Create patient-level train/val/test spreadsheets for Vivli task datasets.
"""
import argparse
import os
import random
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from ghlobus.utilities.constants import GOOD_VIDEO_MSG, META_DIR
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
            f"{os.path.splitext(row['filename'])[0]}.pt",
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
    norm_stats = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["PID"].isin(split_pids[split_name])].copy()
        split_df = _filter_tags(split_df, tag_filters.get(split_name))
        for col in log_norm_cols:
            log_col = f"log_{col}"
            z_col = f"z_log_{col.lower()}" if col != "GA" else "z_log_ga"
            values = np.log(split_df[col].astype(float))
            if split_name == "train":
                norm_stats[col] = (values.mean(), values.std())
            mean, std = norm_stats[col]
            if pd.isna(std) or std == 0:
                std = 1
            split_df[log_col] = values
            split_df[z_col] = (values - mean) / std
        split_df.sort_values(["PID", "exam_dir", "filename"]).to_csv(
            os.path.join(out_path, f"{split_name}.csv"),
            index=False,
        )
        print(f"Wrote {split_df.shape[0]} rows to {split_name}.csv.")


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

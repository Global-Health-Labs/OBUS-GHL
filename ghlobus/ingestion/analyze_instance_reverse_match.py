#!/usr/bin/env python3
"""Match instance-table rows back to the provided cohort zip files.

This is a reverse analysis:
- rows/files in C3_INSTANCE_TABLE.csv that are present in the provided cohort zips
- rows/files in C3_INSTANCE_TABLE.csv that are not present in the provided cohort zips

Optional `.dcm` suffix normalization is applied so `UID` and `UID.dcm`
are treated as equivalent.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_STRUCTURED_ZIP = "Vivli-UNC-FAMLI3Twins-StructuredData-19-Feb-2026.zip"
REPORT_MD_NAME = "instance_reverse_match_report.md"
REPORT_JSON_NAME = "instance_reverse_match_report.json"
MATCHED_ROWS_CSV_NAME = "instance_rows_found_in_provided_cohorts.csv"
MISSING_ROWS_CSV_NAME = "instance_rows_missing_from_provided_cohorts.csv"
DIFF_PID_CSV_NAME = "instance_rows_found_under_different_pidscan.csv"


STUDY_DIR_RE = re.compile(r"^(FA3-[^_]+)_(\d{8})_(\d{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the structured-data zip and cohort zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report outputs. Defaults to --data-dir.",
    )
    parser.add_argument(
        "--structured-zip",
        type=Path,
        default=None,
        help="Optional path to the structured-data zip. Defaults to auto-detect in --data-dir.",
    )
    parser.add_argument(
        "--cohort-zip",
        action="append",
        default=[],
        help="Optional cohort zip path or basename. Repeat to pin a specific set of cohort zips.",
    )
    return parser.parse_args()


def resolve_path(value: Path, data_dir: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path.resolve()


def resolve_structured_zip(data_dir: Path, structured_zip: Path | None) -> Path:
    if structured_zip is not None:
        path = resolve_path(structured_zip, data_dir)
        if not path.exists():
            raise FileNotFoundError(f"Structured zip not found: {path}")
        return path

    preferred = data_dir / DEFAULT_STRUCTURED_ZIP
    if preferred.exists():
        return preferred

    zip_paths = sorted(path for path in data_dir.glob("*.zip") if path.is_file())
    structured_candidates = [path for path in zip_paths if "structured" in path.name.lower()]
    if len(structured_candidates) == 1:
        return structured_candidates[0]

    non_cohort_candidates = [
        path for path in zip_paths if not path.name.lower().startswith("cohort")
    ]
    if len(non_cohort_candidates) == 1:
        return non_cohort_candidates[0]

    raise FileNotFoundError(
        "Unable to uniquely determine the structured-data zip. "
        "Pass --structured-zip explicitly."
    )


def resolve_cohort_zips(data_dir: Path, cohort_zips: list[str]) -> list[Path]:
    if cohort_zips:
        resolved = [resolve_path(Path(value), data_dir) for value in cohort_zips]
    else:
        resolved = sorted(
            path
            for path in data_dir.glob("*.zip")
            if path.is_file() and path.name.lower().startswith("cohort")
        )

    if not resolved:
        raise FileNotFoundError("No cohort zip files found. Pass --cohort-zip or stage Cohort*.zip files.")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cohort zip files not found: {missing}")

    return resolved


def normalize_filename(name: str) -> str:
    base = Path(name).name
    if base.lower().endswith(".dcm"):
        return base[:-4]
    return base


def normalize_studytime(value: str) -> str:
    text = (value or "").strip()
    whole = text.split(".", 1)[0]
    digits = "".join(ch for ch in whole if ch.isdigit())
    return digits.zfill(6) if digits else ""


def open_csv_from_zip(zip_path: Path, suffix: str):
    with zipfile.ZipFile(zip_path) as zf:
        target = next(name for name in zf.namelist() if name.endswith(suffix))
        with zf.open(target, "r") as handle:
            reader = csv.DictReader(line.decode("utf-8-sig", errors="replace") for line in handle)
            yield from reader


def scan_raw_files(cohort_zips: list[Path]):
    raw_by_pidscan: dict[str, set[str]] = defaultdict(set)
    raw_any: dict[str, set[str]] = defaultdict(set)
    file_sources: dict[tuple[str, str], list[str]] = defaultdict(list)

    for zip_path in cohort_zips:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                stripped = name.strip("/")
                if not stripped or stripped.endswith("/"):
                    continue
                parts = stripped.split("/")
                if len(parts) < 3:
                    continue
                study_dir = parts[-2]
                match = STUDY_DIR_RE.match(study_dir)
                if not match:
                    continue
                pidscan = match.group(1)
                filename = parts[-1]
                normalized = normalize_filename(filename)
                raw_by_pidscan[pidscan].add(normalized)
                raw_any[normalized].add(pidscan)
                file_sources[(pidscan, normalized)].append(f"{zip_path.name}:{stripped}")

    return raw_by_pidscan, raw_any, file_sources


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or data_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    structured_zip = resolve_structured_zip(data_dir, args.structured_zip)
    cohort_zips = resolve_cohort_zips(data_dir, args.cohort_zip)
    report_md = output_dir / REPORT_MD_NAME
    report_json = output_dir / REPORT_JSON_NAME
    matched_rows_csv = output_dir / MATCHED_ROWS_CSV_NAME
    missing_rows_csv = output_dir / MISSING_ROWS_CSV_NAME
    diff_pid_csv = output_dir / DIFF_PID_CSV_NAME

    raw_by_pidscan, raw_any, file_sources = scan_raw_files(cohort_zips)

    matched_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    different_pid_rows: list[dict[str, object]] = []
    row_status_counter: Counter[str] = Counter()
    ext_counter = Counter()
    matched_ext_counter = Counter()
    missing_ext_counter = Counter()

    unique_keys = set()
    unique_matched_keys = set()
    unique_missing_keys = set()
    unique_diff_pid_keys = set()

    for row in open_csv_from_zip(structured_zip, "C3_INSTANCE_TABLE.csv"):
        pidscan = (row.get("pidscan") or "").strip()
        studydate = (row.get("studydate") or "").strip()
        studytime = normalize_studytime(row.get("studytime") or "")
        file_name = (row.get("file") or "").strip()
        normalized_file = normalize_filename(file_name)
        extension = Path(file_name).suffix.lower() or "<no_ext>"

        row_out = {
            "pidscan": pidscan,
            "studydate": studydate,
            "studytime": studytime,
            "file": file_name,
            "normalized_file": normalized_file,
            "extension": extension,
        }

        unique_key = (pidscan, normalized_file)
        unique_keys.add(unique_key)
        ext_counter[extension] += 1

        if normalized_file in raw_by_pidscan.get(pidscan, set()):
            row_status_counter["found_same_pidscan"] += 1
            matched_ext_counter[extension] += 1
            unique_matched_keys.add(unique_key)
            row_out["raw_locations"] = "; ".join(file_sources.get((pidscan, normalized_file), [])[:10])
            matched_rows.append(row_out)
        elif normalized_file in raw_any:
            row_status_counter["found_different_pidscan_only"] += 1
            unique_diff_pid_keys.add(unique_key)
            row_out["raw_pidscans"] = "; ".join(sorted(raw_any[normalized_file]))
            different_pid_rows.append(row_out)
        else:
            row_status_counter["not_found_in_provided_cohorts"] += 1
            missing_ext_counter[extension] += 1
            unique_missing_keys.add(unique_key)
            missing_rows.append(row_out)

    report = {
        "row_level_summary": {
            "total_instance_rows": sum(row_status_counter.values()),
            "found_same_pidscan": row_status_counter["found_same_pidscan"],
            "found_different_pidscan_only": row_status_counter["found_different_pidscan_only"],
            "not_found_in_provided_cohorts": row_status_counter["not_found_in_provided_cohorts"],
            "by_extension_total": dict(sorted(ext_counter.items())),
            "by_extension_found_same_pidscan": dict(sorted(matched_ext_counter.items())),
            "by_extension_not_found": dict(sorted(missing_ext_counter.items())),
        },
        "unique_file_level_summary": {
            "total_unique_pidscan_file_pairs": len(unique_keys),
            "found_same_pidscan": len(unique_matched_keys),
            "found_different_pidscan_only": len(unique_diff_pid_keys),
            "not_found_in_provided_cohorts": len(unique_missing_keys),
        },
        "sample_found_different_pidscan_only": different_pid_rows[:50],
    }

    write_csv(matched_rows_csv, matched_rows)
    write_csv(missing_rows_csv, missing_rows)
    write_csv(diff_pid_csv, different_pid_rows)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Instance Table Reverse Match Report",
        "",
        "This report checks which rows in `C3_INSTANCE_TABLE.csv` have a corresponding raw file in the provided cohort zip files.",
        "",
        "## Row-Level Summary",
        "",
        f"- Total instance-table rows: **{report['row_level_summary']['total_instance_rows']}**",
        f"- Found in provided cohort zips under the same `pidscan`: **{report['row_level_summary']['found_same_pidscan']}**",
        f"- Found in provided cohort zips only under a different `pidscan`: **{report['row_level_summary']['found_different_pidscan_only']}**",
        f"- Not found in the provided cohort zips: **{report['row_level_summary']['not_found_in_provided_cohorts']}**",
        "",
        "## Unique File-Level Summary",
        "",
        f"- Total unique `pidscan + normalized_file` pairs in the instance table: **{report['unique_file_level_summary']['total_unique_pidscan_file_pairs']}**",
        f"- Found in provided cohort zips under the same `pidscan`: **{report['unique_file_level_summary']['found_same_pidscan']}**",
        f"- Found in provided cohort zips only under a different `pidscan`: **{report['unique_file_level_summary']['found_different_pidscan_only']}**",
        f"- Not found in the provided cohort zips: **{report['unique_file_level_summary']['not_found_in_provided_cohorts']}**",
        "",
        "## Extension Breakdown",
        "",
    ]

    for ext, total in sorted(ext_counter.items()):
        found = matched_ext_counter.get(ext, 0)
        missing = missing_ext_counter.get(ext, 0)
        md_lines.append(f"- `{ext}`: total {total}, found {found}, not found {missing}")

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Matching uses `pidscan + normalized filename` where `UID` and `UID.dcm` are treated as equivalent.",
            "- A row counted as `found_different_pidscan_only` means the normalized filename appears in the provided cohort zips, but only under a different `pidscan` than the instance-table row.",
            "",
            "## Output Files",
            "",
            f"- `{matched_rows_csv.name}`",
            f"- `{missing_rows_csv.name}`",
            f"- `{diff_pid_csv.name}`",
            f"- `{report_json.name}`",
        ]
    )

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Structured zip: {structured_zip}")
    print(f"Cohort zips: {[path.name for path in cohort_zips]}")
    print(f"Wrote {matched_rows_csv}")
    print(f"Wrote {missing_rows_csv}")
    print(f"Wrote {diff_pid_csv}")
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

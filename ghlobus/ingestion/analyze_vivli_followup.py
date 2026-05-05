#!/usr/bin/env python3
"""Follow-up comparisons for unmatched cohort studies/files.

This script compares two recovery strategies:
1. Link still-unreferenced cohort studies to CRF/SR rows by pidscan + visit date.
2. Re-check study-referenced but file-missing cohort files using a flexible
   DICOM filename normalization that strips an optional `.dcm` suffix.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_STRUCTURED_ZIP = "Vivli-UNC-FAMLI3Twins-StructuredData-19-Feb-2026.zip"
UNMATCHED_STUDIES_CSV_NAME = "cohort_unmatched_studies.csv"
UNMATCHED_FILES_CSV_NAME = "cohort_unmatched_files.csv"
FOLLOWUP_JSON_NAME = "cohort_followup_report.json"
FOLLOWUP_MD_NAME = "cohort_followup_report.md"
CRF_SR_LINKAGE_CSV_NAME = "cohort_unmatched_studies_crf_sr_linkage.csv"


MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the structured-data zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory containing the main alignment outputs and destination for follow-up outputs. Defaults to --data-dir.",
    )
    parser.add_argument(
        "--structured-zip",
        type=Path,
        default=None,
        help="Optional path to the structured-data zip. Defaults to auto-detect in --data-dir.",
    )
    return parser.parse_args()


def resolve_path(value: Path, base_dir: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = base_dir / path
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


def open_csv_from_zip(zip_path: Path, suffix: str):
    with zipfile.ZipFile(zip_path) as zf:
        target = next(name for name in zf.namelist() if name.endswith(suffix))
        with zf.open(target, "r") as handle:
            reader = csv.DictReader(line.decode("utf-8-sig", errors="replace") for line in handle)
            yield from reader


def normalize_crf_date(value: str) -> str:
    text = (value or "").strip().upper()
    if len(text) == 9 and text[2:5] in MONTHS:
        return f"{text[5:9]}{MONTHS[text[2:5]]}{text[:2]}"
    return ""


def normalize_sr_date(value: str) -> str:
    text = (value or "").strip()
    return text.replace("-", "") if text else ""


def normalize_studytime(value: str) -> str:
    text = (value or "").strip()
    whole = text.split(".", 1)[0]
    digits = "".join(ch for ch in whole if ch.isdigit())
    return digits.zfill(6) if digits else ""


def strip_optional_dcm(filename: str) -> str:
    return filename[:-4] if filename.lower().endswith(".dcm") else filename


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    unmatched_studies_csv = output_dir / UNMATCHED_STUDIES_CSV_NAME
    unmatched_files_csv = output_dir / UNMATCHED_FILES_CSV_NAME
    followup_json = output_dir / FOLLOWUP_JSON_NAME
    followup_md = output_dir / FOLLOWUP_MD_NAME
    crf_sr_linkage_csv = output_dir / CRF_SR_LINKAGE_CSV_NAME

    unmatched_studies = read_csv(unmatched_studies_csv)
    unmatched_files = read_csv(unmatched_files_csv)

    crf_rows_by_pid: dict[str, list[dict[str, str]]] = defaultdict(list)
    sr_rows_by_pid: dict[str, list[dict[str, str]]] = defaultdict(list)
    crf_dates = set()
    sr_dates = set()

    for row in open_csv_from_zip(structured_zip, "C3_CRF.csv"):
        pidscan = (row.get("pidscan") or "").strip()
        date = normalize_crf_date(row.get("visit_date_date") or "")
        if pidscan:
            crf_rows_by_pid[pidscan].append(row)
        if pidscan and date:
            crf_dates.add((pidscan, date))

    for row in open_csv_from_zip(structured_zip, "C3_SR.csv"):
        pidscan = (row.get("pidscan") or "").strip()
        date = normalize_sr_date(row.get("visit_date") or "")
        if pidscan:
            sr_rows_by_pid[pidscan].append(row)
        if pidscan and date:
            sr_dates.add((pidscan, date))

    linkage_rows: list[dict[str, object]] = []
    linkage_counter: Counter[str] = Counter()
    for row in unmatched_studies:
        pidscan = row["pidscan"]
        studydate = row["studydate"]
        in_crf_on_date = (pidscan, studydate) in crf_dates
        in_sr_on_date = (pidscan, studydate) in sr_dates
        in_crf_any = pidscan in crf_rows_by_pid
        in_sr_any = pidscan in sr_rows_by_pid

        if in_crf_on_date and in_sr_on_date:
            linkage_status = "exact_date_in_both_crf_and_sr"
        elif in_crf_on_date or in_sr_on_date:
            linkage_status = "exact_date_in_one_table_only"
        elif in_crf_any and in_sr_any:
            linkage_status = "pidscan_in_both_tables_but_date_mismatch"
        elif in_crf_any or in_sr_any:
            linkage_status = "pidscan_in_one_table_only_and_date_mismatch"
        else:
            linkage_status = "absent_from_both_crf_and_sr"

        linkage_counter[linkage_status] += 1
        linkage_rows.append(
            {
                **row,
                "crf_exact_date_match": in_crf_on_date,
                "sr_exact_date_match": in_sr_on_date,
                "crf_pidscan_present": in_crf_any,
                "sr_pidscan_present": in_sr_any,
                "linkage_status": linkage_status,
            }
        )

    instance_files_by_study: dict[str, set[str]] = defaultdict(set)
    for row in open_csv_from_zip(structured_zip, "C3_INSTANCE_TABLE.csv"):
        pidscan = (row.get("pidscan") or "").strip()
        studydate = (row.get("studydate") or "").strip()
        studytime = normalize_studytime(row.get("studytime") or "")
        file_name = (row.get("file") or "").strip()
        if pidscan and studydate and studytime and file_name:
            study_key = f"{pidscan}_{studydate}_{studytime}"
            instance_files_by_study[study_key].add(file_name)

    ref_missing_files = [
        row
        for row in unmatched_files
        if row["status"] == "study_reference_found_but_file_missing"
    ]
    file_match_counter: Counter[str] = Counter()
    strip_match_examples: list[dict[str, str]] = []
    for row in ref_missing_files:
        study_key = row["normalized_instance_study_key"]
        filename = row["filename"]
        candidates = instance_files_by_study.get(study_key, set())
        stripped = strip_optional_dcm(filename)
        match = next((candidate for candidate in candidates if strip_optional_dcm(candidate) == stripped), "")
        if match:
            file_match_counter["extension_only_match"] += 1
            if len(strip_match_examples) < 20:
                strip_match_examples.append(
                    {
                        "study_key": study_key,
                        "cohort_filename": filename,
                        "instance_filename": match,
                    }
                )
        else:
            file_match_counter["no_match_after_optional_dcm_normalization"] += 1

    report = {
        "crf_sr_linkage": {
            "total_unmatched_studies": len(unmatched_studies),
            "linkage_status_counts": dict(sorted(linkage_counter.items())),
            "sample_exact_date_matches": [
                row["study_key"]
                for row in linkage_rows
                if row["linkage_status"] == "exact_date_in_both_crf_and_sr"
            ][:20],
            "sample_absent_from_both": [
                row["study_key"]
                for row in linkage_rows
                if row["linkage_status"] == "absent_from_both_crf_and_sr"
            ][:20],
        },
        "flexible_file_matching": {
            "total_study_reference_found_but_file_missing": len(ref_missing_files),
            "match_status_counts": dict(sorted(file_match_counter.items())),
            "strip_match_examples": strip_match_examples,
        },
    }

    write_csv(crf_sr_linkage_csv, linkage_rows)
    followup_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Cohort Follow-Up Comparison",
        "",
        "## CRF/SR Linkage For Unmatched Studies",
        "",
        f"- Unmatched study folders from the main report: **{len(unmatched_studies)}**",
    ]
    for status, count in sorted(linkage_counter.items()):
        md_lines.append(f"- `{status}`: {count}")

    md_lines.extend(
        [
            "",
            "## Flexible File Matching In Referenceable Studies",
            "",
            f"- Files flagged as `study_reference_found_but_file_missing`: **{len(ref_missing_files)}**",
        ]
    )
    for status, count in sorted(file_match_counter.items()):
        md_lines.append(f"- `{status}`: {count}")

    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The CRF/SR approach can confirm whether an unmatched study folder still corresponds to a documented visit, but it does not provide file-level references.",
            "- The flexible file check tests whether file misses are caused by naming differences only.",
            "- In this dataset, the dominant flexible file pattern is an optional `.dcm` suffix difference rather than a genuinely different instance UID.",
            "",
            "## Output Files",
            "",
            f"- `{crf_sr_linkage_csv.name}`",
            f"- `{followup_json.name}`",
        ]
    )

    followup_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Structured zip: {structured_zip}")
    print(f"Wrote {crf_sr_linkage_csv}")
    print(f"Wrote {followup_json}")
    print(f"Wrote {followup_md}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

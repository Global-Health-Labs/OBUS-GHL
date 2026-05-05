#!/usr/bin/env python3
"""Compare cohort archive contents against structured instance tables.

This script scans the cohort zip files and the structured data zip without
fully extracting them. It reports whether cohort study folders and their
files are referenceable in the contributor CSV tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_STRUCTURED_ZIP = "Vivli-UNC-FAMLI3Twins-StructuredData-19-Feb-2026.zip"
REPORT_MD_NAME = "cohort_alignment_report.md"
REPORT_JSON_NAME = "cohort_alignment_report.json"
UNMATCHED_STUDIES_CSV_NAME = "cohort_unmatched_studies.csv"
UNMATCHED_FILES_CSV_NAME = "cohort_unmatched_files.csv"


STUDY_DIR_RE = re.compile(r"^(FA3-[^_]+)_(\d{8})_(\d{6})$")
UID_LIKE_RE = re.compile(r"^\d+(?:\.\d+)+$")


@dataclass(frozen=True)
class StudyFolder:
    cohort_zip: str
    internal_dir: str
    pidscan: str
    studydate: str
    studytime: str
    file_count: int
    file_extensions: tuple[str, ...]

    @property
    def study_key(self) -> str:
        return f"{self.pidscan}_{self.studydate}_{self.studytime}"


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


def normalize_studytime(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    whole = text.split(".", 1)[0]
    digits = "".join(ch for ch in whole if ch.isdigit())
    return digits.zfill(6) if digits else ""


def detect_extension(filename: str) -> str:
    name_only = Path(filename).name
    if UID_LIKE_RE.match(name_only):
        return ".dcm"
    lower = filename.lower()
    for suffix in (".dcm", ".mp4", ".png", ".jpg", ".jpeg", ".avi", ".mov", ".gif", ".bmp", ".tif", ".tiff"):
        if lower.endswith(suffix):
            return suffix
    return Path(filename).suffix.lower() or "<no_ext>"


def open_csv_from_zip(zip_path: Path, suffix: str) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        target = next(name for name in zf.namelist() if name.endswith(suffix))
        with zf.open(target, "r") as handle:
            text_handle = (line.decode("utf-8-sig", errors="replace") for line in handle)
            reader = csv.DictReader(text_handle)
            yield from reader


def load_instance_table(structured_zip: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    by_study_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_file_key: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in open_csv_from_zip(structured_zip, "C3_INSTANCE_TABLE.csv"):
        pidscan = (row.get("pidscan") or "").strip()
        studydate = (row.get("studydate") or "").strip()
        studytime = normalize_studytime(row.get("studytime") or "")
        file_name = Path((row.get("file") or "").strip()).name

        study_key = f"{pidscan}_{studydate}_{studytime}"
        if pidscan and studydate and studytime:
            by_study_key[study_key].append(row)

        if pidscan and studydate and studytime and file_name:
            file_key = f"{study_key}/{file_name}"
            by_file_key[file_key].append(row)

    return by_study_key, by_file_key


def build_time_lookup(by_study_key: dict[str, list[dict[str, str]]]) -> dict[tuple[str, str, str], list[str]]:
    lookup: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for study_key in by_study_key:
        pidscan, studydate, studytime = study_key.rsplit("_", 2)
        lookup[(pidscan, studydate, studytime[2:])].append(study_key)
    return lookup


def load_pidscan_sets(structured_zip: Path) -> tuple[set[str], set[str]]:
    crf_pidscans = {
        (row.get("pidscan") or "").strip()
        for row in open_csv_from_zip(structured_zip, "C3_CRF.csv")
        if (row.get("pidscan") or "").strip()
    }
    sr_pidscans = {
        (row.get("pidscan") or "").strip()
        for row in open_csv_from_zip(structured_zip, "C3_SR.csv")
        if (row.get("pidscan") or "").strip()
    }
    return crf_pidscans, sr_pidscans


def scan_cohort_zip(zip_path: Path) -> tuple[list[StudyFolder], list[dict[str, str]]]:
    file_lists: dict[str, list[str]] = defaultdict(list)

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            stripped = name.strip("/")
            if not stripped or stripped.endswith("/"):
                continue

            parts = stripped.split("/")
            if len(parts) < 3:
                continue

            parent = "/".join(parts[:-1])
            filename = parts[-1]
            file_lists[parent].append(filename)

    studies: list[StudyFolder] = []
    files: list[dict[str, str]] = []

    for internal_dir, filenames in sorted(file_lists.items()):
        leaf = internal_dir.rsplit("/", 1)[-1]
        match = STUDY_DIR_RE.match(leaf)
        if not match:
            continue

        pidscan, studydate, studytime = match.groups()
        extensions = tuple(sorted({detect_extension(name) for name in filenames}))
        study = StudyFolder(
            cohort_zip=zip_path.name,
            internal_dir=internal_dir,
            pidscan=pidscan,
            studydate=studydate,
            studytime=studytime,
            file_count=len(filenames),
            file_extensions=extensions,
        )
        studies.append(study)

        for filename in filenames:
            files.append(
                {
                    "cohort_zip": zip_path.name,
                    "internal_dir": internal_dir,
                    "study_key": study.study_key,
                    "pidscan": pidscan,
                    "studydate": studydate,
                    "studytime": studytime,
                    "filename": filename,
                    "extension": detect_extension(filename),
                }
            )

    return studies, files


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_extensions(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counter = Counter(row["extension"] for row in rows)
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or data_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    structured_zip = resolve_structured_zip(data_dir, args.structured_zip)
    cohort_zips = resolve_cohort_zips(data_dir, args.cohort_zip)

    report_md = output_dir / REPORT_MD_NAME
    report_json = output_dir / REPORT_JSON_NAME
    unmatched_studies_csv = output_dir / UNMATCHED_STUDIES_CSV_NAME
    unmatched_files_csv = output_dir / UNMATCHED_FILES_CSV_NAME

    by_study_key, by_file_key = load_instance_table(structured_zip)
    time_lookup = build_time_lookup(by_study_key)
    crf_pidscans, sr_pidscans = load_pidscan_sets(structured_zip)

    all_studies: list[StudyFolder] = []
    all_files: list[dict[str, str]] = []
    for zip_path in cohort_zips:
        studies, files = scan_cohort_zip(zip_path)
        all_studies.extend(studies)
        all_files.extend(files)

    unmatched_studies: list[dict[str, object]] = []
    exact_study_match_count = 0
    shifted_study_match_count = 0
    normalized_study_match_count = 0
    study_status_counter: Counter[str] = Counter()
    time_shift_counter: Counter[str] = Counter()
    normalized_study_keys: dict[str, str] = {}
    for study in all_studies:
        exact_rows = by_study_key.get(study.study_key, [])
        shifted_candidates = time_lookup.get((study.pidscan, study.studydate, study.studytime[2:]), [])
        shifted_candidates = [candidate for candidate in shifted_candidates if candidate != study.study_key]

        if exact_rows:
            exact_study_match_count += 1
            normalized_study_match_count += 1
            normalized_study_keys[study.study_key] = study.study_key
            study_status_counter["exact"] += 1
            continue

        if len(shifted_candidates) == 1:
            shifted_study_match_count += 1
            normalized_study_match_count += 1
            normalized_study_keys[study.study_key] = shifted_candidates[0]
            study_status_counter["time_shifted_unique"] += 1
            _, _, candidate_time = shifted_candidates[0].rsplit("_", 2)
            source_dt = datetime.strptime(f"{study.studydate}{study.studytime}", "%Y%m%d%H%M%S")
            candidate_dt = datetime.strptime(f"{study.studydate}{candidate_time}", "%Y%m%d%H%M%S")
            delta_hours = int((source_dt - candidate_dt).total_seconds() // 3600)
            time_shift_counter[f"{delta_hours:+d}h"] += 1
            continue

        if len(shifted_candidates) > 1:
            status = "time_shifted_ambiguous"
            candidate_keys = shifted_candidates
        else:
            status = "unreferenced"
            candidate_keys = []

        study_status_counter[status] += 1
        unmatched_studies.append(
            {
                "cohort_zip": study.cohort_zip,
                "internal_dir": study.internal_dir,
                "pidscan": study.pidscan,
                "studydate": study.studydate,
                "studytime": study.studytime,
                "study_key": study.study_key,
                "file_count": study.file_count,
                "file_extensions": ";".join(study.file_extensions),
                "status": status,
                "candidate_instance_study_keys": ";".join(candidate_keys),
                "pidscan_in_crf": study.pidscan in crf_pidscans,
                "pidscan_in_sr": study.pidscan in sr_pidscans,
            }
        )

    unmatched_files: list[dict[str, object]] = []
    exact_file_match_count = 0
    shifted_file_match_count = 0
    normalized_file_match_count = 0
    file_status_counter: Counter[str] = Counter()
    for row in all_files:
        file_key = f"{row['study_key']}/{row['filename']}"
        normalized_study_key = normalized_study_keys.get(row["study_key"])
        normalized_file_key = f"{normalized_study_key}/{row['filename']}" if normalized_study_key else ""

        if by_file_key.get(file_key):
            exact_file_match_count += 1
            normalized_file_match_count += 1
            file_status_counter["exact"] += 1
            continue

        if normalized_study_key and normalized_study_key != row["study_key"] and by_file_key.get(normalized_file_key):
            shifted_file_match_count += 1
            normalized_file_match_count += 1
            file_status_counter["time_shifted_unique"] += 1
            continue

        if normalized_study_key:
            status = "study_reference_found_but_file_missing"
        else:
            status = "study_unreferenced_or_ambiguous"
        file_status_counter[status] += 1

        unmatched_files.append(
            {
                "cohort_zip": row["cohort_zip"],
                "internal_dir": row["internal_dir"],
                "pidscan": row["pidscan"],
                "studydate": row["studydate"],
                "studytime": row["studytime"],
                "study_key": row["study_key"],
                "filename": row["filename"],
                "extension": row["extension"],
                "status": status,
                "normalized_instance_study_key": normalized_study_key or "",
                "study_key_exists_in_instance_table": row["study_key"] in by_study_key,
                "pidscan_in_crf": row["pidscan"] in crf_pidscans,
                "pidscan_in_sr": row["pidscan"] in sr_pidscans,
            }
        )

    unmatched_studies_by_zip = Counter(row["cohort_zip"] for row in unmatched_studies)
    unmatched_files_by_zip = Counter(row["cohort_zip"] for row in unmatched_files)

    report = {
        "structured_zip": structured_zip.name,
        "cohort_zips": [path.name for path in cohort_zips],
        "summary": {
            "total_study_folders": len(all_studies),
            "exact_study_matches": exact_study_match_count,
            "time_shifted_study_matches": shifted_study_match_count,
            "referenceable_study_folders": normalized_study_match_count,
            "unmatched_study_folders": len(unmatched_studies),
            "total_files": len(all_files),
            "exact_file_matches": exact_file_match_count,
            "time_shifted_file_matches": shifted_file_match_count,
            "referenceable_files": normalized_file_match_count,
            "unmatched_files": len(unmatched_files),
            "study_match_status_counts": dict(sorted(study_status_counter.items())),
            "time_shift_distribution_hours": dict(sorted(time_shift_counter.items())),
            "file_match_status_counts": dict(sorted(file_status_counter.items())),
            "cohort_file_extensions": summarize_extensions(all_files),
            "unmatched_file_extensions": summarize_extensions(unmatched_files),
        },
        "unmatched_studies_by_zip": dict(sorted(unmatched_studies_by_zip.items())),
        "unmatched_files_by_zip": dict(sorted(unmatched_files_by_zip.items())),
        "sample_unmatched_studies": unmatched_studies[:50],
        "sample_unmatched_files": unmatched_files[:100],
    }

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(unmatched_studies_csv, unmatched_studies)
    write_csv(unmatched_files_csv, unmatched_files)

    md_lines = [
        "# Cohort Alignment Report",
        "",
        f"Structured zip: `{structured_zip.name}`",
        f"Cohort zips: {', '.join(f'`{path.name}`' for path in cohort_zips)}",
        "",
        "## Summary",
        "",
        f"- Study folders found in cohort zips: **{len(all_studies)}**",
        f"- Exact study-folder matches to `C3_INSTANCE_TABLE.csv` by `pidscan + studydate + studytime`: **{exact_study_match_count}**",
        f"- Time-shifted study-folder matches using same `pidscan + studydate + minute:second`: **{shifted_study_match_count}**",
        f"- Referenceable study folders after normalization: **{normalized_study_match_count}**",
        f"- Study folders still not referenceable at that level: **{len(unmatched_studies)}**",
        f"- Files found in cohort zips: **{len(all_files)}**",
        f"- Exact file matches to `C3_INSTANCE_TABLE.csv` by `pidscan + studydate + studytime + file`: **{exact_file_match_count}**",
        f"- Time-shifted file matches after study normalization: **{shifted_file_match_count}**",
        f"- Referenceable files after normalization: **{normalized_file_match_count}**",
        f"- Files still not referenceable at that level: **{len(unmatched_files)}**",
        "",
        "## Unmatched Studies By Zip",
        "",
    ]

    for zip_name in sorted(path.name for path in cohort_zips):
        md_lines.append(f"- `{zip_name}`: {unmatched_studies_by_zip.get(zip_name, 0)}")

    md_lines.extend(
        [
            "",
            "## Unmatched Files By Zip",
            "",
        ]
    )
    for zip_name in sorted(path.name for path in cohort_zips):
        md_lines.append(f"- `{zip_name}`: {unmatched_files_by_zip.get(zip_name, 0)}")

    md_lines.extend(
        [
            "",
            "## Time Shift Distribution",
            "",
        ]
    )
    for shift, count in sorted(time_shift_counter.items()):
        md_lines.append(f"- `{shift}`: {count}")

    md_lines.extend(
        [
            "",
            "## File Extension Counts",
            "",
        ]
    )
    for ext, count in sorted(report["summary"]["cohort_file_extensions"].items()):
        md_lines.append(f"- `{ext}`: {count}")

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Study folder` matching uses the folder pattern `PIDSCAN_YYYYMMDD_HHMMSS` from the cohort zip path.",
            "- `Exact study` matching uses `pidscan + studydate + studytime` directly.",
            "- `Time-shifted study` matching uses the same `pidscan + studydate` and the same minute/second, allowing the hour to differ when there is a unique candidate in `C3_INSTANCE_TABLE.csv`.",
            "- `File` matching uses the exact cohort filename against the `file` column in `C3_INSTANCE_TABLE.csv` within the exact or normalized study key.",
            "- `pidscan_in_crf` and `pidscan_in_sr` are included in the CSV outputs to show whether unmatched cohort content still belongs to subjects present in the CRF or SR tables.",
            "",
            "## Output Files",
            "",
            f"- `{report_json.name}`",
            f"- `{unmatched_studies_csv.name}`",
            f"- `{unmatched_files_csv.name}`",
        ]
    )

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Structured zip: {structured_zip}")
    print(f"Cohort zips: {[path.name for path in cohort_zips]}")
    print(f"Wrote {report_md}")
    print(f"Wrote {report_json}")
    print(f"Wrote {unmatched_studies_csv}")
    print(f"Wrote {unmatched_files_csv}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

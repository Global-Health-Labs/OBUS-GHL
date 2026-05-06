#!/usr/bin/env python3
"""Stage Vivli cohort zips or expanded folders into the repo's preprocessing layout.

This bridges the Vivli zip-based delivery format to the existing v9
preprocessing pipeline with minimal downstream code changes:

1. Scan the structured-data zip plus all `Cohort*.zip` archives and/or
   expanded cohort directories.
2. Build a preprocess-ready instance metadata CSV in `sheets/`.
3. Optionally extract the referenced raw files into
   `<raw_root>/<project>/Ultrasound/YYYY-MM/<study_key>/`.

The output metadata is designed to be consumed directly by
`preprocess_data_v9.py` via a dedicated YAML config.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ghlobus.utilities.constants import META_DIR


DEFAULT_STRUCTURED_ZIP = "Vivli-UNC-FAMLI3Twins-StructuredData-19-Feb-2026.zip"
DEFAULT_PROJECT = "VIVLI_FAMLI3"
DEFAULT_METADATA_FILE = "VIVLI_FAMLI3_instance_metadata.csv"
DEFAULT_SUMMARY_FILE = "VIVLI_FAMLI3_staging_summary.json"
DEFAULT_SKIPPED_FILE = "VIVLI_FAMLI3_skipped_raw_files.csv"
DEFAULT_MANIFEST_FILE = "VIVLI_FAMLI3_staging_manifest.csv"

STUDY_DIR_RE = re.compile(r"^(FA3-[^_]+)_(\d{8})_(\d{6})$")
UID_LIKE_RE = re.compile(r"^\d+(?:\.\d+)+$")


@dataclass(frozen=True)
class RawEntry:
    source_label: str
    source_type: str
    source_member: str
    internal_dir: str
    study_key: str
    pidscan: str
    studydate: str
    studytime: str
    relpath: str
    filename: str
    normalized_filename: str
    extension: str
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the structured-data zip and cohort zip files.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Root directory where raw files should be staged for preprocessing.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output dataset root. The metadata CSV is written to <out-root>/sheets/.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="Project name used in staged relpaths and preprocessing configs.",
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
    parser.add_argument(
        "--cohort-dir",
        action="append",
        default=[],
        help="Optional expanded cohort directory path or basename. Repeat to pin specific extracted roots.",
    )
    parser.add_argument(
        "--metadata-file",
        default=DEFAULT_METADATA_FILE,
        help="Filename for the preprocess-ready metadata CSV written under sheets/.",
    )
    parser.add_argument(
        "--summary-file",
        default=DEFAULT_SUMMARY_FILE,
        help="Filename for the staging summary JSON written under sheets/.",
    )
    parser.add_argument(
        "--skipped-file",
        default=DEFAULT_SKIPPED_FILE,
        help="Filename for skipped-raw-file diagnostics written under sheets/.",
    )
    parser.add_argument(
        "--manifest-file",
        default=DEFAULT_MANIFEST_FILE,
        help="Filename for the staging manifest CSV written under sheets/.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Stage files into --raw-root by extracting from zips or copying from expanded directories.",
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

    non_cohort_candidates = [path for path in zip_paths if not path.name.lower().startswith("cohort")]
    if len(non_cohort_candidates) == 1:
        return non_cohort_candidates[0]

    raise FileNotFoundError(
        "Unable to uniquely determine the structured-data zip. Pass --structured-zip explicitly."
    )


def resolve_cohort_zips(data_dir: Path, cohort_zips: list[str], cohort_dirs: list[str]) -> list[Path]:
    if cohort_zips:
        resolved = [resolve_path(Path(value), data_dir) for value in cohort_zips]
    else:
        resolved = sorted(
            path for path in data_dir.glob("*.zip") if path.is_file() and path.name.lower().startswith("cohort")
        )

    if not resolved and not cohort_dirs:
        raise FileNotFoundError("No cohort zip files found. Pass --cohort-zip or stage Cohort*.zip files.")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cohort zip files not found: {missing}")

    return resolved


def resolve_cohort_dirs(data_dir: Path, cohort_dirs: list[str], cohort_zips: list[Path]) -> list[Path]:
    if cohort_dirs:
        resolved = [resolve_path(Path(value), data_dir) for value in cohort_dirs]
    else:
        zip_names = {path.name for path in cohort_zips}
        resolved = sorted(
            path
            for path in data_dir.iterdir()
            if path.is_dir()
            and path.name.lower().startswith("cohort")
            and f"{path.name}.zip" not in zip_names
        )

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cohort directories not found: {missing}")

    return resolved


def open_csv_from_zip(zip_path: Path, suffix: str) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        target = next(name for name in zf.namelist() if name.endswith(suffix))
        with zf.open(target, "r") as handle:
            text_handle = (line.decode("utf-8-sig", errors="replace") for line in handle)
            reader = csv.DictReader(text_handle)
            yield from reader


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


def normalize_filename(filename: str) -> str:
    base = Path(filename).name
    if base.lower().endswith(".dcm"):
        return base[:-4]
    return base


def build_relpath(project: str, study_key: str, studydate: str) -> str:
    return f"{project}/Ultrasound/{studydate[:4]}-{studydate[4:6]}/{study_key}"


def scan_cohort_zips(cohort_zips: list[Path], project: str) -> list[RawEntry]:
    raw_entries: list[RawEntry] = []

    for zip_path in cohort_zips:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                stripped = info.filename.strip("/")
                if not stripped or stripped.endswith("/"):
                    continue

                parts = stripped.split("/")
                if len(parts) < 3:
                    continue

                internal_dir = "/".join(parts[:-1])
                leaf = internal_dir.rsplit("/", 1)[-1]
                match = STUDY_DIR_RE.match(leaf)
                if not match:
                    continue

                pidscan, studydate, studytime = match.groups()
                filename = parts[-1]
                study_key = f"{pidscan}_{studydate}_{studytime}"
                relpath = build_relpath(project, study_key, studydate)
                raw_entries.append(
                    RawEntry(
                        source_label=zip_path.name,
                        source_type="zip",
                        source_member=info.filename,
                        internal_dir=internal_dir,
                        study_key=study_key,
                        pidscan=pidscan,
                        studydate=studydate,
                        studytime=studytime,
                        relpath=relpath,
                        filename=filename,
                        normalized_filename=normalize_filename(filename),
                        extension=detect_extension(filename),
                        source_path=f"{zip_path}:{info.filename}",
                    )
                )

    return raw_entries


def scan_cohort_dirs(cohort_dirs: list[Path], project: str) -> list[RawEntry]:
    raw_entries: list[RawEntry] = []

    for root_dir in cohort_dirs:
        for path in sorted(root_dir.rglob("*")):
            if not path.is_file():
                continue

            parts = path.relative_to(root_dir).parts
            if len(parts) < 2:
                continue

            study_index = None
            study_match = None
            for idx, part in enumerate(parts[:-1]):
                match = STUDY_DIR_RE.match(part)
                if match:
                    study_index = idx
                    study_match = match
                    break

            if study_index is None or study_match is None:
                continue

            pidscan, studydate, studytime = study_match.groups()
            study_key = f"{pidscan}_{studydate}_{studytime}"
            relpath = build_relpath(project, study_key, studydate)
            filename = path.name
            internal_dir = "/".join(parts[: len(parts) - 1])
            raw_entries.append(
                RawEntry(
                    source_label=root_dir.name,
                    source_type="dir",
                    source_member=str(path.relative_to(root_dir)),
                    internal_dir=internal_dir,
                    study_key=study_key,
                    pidscan=pidscan,
                    studydate=studydate,
                    studytime=studytime,
                    relpath=relpath,
                    filename=filename,
                    normalized_filename=normalize_filename(filename),
                    extension=detect_extension(filename),
                    source_path=str(path),
                )
            )

    return raw_entries


def load_instance_rows(structured_zip: Path) -> list[dict[str, str]]:
    rows = []
    for row in open_csv_from_zip(structured_zip, "C3_INSTANCE_TABLE.csv"):
        filename = Path((row.get("file") or "").strip()).name
        studytime = normalize_studytime(row.get("studytime") or "")
        row = dict(row)
        row["_filename"] = filename
        row["_normalized_filename"] = normalize_filename(filename)
        row["_studytime"] = studytime
        row["_study_key"] = f"{(row.get('pidscan') or '').strip()}_{(row.get('studydate') or '').strip()}_{studytime}"
        rows.append(row)
    return rows


def choose_instance_row(raw_entry: RawEntry, candidate_rows: list[dict[str, str]]) -> tuple[dict[str, str] | None, str]:
    if not candidate_rows:
        return None, "no_instance_candidate"

    if len(candidate_rows) == 1:
        return candidate_rows[0], "unique_pidscan_filename_match"

    exact_filename = [row for row in candidate_rows if row["_filename"] == raw_entry.filename]
    if len(exact_filename) == 1:
        return exact_filename[0], "unique_exact_filename_match"

    exact_study = [row for row in candidate_rows if row["_study_key"] == raw_entry.study_key]
    if len(exact_study) == 1:
        return exact_study[0], "unique_exact_study_match"

    same_date_mmss = [
        row
        for row in candidate_rows
        if (row.get("studydate") or "").strip() == raw_entry.studydate
        and row["_studytime"][2:] == raw_entry.studytime[2:]
    ]
    if len(same_date_mmss) == 1:
        return same_date_mmss[0], "unique_same_date_mmss_match"

    return None, "ambiguous_instance_candidate"


def normalize_manufacturer(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "Unknown"
    lower = text.lower()
    if "sonosite" in lower:
        return "Sonosite"
    if "butterfly" in lower:
        return "Butterfly"
    if "clarius" in lower:
        return "Clarius"
    if "ge" in lower:
        return "GE"
    return text


def normalize_model(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "Unnamed"
    if text == "Voluson-S":
        return "Voluson S"
    return text


def prefer_spacing(row: dict[str, str]) -> tuple[str, str]:
    pdx = (row.get("pixel_spacing_x_mp4") or row.get("pixel_spacing_x_dcm") or "").strip()
    pdy = (row.get("pixel_spacing_y_mp4") or row.get("pixel_spacing_y_dcm") or "").strip()
    return pdx, pdy


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_extracted(zip_path: Path, zip_member: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "already_present"

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(zip_member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return "extracted"


def ensure_copied(source_path: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "already_present"
    shutil.copy2(source_path, destination)
    return "copied"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    sheets_dir = out_root / META_DIR
    sheets_dir.mkdir(parents=True, exist_ok=True)

    structured_zip = resolve_structured_zip(data_dir, args.structured_zip)
    cohort_zips = resolve_cohort_zips(data_dir, args.cohort_zip, args.cohort_dir)
    cohort_zip_map = {path.name: path for path in cohort_zips}

    metadata_path = sheets_dir / args.metadata_file
    summary_path = sheets_dir / args.summary_file
    skipped_path = sheets_dir / args.skipped_file
    manifest_path = sheets_dir / args.manifest_file

    cohort_dirs = resolve_cohort_dirs(data_dir, args.cohort_dir, cohort_zips)
    raw_entries = scan_cohort_zips(cohort_zips, args.project) + scan_cohort_dirs(cohort_dirs, args.project)
    instance_rows = load_instance_rows(structured_zip)

    instance_by_pidscan_filename: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in instance_rows:
        pidscan = (row.get("pidscan") or "").strip()
        normalized_filename = row["_normalized_filename"]
        if pidscan and normalized_filename:
            instance_by_pidscan_filename[(pidscan, normalized_filename)].append(row)

    metadata_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    match_reason_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    input_type_counter: Counter[str] = Counter()
    stage_action_counter: Counter[str] = Counter()
    seen_destinations: dict[str, str] = {}

    study_defaults: dict[str, dict[str, str]] = defaultdict(dict)
    resolved_raw_entries: list[tuple[RawEntry, dict[str, object]]] = []

    for raw_entry in raw_entries:
        input_type_counter[raw_entry.source_type] += 1
        candidates = instance_by_pidscan_filename.get((raw_entry.pidscan, raw_entry.normalized_filename), [])
        matched_row, match_reason = choose_instance_row(raw_entry, candidates)
        match_reason_counter[match_reason] += 1

        if matched_row is not None:
            manufacturer = normalize_manufacturer(matched_row.get("manufacturer") or "")
            model = normalize_model(matched_row.get("model") or "")
            pdx, pdy = prefer_spacing(matched_row)
            if raw_entry.extension not in {".dcm", ".mp4"}:
                skipped_rows.append(
                    {
                        "study_key": raw_entry.study_key,
                        "filename": raw_entry.filename,
                        "extension": raw_entry.extension,
                        "skip_reason": "unsupported_file_type_for_v9_preprocessing",
                        "source_label": raw_entry.source_label,
                        "source_type": raw_entry.source_type,
                        "source_path": raw_entry.source_path,
                    }
                )
                continue
            metadata = {
                "PID": (matched_row.get("pid") or "").strip(),
                "StudyID": raw_entry.pidscan,
                "relpath": raw_entry.relpath,
                "StudyDate": (matched_row.get("studydate") or "").strip() or raw_entry.studydate,
                "StudyTime": matched_row["_studytime"] or raw_entry.studytime,
                "filename": raw_entry.filename,
                "file_type": raw_entry.extension.lstrip("."),
                "Manufacturer": manufacturer,
                "ManufacturerModelName": model,
                "DeviceSerialNumber": "",
                "SOPInstanceUID": (matched_row.get("sopinstanceuid") or "").strip(),
                "StudyInstanceUID": "",
                "SOPClassUID": (matched_row.get("sopclassuid") or "").strip(),
                "NumberOfFrames": (matched_row.get("num_frames") or "").strip(),
                "Rows": (matched_row.get("rows_dcm") or matched_row.get("rows_mp4") or "").strip(),
                "Columns": (matched_row.get("columns_dcm") or matched_row.get("columns_mp4") or "").strip(),
                "PhysicalDeltaX": pdx,
                "PhysicalDeltaY": pdy,
                "tag": (matched_row.get("tag") or "").strip() or "Unknown",
                "source_cohort_zip": raw_entry.source_label,
                "source_internal_dir": raw_entry.internal_dir,
                "source_type": raw_entry.source_type,
                "source_path": raw_entry.source_path,
                "metadata_source": "instance_table",
                "instance_match_reason": match_reason,
                "instance_table_file": matched_row["_filename"],
            }
            study_defaults[raw_entry.study_key] = {
                "Manufacturer": manufacturer,
                "ManufacturerModelName": model,
                "PhysicalDeltaX": pdx,
                "PhysicalDeltaY": pdy,
            }
            resolved_raw_entries.append((raw_entry, metadata))
            continue

        study_default = study_defaults.get(raw_entry.study_key, {})
        manufacturer = study_default.get("Manufacturer", "Unknown")
        model = study_default.get("ManufacturerModelName", "Unnamed")
        pdx = study_default.get("PhysicalDeltaX", "")
        pdy = study_default.get("PhysicalDeltaY", "")

        if raw_entry.extension not in {".dcm", ".mp4"}:
            skipped_rows.append(
                {
                    "study_key": raw_entry.study_key,
                    "filename": raw_entry.filename,
                    "extension": raw_entry.extension,
                    "skip_reason": "unsupported_file_type_for_v9_preprocessing",
                    "source_label": raw_entry.source_label,
                    "source_type": raw_entry.source_type,
                    "source_path": raw_entry.source_path,
                }
            )
            continue

        if raw_entry.extension == ".mp4" and not pdx:
            skipped_rows.append(
                {
                    "study_key": raw_entry.study_key,
                    "filename": raw_entry.filename,
                    "extension": raw_entry.extension,
                    "skip_reason": "raw_only_mp4_missing_physical_delta_x",
                    "source_label": raw_entry.source_label,
                    "source_type": raw_entry.source_type,
                    "source_path": raw_entry.source_path,
                }
            )
            continue

        metadata = {
            "PID": "",
            "StudyID": raw_entry.pidscan,
            "relpath": raw_entry.relpath,
            "StudyDate": raw_entry.studydate,
            "StudyTime": raw_entry.studytime,
            "filename": raw_entry.filename,
            "file_type": raw_entry.extension.lstrip("."),
            "Manufacturer": manufacturer,
            "ManufacturerModelName": model,
            "DeviceSerialNumber": "",
            "SOPInstanceUID": raw_entry.normalized_filename if raw_entry.extension == ".dcm" else "",
            "StudyInstanceUID": "",
            "SOPClassUID": "",
            "NumberOfFrames": "",
            "Rows": "",
            "Columns": "",
            "PhysicalDeltaX": pdx,
            "PhysicalDeltaY": pdy,
            "tag": "Unknown",
            "source_cohort_zip": raw_entry.source_label,
            "source_internal_dir": raw_entry.internal_dir,
            "source_type": raw_entry.source_type,
            "source_path": raw_entry.source_path,
            "metadata_source": "cohort_only",
            "instance_match_reason": match_reason,
            "instance_table_file": "",
        }
        resolved_raw_entries.append((raw_entry, metadata))

    for raw_entry, metadata in resolved_raw_entries:
        source_counter[str(metadata["metadata_source"])] += 1
        metadata_rows.append(metadata)
        destination = raw_root / metadata["relpath"] / metadata["filename"]
        dest_key = str(destination)
        manifest_row = {
            "source_type": raw_entry.source_type,
            "source_label": raw_entry.source_label,
            "source_member": raw_entry.source_member,
            "source_path": raw_entry.source_path,
            "study_key": raw_entry.study_key,
            "filename": raw_entry.filename,
            "destination_path": dest_key,
            "metadata_source": metadata["metadata_source"],
            "instance_match_reason": metadata["instance_match_reason"],
            "stage_action": "not_requested",
        }
        if dest_key in seen_destinations and seen_destinations[dest_key] != raw_entry.source_path:
            manifest_row["stage_action"] = "duplicate_destination_conflict"
            manifest_row["conflict_with"] = seen_destinations[dest_key]
            stage_action_counter[manifest_row["stage_action"]] += 1
            manifest_rows.append(manifest_row)
            continue
        seen_destinations[dest_key] = raw_entry.source_path
        if args.extract:
            if raw_entry.source_type == "zip":
                action = ensure_extracted(cohort_zip_map[raw_entry.source_label], raw_entry.source_member, destination)
            else:
                action = ensure_copied(Path(raw_entry.source_path), destination)
            manifest_row["stage_action"] = action
        stage_action_counter[manifest_row["stage_action"]] += 1
        manifest_rows.append(manifest_row)

    metadata_rows.sort(key=lambda row: (str(row["relpath"]), str(row["filename"])))
    write_csv(metadata_path, metadata_rows)
    write_csv(skipped_path, skipped_rows)
    write_csv(manifest_path, manifest_rows)

    summary = {
        "project": args.project,
        "structured_zip": structured_zip.name,
        "cohort_zips": [path.name for path in cohort_zips],
        "cohort_dirs": [path.name for path in cohort_dirs],
        "extract_enabled": args.extract,
        "raw_entry_count": len(raw_entries),
        "metadata_row_count": len(metadata_rows),
        "skipped_row_count": len(skipped_rows),
        "metadata_source_counts": dict(sorted(source_counter.items())),
        "input_type_counts": dict(sorted(input_type_counter.items())),
        "instance_match_reason_counts": dict(sorted(match_reason_counter.items())),
        "file_type_counts": dict(sorted(Counter(str(row["file_type"]) for row in metadata_rows).items())),
        "skipped_reason_counts": dict(sorted(Counter(str(row["skip_reason"]) for row in skipped_rows).items())),
        "stage_action_counts": dict(sorted(stage_action_counter.items())),
        "metadata_csv": str(metadata_path),
        "skipped_csv": str(skipped_path),
        "manifest_csv": str(manifest_path),
        "raw_root": str(raw_root),
        "out_root": str(out_root),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Data directory: {data_dir}")
    print(f"Structured zip: {structured_zip}")
    print(f"Cohort zips: {[path.name for path in cohort_zips]}")
    print(f"Cohort dirs: {[path.name for path in cohort_dirs]}")
    print(f"Raw root: {raw_root}")
    print(f"Output root: {out_root}")
    print(f"Extract enabled: {args.extract}")
    print(f"Wrote metadata CSV: {metadata_path}")
    print(f"Wrote skipped-file CSV: {skipped_path}")
    print(f"Wrote manifest CSV: {manifest_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

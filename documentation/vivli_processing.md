# Vivli Processing Bridge

The Vivli delivery is zip-based, while the v9 preprocessing pipeline expects:

- a normal raw file tree on disk
- a preprocess-ready instance metadata CSV in `sheets/`

The bridge script [stage_vivli_for_preproc.py](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/stage_vivli_for_preproc.py) creates that staging layout with minimal downstream changes.

## What It Writes

- Raw files under:
  - `<raw_root>/<project>/Ultrasound/YYYY-MM/<PIDSCAN_YYYYMMDD_HHMMSS>/`
- Metadata under:
  - `<out_root>/sheets/VIVLI_FAMLI3_instance_metadata.csv`
- Diagnostics under:
  - `<out_root>/sheets/VIVLI_FAMLI3_staging_summary.json`
  - `<out_root>/sheets/VIVLI_FAMLI3_skipped_raw_files.csv`

## Metadata Strategy

- Prefer `C3_INSTANCE_TABLE.csv` metadata when a cohort file can be matched by `pidscan + normalized filename`.
- Use the raw filename in the output metadata so `preprocess_data_v9.py` can open the extracted file on disk.
- Allow fallback `cohort_only` rows for raw DICOM files that lack a confident instance-table row.
- Skip raw-only MP4 files when no `PhysicalDeltaX` value can be inferred, because the current MP4 preprocessing code requires it.
- Skip PNG files because the current v9 preprocessing pipeline is video-oriented and filters PNG instances upstream.

## Example Commands

Stage metadata only:

```bash
python ghlobus/ingestion/stage_vivli_for_preproc.py \
  --data-dir /path/to/vivli \
  --raw-root /data/ML-Raw-Data/VIVLI \
  --out-root /data/ML-Project-Data/VIVLI/Datasets/v1
```

Stage metadata and extract raw files:

```bash
python ghlobus/ingestion/stage_vivli_for_preproc.py \
  --data-dir /path/to/vivli \
  --raw-root /data/ML-Raw-Data/VIVLI \
  --out-root /data/ML-Project-Data/VIVLI/Datasets/v1 \
  --extract
```

Then preprocess with:

```bash
python ghlobus/ingestion/preprocess_data_v9.py \
  --yaml ghlobus/ingestion/configs/VIVLI_FAMLI3_preproc_v9.yaml
```

## Notes

- Update the paths in [VIVLI_FAMLI3_preproc_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_preproc_v9.yaml) for your VM or container before running preprocessing.
- The bridge script is deliberately conservative. It favors producing a usable preprocessing sheet over forcing Vivli through the full SR/CRF merge pipeline.

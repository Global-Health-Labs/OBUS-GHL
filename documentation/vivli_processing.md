# Vivli Processing Bridge

The Vivli delivery may be zip-based or already expanded, while the v9 preprocessing pipeline expects:

- a normal raw file tree on disk
- a preprocess-ready instance metadata CSV in `sheets/`

The bridge script [stage_vivli_for_preproc.py](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/stage_vivli_for_preproc.py) creates that staging layout with minimal downstream changes.

It accepts:

- `Cohort*.zip` files
- expanded cohort directories
- a mix of both

## What It Writes

- Raw files under:
  - `<raw_root>/<project>/Ultrasound/YYYY-MM/<PIDSCAN_YYYYMMDD_HHMMSS>/`
- Metadata under:
  - `<out_root>/sheets/VIVLI_FAMLI3_instance_metadata.csv`
- Diagnostics under:
  - `<out_root>/sheets/VIVLI_FAMLI3_staging_summary.json`
  - `<out_root>/sheets/VIVLI_FAMLI3_skipped_raw_files.csv`
  - `<out_root>/sheets/VIVLI_FAMLI3_staging_manifest.csv`

## Metadata Strategy

- Prefer `C3_INSTANCE_TABLE.csv` metadata when a cohort file can be matched by `pidscan + normalized filename`.
- Use the raw filename in the output metadata so `preprocess_data_v9.py` can open the extracted file on disk.
- Allow fallback `cohort_only` rows for raw DICOM files that lack a confident instance-table row.
- Skip raw-only MP4 files when no `PhysicalDeltaX` value can be inferred, because the current MP4 preprocessing code requires it.
- Skip PNG files because the current v9 preprocessing pipeline is video-oriented and filters PNG instances upstream.
- When `--extract` is used, zip-backed inputs are extracted and directory-backed inputs are copied into the normalized layout automatically.
- For already-expanded cohort directories, prefer `--stage-mode symlink` to create the normalized layout without duplicating terabytes of source data.
- Duplicate destination conflicts are logged in the manifest instead of being silently overwritten.

## Example Commands

Stage metadata only:

```bash
python ghlobus/ingestion/stage_vivli_for_preproc.py \
  --data-dir /path/to/vivli \
  --raw-root /data/ML-Raw-Data/VIVLI \
  --out-root /data/ML-Project-Data/VIVLI/Datasets/v1
```

Stage from an already-expanded cohort directory:

```bash
python ghlobus/ingestion/stage_vivli_for_preproc.py \
  --data-dir /path/to/structured-data \
  --cohort-dir /path/to/CohortExpanded \
  --raw-root /data/ML-Raw-Data/VIVLI \
  --out-root /data/ML-Project-Data/VIVLI/Datasets/v1 \
  --stage-mode symlink
```

For GCS data mounted through Cloud Storage FUSE, avoid recursive filesystem
walks during staging by using the GCS listing backend. With a running
`gcsfuse` process, `auto` can usually infer the matching `gs://` prefix:

```bash
python ghlobus/ingestion/stage_vivli_for_preproc.py \
  --data-dir /mnt/localssd/vivli_input \
  --structured-zip /mnt/localssd/vivli_input/Vivli-UNC-FAMLI3Twins-StructuredData-19-Feb-2026.zip \
  --cohort-dir /mnt/vivli_gcs/Cohort1 \
  --cohort-dir /mnt/vivli_gcs/Cohort2 \
  --raw-root /mnt/localssd/vivli_raw_full \
  --out-root /mnt/localssd/vivli_out_full \
  --stage-mode symlink
```

If automatic mount inference is unavailable, add:

```bash
  --listing-backend gcs \
  --gcs-uri-base gs://vivli-data/expanded/FAMLI3 \
  --gcs-mount-root /mnt/vivli_gcs
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

## Downstream Curation

After preprocessing has produced `VIVLI_FAMLI3_prototype.csv`, the Vivli v9
curation runner builds exam metadata, merges it onto the preprocessed
prototype, and writes task-ready GA, FP, EFW, and Twin splits.

Required inputs:

- `C3_CRF.csv` and `C3_SR.csv`, either in the structured-data zip or an
  expanded structured-data directory.
- `<out_root>/sheets/VIVLI_FAMLI3_prototype.csv` from preprocessing.
- Preprocessed video tensors under the configured raw/project output root.

Update the paths in these configs before running outside the default
`/data/...` layout:

- [VIVLI_FAMLI3_sr_crf_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_sr_crf_v9.yaml)
- [VIVLI_FAMLI3_prototype_curated_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_prototype_curated_v9.yaml)
- [VIVLI_FAMLI3_efw_data_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_efw_data_v9.yaml)
- [VIVLI_FAMLI3_twin_data_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_twin_data_v9.yaml)
- [VIVLI_FAMLI3_ga_split_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_ga_split_v9.yaml)
- [VIVLI_FAMLI3_fp_split_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_fp_split_v9.yaml)
- [VIVLI_FAMLI3_efw_split_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_efw_split_v9.yaml)
- [VIVLI_FAMLI3_twin_split_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_twin_split_v9.yaml)

Run the full downstream flow from the repository root:

```bash
cd ghlobus/ingestion
./run_vivli_curation_v9.sh
```

The runner executes:

1. Vivli SR/CRF exam curation.
2. Prototype plus exam metadata merge.
3. EFW and Twin v9 selection.
4. GA, FP, EFW, and Twin patient-level train/val/test split generation.

Main outputs:

- `<out_root>/sheets/VIVLI_FAMLI3_Exam_Data.csv`
- `<out_root>/sheets/VIVLI_FAMLI3_prototype_curated.csv`
- `<out_root>/sheets/VIVLI_FAMLI3_prototype_efw_selected.csv`
- `<out_root>/sheets/VIVLI_FAMLI3_prototype_twin_selected.csv`
- `<out_root>/splits/GA_T75_V10_H15/GA_100/{train,val,test}.csv`
- `<out_root>/splits/FP_T75_V10_H15/FP_100/{train,val,test}.csv`
- `<out_root>/splits/EFW_T75_V10_H15/EFW_100/{train,val,test}.csv`
- `<out_root>/splits/TWIN_T75_V10_H15/TWIN_100/{train,val,test}.csv`

The curated prototype normalizes task fields used by the split step:

- `GA`: SR `ega` preferred, with CRF `ega` as fallback.
- `BPD`, `HC`, `AC`, `FL`, `CRL`: SR means preferred, with CRF ultrasound
  biometric columns as fallback.
- `EFW`: from CRF `us_efw`.
- `NOF`: derived from available twin and multiple-gestation fields.
- `lie`: binary fetal presentation label where `0` is cephalic and `1` is
  non-cephalic; missing, unknown, and variable lie values are excluded.

The task split step keeps only `Good_video` rows, writes non-empty `outpath`
values, and splits at patient level to reduce leakage.

## Batch Handoff

`preprocess_data_v9.py` can write a per-batch manifest and run a configured
post-batch command after each completed log file. The core preprocessor does
not assume GCS or any other destination; it only passes the manifest path,
batch log path, and local output root to the configured command.

The Vivli config includes a disabled example that calls the GCS-specific
adapter `upload_batch_manifest_to_gcs.py`. The adapter retries failed
`gcloud storage` operations and falls back to uploading through a temporary
object when a resumable upload for the destination object gets stuck. For
quota-limited runs, enable it and set:

```yaml
processing:
  post_batch:
    enabled: true
    delete_local_outputs: true
```

The default manifest location is `<out_dir>/batch_manifests/<project>` if
`manifest_dir` is omitted.

## Notes

- Update the paths in [VIVLI_FAMLI3_preproc_v9.yaml](/Users/dan/code/OBUS-GHL/ghlobus/ingestion/configs/VIVLI_FAMLI3_preproc_v9.yaml) for your VM or container before running preprocessing.
- The bridge script is deliberately conservative. It favors producing a usable preprocessing sheet over forcing Vivli through the full SR/CRF merge pipeline.
- Users do not need to manually rearrange expanded cohort folders; the staging script detects study folders and rebuilds the expected `YYYY-MM/study_key` layout itself.

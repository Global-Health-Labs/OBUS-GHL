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

## Label and artifact contracts

EFW training consumes standardized `log_AC`, `log_FL`, `log_HC`, and
`log_BPD` columns. These use the existing v9 biometry mean/std constants
shared with prediction decoding in `ghlobus.utilities.normalization`.
They intentionally do not fit new Vivli biometry statistics: doing so would
require updating checkpoint decoding as well. `z_log_*` columns are aliases
for those standardized values; `raw_log_*` columns contain the unscaled logs.
GA uses the shared legacy v4 constants so the existing dataloader does not
change labels when it reloads the splits. Other normalized labels retain their
training-split statistics.
`normalization.json` beside each split records the applied means, standard
deviations, source, and normalized column names. Keep this file with the
dataset used for a training run.

Regenerate existing Vivli EFW splits before training: older versions wrote
raw logs into the `log_*` columns consumed by EFW training. These changes do
not repair a model already trained with those incompatible labels.

Tensor filenames strip only `.dcm` and `.mp4` (case insensitive). An
extensionless DICOM UID such as `1.2.840.12345` becomes
`1.2.840.12345.pt`. Preprocessing, batch manifests, and task splits share
this convention. Regenerate splits with truncated UID paths; correctly
named existing tensors do not need to be regenerated.

## Resuming interrupted handoffs

A batch CSV records processing completion, not upload completion. The
preprocessor keeps the output manifest and writes an `.uploaded.json`
receipt only after the post-batch command succeeds. On restart it retries
pending handoffs before decoding subsequent batches, and finishes local
cleanup for batches with matching receipts. Handoff commands must tolerate
repeated execution: interruption after a remote upload but before its local
receipt can cause a retry.

Keep the local `logs/<project>` and `batch_manifests/<project>` directories
when resuming, even when uploaded tensors are deleted locally. Batch logs,
manifests, and receipts are published with atomic replacement. A run
fingerprint binds checkpoints to the input sheet and configuration; use a
separate output directory for changed inputs or configuration. Missing batch
logs or logs inconsistent with the current input cause an explicit error.

Older logs without receipts can resume if their local artifacts remain:
they are validated against the current input and handed off again. If older
runs already deleted artifacts without receipts, restore/reprocess those
artifacts before resuming. Missing tensors are not treated as successful
uploads. Local receipts certify successful command completion; they do not
independently verify remote object integrity.

## Staging conflicts, recovery, and patient IDs

Each staged destination contributes at most one accepted metadata row. The
first encountered entry is retained; subsequent entries targeting that path
are recorded in the staging manifest with `duplicate_destination_conflict`
and `conflict_with`. Review that manifest before using a delivery with
conflicts. CSV reports are replaced atomically after writing completes.

Copy and ZIP extraction write a temporary sibling file before atomically
replacing the destination. An interrupted operation does not publish a partial
new file. Existing files are skipped only after size and content validation:
SHA-256 against the source for expanded files, and the archive member CRC for
ZIP files. Resuming copy staging therefore reads existing source/destination
content and can be I/O intensive on remote mounts. The symlink staging mode
remains available for expanded datasets.

Curation fills missing prototype patient IDs from exam metadata, validates a
many-to-one study join, and rejects conflicting nonblank IDs. Rows still
missing an ID produce a warning rather than silently disappearing during
patient-level grouping. Regenerate curated prototypes and downstream splits
to recover eligible rows from older runs. Twin-label selection and study
spacing fallback behavior are unchanged.

## GA compatibility and small validation batches

Vivli splits expose `ga_boe` and `log_ga_boe` as compatibility aliases for
Vivli `GA` and its raw log, because existing GA dataloaders and prediction
writers require those column names. This does not change the label source:
Vivli GA still comes from the documented SR/CRF precedence, not an independent
BOE assessment. `z_log_ga` uses the legacy v4 constants shared with inference.
Regenerate GA splits before using the existing datamodule.

Batch targets are snapped to exam starts, then sorted and deduplicated with
zero boundaries removed. Very small batch sizes can exceed their target to
keep an exam together, but no longer create empty batches. Runs with older
empty-batch logs should use a fresh validation output directory.

## Cloud sample validation (2026-10-11 Pacific)

Validated this working-tree snapshot on `vivli-preprocess-vm` in
`famli-3-vivli`, using an isolated `/data/obus-validation-20261011` directory.
Code archive SHA-256:
`010657bf5dfea4833a5a0a7bb37fe7475bac5dfba40245e1a9129427d8a4b997`.
Runtime: Python 3.10.12, torch 2.12.0+cu130, torchvision 0.27.0+cu130,
pandas 2.3.3; CPU processing with two workers.

The sample comprised three DICOM and three MP4 files across five studies.
Four produced uint8 tensors shaped `(frames, 1, 288, 288)`; each tensor was
exactly equal to its existing full-run reference. Two files were rejected
(`Multichannel_shape_invalid`, `Invalid_ultrasound_region_dicom`); all six
statuses matched the existing reference prototype. Resuming the run preserved
all tensor hashes.

A separate handoff check injected a subprocess failure before upload, verified
that no receipt appeared, retried successfully against a new GCS prefix, then
verified receipt reuse on replay. Four artifacts were uploaded and local
outputs were retained. Aggregate evidence is under
`gs://vivli-data/validation/20261011-data-contract-recovery/evidence/`.
Production output objects were not changed.

This is sampled preprocessing and handoff validation, not full-cohort coverage
or a model-training test. The local regression suite passed 48 tests. Real
GA/EFW dataloader and training smoke tests remain; the missing exam dataset
classes must be resolved before claiming twin training validation.

## Committed GA/EFW training smoke (2026-10-12 Pacific)

Commit `4a8906caf3a260dd0dd696b8e3715d61686900ff` passed real GA and EFW
video dataloader and Lightning training smoke tests on the existing CPU VM.
The deployment used `git archive`; local and remote archive SHA-256 matched:
`8a74d20616ec14010f4ce7206cd1f72cff44c9a7177306f3dcd0087ec1799dd4`.
No source modifications were required on the VM.

Both task splits were regenerated from existing FAMLI3 cloud metadata using
the committed split code and task configurations. Four training videos and
two validation videos per task were selected from distinct patients, with no
train/validation patient overlap. Existing processed tensors supplied the
images. Generated GA splits contained 166,647 training and 22,134 validation
rows; EFW contained 77,010 training and 10,049 validation rows.

The actual video datamodule produced input shape `(2, 2, 3, 256, 256)`, GA
labels `(2,)`, and EFW labels `(2, 4)`. All inputs and labels were finite;
loaded normalized labels matched the shared GA/biometry constants.
The repository's Lightning CLI and configured model architectures ran one
epoch, limited to two training batches and one validation batch, with two
frames per video. GA used EfficientNet V2 S with basic additive attention;
EFW used MobileNet V2 with eight additive attention modules. CSV logging
replaced external experiment logging, and checkpointing was disabled.

Both runs exited successfully with finite epoch losses:

| Task | Training loss | Validation loss |
| --- | ---: | ---: |
| GA | 0.995563 | 0.828135 |
| EFW | 0.591288 | 0.760351 |

These losses confirm execution only; the tiny sample and short run do not
measure predictive performance. Runtime: Python 3.10.12, torch 2.12.0,
torchvision 0.27.0, lightning 2.1.0, pandas 2.3.3, numpy 1.26.4.
GPU/mixed precision, full-length video training, full-cohort execution, and
twin exam training were not tested.

Reproduction scripts, generated configs, CSV metrics, logs, and aggregate
summaries are saved at
`gs://vivli-data/validation/20261012-4a8906c-training/evidence/`.
Persistent VM workspace: `/data/obus-training-4a8906c-20261012`.
Production outputs were not changed.

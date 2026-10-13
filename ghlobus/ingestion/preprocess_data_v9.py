"""
preprocess_data_v9.py

This module preprocesses raw data from any of the following projects:
# FAMLI2_enrolled
# FAMLI2
# FAMLI3
# DXA

Preprocessing comprises:

- crop ultrasound "fan" region
- rescale and pad to final dimensions
- standardization of target value
- write to pytorch file, spreadsheet

To avoid memory crashes, raw input data is processed by batches of ~1,000 videos.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import yaml
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from functools import partial
from joblib import Parallel, delayed

from ghlobus.utilities.constants import META_DIR, GOOD_VIDEO_MSG
from ghlobus.utilities.paths import output_file_base_name


def out_folder_name(df: pd.DataFrame) -> pd.Series:
    """
    Create a folder name from the StudyID, Manufacturer, and ManufacturerModelName columns.

    Args:
        df (pd.DataFrame): DataFrame with StudyID, Manufacturer, and ManufacturerModelName.

    Returns:
        pd.Series: Folder names created from the relpath, Manufacturer, and ManufacturerModelName.
    """
    # make sure ManufacturerModelName has no ' Probe' and no spaces
    df['ManufacturerModelName'] = df['ManufacturerModelName'].str.replace(' Probe', '')
    df['ManufacturerModelName'] = df['ManufacturerModelName'].str.replace(' ', '-')
    relpath_folders = df['relpath'].apply(os.path.basename)
    # create folder name from StudyID, Manufacturer, ManufacturerModelName, and relpath folders
    return relpath_folders + "_" + df['Manufacturer'] + "_" + df['ManufacturerModelName']


def batch_output_paths(batch_df: pd.DataFrame,
                       out_dir: str,
                       project: str,
                       folders: dict,
                       ) -> list[str]:
    """
    List local outputs owned by a completed batch.
    """
    output_paths = []

    for fmt in ['pt', 'jpg']:
        folder = folders.get(fmt)
        if folder is None:
            continue
        for _, row in batch_df.iterrows():
            file_base_name = output_file_base_name(row['filename'])
            local_base = os.path.join(
                out_dir,
                folder,
                project,
                row['exam_dir'],
                file_base_name,
            )
            if fmt == 'pt':
                local_path = local_base + '.pt'
                if os.path.exists(local_path):
                    output_paths.append(local_path)
            else:
                output_paths.extend(glob(local_base + "_#*.jpg"))

    raw_folder = folders.get('raw')
    if raw_folder is not None:
        for _, row in batch_df.iterrows():
            file_base_name = output_file_base_name(row['filename'])
            if row['fail_reason'] == GOOD_VIDEO_MSG:
                local_path = os.path.join(
                    out_dir,
                    raw_folder,
                    project,
                    f"{row['exam_dir']}_{file_base_name}_#0005.png",
                )
                if os.path.exists(local_path):
                    output_paths.append(local_path)
            else:
                local_dir = os.path.join(out_dir, raw_folder, project + "_bad")
                from ghlobus.utilities.data_utils import fail_reason_acronym
                try:
                    reason = fail_reason_acronym(row['fail_reason'])
                except KeyError:
                    reason = "*"
                output_paths.extend(glob(
                    os.path.join(
                        local_dir,
                        f"{row['exam_dir']}_{file_base_name}_{reason}_#0005.png",
                    )
                ))

    return sorted(output_paths)


def write_batch_manifest(batch_idx: int | str,
                         output_paths: list[str],
                         manifest_dir: str,
                         ) -> str:
    """
    Write one local manifest file containing the outputs owned by a batch.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    if isinstance(batch_idx, int):
        manifest_name = f"batch_{batch_idx:03d}_outputs.txt"
    else:
        manifest_name = f"{batch_idx}_outputs.txt"
    manifest_path = os.path.join(manifest_dir, manifest_name)
    atomic_write(manifest_path, "\n".join(output_paths) + "\n")
    return manifest_path


def run_post_batch_command(command: list[str],
                           batch_idx: int | str,
                           manifest_path: str,
                           batch_log_path: str,
                           out_dir: str,
                           ) -> None:
    """
    Run a configured post-batch command after placeholder substitution.
    """
    replacements = {
        "batch_idx": str(batch_idx),
        "manifest_path": manifest_path,
        "batch_log_path": batch_log_path,
        "out_dir": out_dir,
    }
    resolved_command = [
        part.format(**replacements)
        for part in command
    ]
    subprocess.run(resolved_command, check=True)


def final_output_paths(meta_path: str) -> list[str]:
    """
    List final metadata/prototype outputs to hand off after preprocessing.
    """
    return sorted(
        path for path in glob(os.path.join(meta_path, "*"))
        if os.path.isfile(path)
    )


def delete_batch_outputs(output_paths: list[str],
                         out_dir: str,
                         ) -> None:
    """
    Remove local outputs after a successful post-batch handoff.
    """
    parent_dirs = set()
    for path in output_paths:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        parent_dirs.add(os.path.dirname(path))

    out_dir = os.path.abspath(out_dir)
    for parent_dir in sorted(parent_dirs, key=len, reverse=True):
        parent_dir = os.path.abspath(parent_dir)
        while parent_dir.startswith(out_dir) and parent_dir != out_dir:
            try:
                os.rmdir(parent_dir)
            except OSError:
                break
            parent_dir = os.path.dirname(parent_dir)


def compute_batch_ends(exam_starts, n_videos: int, batch_size: int) -> list[int]:
    """Return increasing nonempty batch ends without splitting exams."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_videos <= 0:
        raise ValueError("No input videos to preprocess")
    boundaries = np.asarray(sorted(set(int(x) for x in exam_starts)))
    targets = range(batch_size, n_videos, batch_size)
    ends = {int(boundaries[np.argmin(np.abs(boundaries - target))]) for target in targets}
    return sorted(end for end in ends if 0 < end < n_videos) + [n_videos]


def atomic_write(path: str, text: str) -> None:
    """Publish checkpoint state only after the complete file has been written."""
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def handoff_batch(batch_idx: int, batch_df: pd.DataFrame, batch_log_path: str,
                  out_dir: str, project: str, folders: dict,
                  manifest_dir: str, post_batch: dict) -> None:
    """Replay pending uploads; a processing log is not an upload receipt."""
    if not post_batch.get("enabled", False):
        return
    manifest_path = os.path.join(manifest_dir, f"batch_{batch_idx:03d}_outputs.txt")
    if not os.path.exists(manifest_path):
        # Supports pre-receipt logs, provided their outputs are still local.
        output_paths = batch_output_paths(batch_df, out_dir, project, folders)
        manifest_path = write_batch_manifest(batch_idx, output_paths, manifest_dir)
    output_paths = Path(manifest_path).read_text().splitlines()
    output_paths = [path for path in output_paths if path]
    command = post_batch.get("command", [])
    if not command:
        return

    receipt_path = manifest_path + ".uploaded.json"
    signature = hashlib.sha256(
        Path(batch_log_path).read_bytes() + Path(manifest_path).read_bytes()
        + json.dumps(command).encode()
    ).hexdigest()
    uploaded = False
    if os.path.exists(receipt_path):
        uploaded = json.loads(Path(receipt_path).read_text()).get("signature") == signature
    if not uploaded:
        # Never turn a missing/deleted artifact into an apparently successful upload.
        required = set(output_paths)
        if folders.get("pt") is not None:
            for _, row in batch_df.iterrows():
                if row["fail_reason"] == GOOD_VIDEO_MSG:
                    required.add(os.path.join(
                        out_dir, folders["pt"], project, row["exam_dir"],
                        output_file_base_name(row["filename"]) + ".pt"))
        missing = [path for path in required if not os.path.isfile(path)]
        unlisted = required - set(output_paths)
        if missing or unlisted:
            raise RuntimeError(
                f"Batch {batch_idx} has no matching upload receipt and missing or "
                "unmanifested outputs. Restore/reprocess its local artifacts before "
                f"resuming. Missing: {missing}; unmanifested: {sorted(unlisted)}"
            )
        run_post_batch_command(command, batch_idx, manifest_path, batch_log_path, out_dir)
        atomic_write(receipt_path, json.dumps({"signature": signature}) + "\n")
    # A crash during cleanup is safe: the receipt permits cleanup on restart.
    if post_batch.get("delete_local_outputs", False):
        delete_batch_outputs(output_paths, out_dir)


# noinspection PyUnboundLocalVariable
def main():
    from ghlobus.utilities.data_utils import (
        read_spreadsheet, preprocess_video, create_video_df,
    )

    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_preproc_v9.yaml"

    # Configure the ArgumentParser
    cli_description = 'Preprocess DICOM or MP4 video files in into frame files. '
    cli_description += 'RGB frames are extracted; and they are: '
    cli_description += '- cropped to the active area; '
    cli_description += '- resized via the target `alpha`; '
    cli_description += '- padded to the final dimension; '
    cli_description += '- saved to the destination by exam folder.'

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # get the raw and csv file directory
    raw_dir = info['input']['raw_dir']
    project = info['input']['project']
    post_batch = info['processing'].get('post_batch', {})
    if post_batch.get('delete_local_outputs', False) and not post_batch.get('command', []):
        raise ValueError("post_batch.command is required when delete_local_outputs is true.")

    # create destination directory if not there
    out_dir = info['output']['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # add root output directory to info['image'] parameter dictionary
    info['image']['out_dir'] = out_dir

    # create data subfolders if not there;
    # and note name of subfolder in info['image'] parameter dictionary
    for fmt in ['raw', 'jpg', 'pt']:
        if fmt in info['output']['folders'] and info['output']['folders'][fmt] is not None:
            subfolder_path = os.path.join(out_dir, info['output']['folders'][fmt])
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path, exist_ok=True)
            info['image'][fmt] = info['output']['folders'][fmt]
        else:
            info['image'][fmt] = None

    parallel_backend = info["processing"].get("backend", "loky")

    # print information about preprocessing run
    print_string = f"Data will be written here {out_dir},\n"
    print_string += f"Prototype file is {info['output']['out_file']},\n"
    print_string += f"Using {info['processing']['cores']} CPU cores,\n"
    print_string += f"Parallel backend is {parallel_backend},\n"
    print_string += f"Target alpha is {info['image']['alpha']},\n"
    print_string += f"Number of channels is {info['image']['channels']},\n"
    print_string += f"Frame dimensions is {(info['image']['img_size'],)*2}."
    print(print_string)

    # define partial function with arguments specified by image dictionary
    # this single argument function is needed for multi-thread processing
    preprocess_with_args = partial(preprocess_video, **info['image'])

    # get name of metadata directory
    meta_path = os.path.join(out_dir, META_DIR)
    batch_manifest_dir = post_batch.get(
        'manifest_dir',
        os.path.join(out_dir, 'batch_manifests', project),
    )

    # read main instance table
    it_path = os.path.join(
        meta_path,
        info['input']['it_file'])
    it_df = read_spreadsheet(it_path)
    print(f"Read {it_df.shape[0]} rows from {info['input']['it_file']}.")

    # Drop duplicates
    it_df.drop_duplicates(subset='filename', inplace=True)
    it_df.reset_index(drop=True, inplace=True)

    # sort the instance metadata table by StudyID
    it_df.sort_values(by='StudyID', inplace=True)

    # create output folder names
    out_exam_folders = out_folder_name(it_df)
    uniq_exam_folders, uniq_inds = np.unique(out_exam_folders, return_index=True)

    # put output exam folder names into instance table and create output folders
    it_df['exam_dir'] = out_exam_folders

    # create output exam folders for each subfolder specified
    # NOTE: at least one data subfolder must be listed in the preproc yaml file
    one_folder_listed = False
    for fmt in ['jpg', 'pt']:
        if info['image'][fmt] is not None:
            one_folder_listed = True
            # Create all the unique folders
            for folder in tqdm(uniq_exam_folders,
                               desc=f"Creating {fmt} exam folders",
                               total=len(uniq_exam_folders)):
                out_path = os.path.join(
                    out_dir,
                    info['image'][fmt],
                    project,
                    folder)
                # create the directory
                os.makedirs(out_path, exist_ok=True)
            # more information output
            info_string = f"Created {len(uniq_exam_folders)} subfolders in "
            info_string += f"{os.path.join(out_dir, info['image'][fmt], project)}."
            print(info_string)

    # Make sure that at least one folder was listed in the preproc yaml file
    if not one_folder_listed:
        raise ValueError("At least one data subfolder must be listed in the preproc yaml file.")

    # Create raw folder if asked
    if info['image']['raw'] is not None:
        # Path for GOOD raw frames
        out_path = os.path.join(out_dir, info['image']['raw'], project)
        # create the directory
        os.makedirs(out_path, exist_ok=True)
        print(f"Created raw folder {out_path}.")
        # Path for BAD raw frames
        out_path = os.path.join(out_dir, info['image']['raw'], project + "_bad")
        # create the directory
        os.makedirs(out_path, exist_ok=True)
        print(f"Created raw folder {out_path}.")

    # Snap targets to exam boundaries, but never emit empty or reversed batches.
    n_videos = it_df.shape[0]
    batch_end = compute_batch_ends(uniq_inds, n_videos, info['processing']['batch_size'])
    n_batches = len(batch_end)
    batch_dfs = [None for _ in range(n_batches)]

    # Checkpoints belong to a specific input sheet and processing configuration.
    log_dir = os.path.join(out_dir, 'logs', project)
    os.makedirs(log_dir, exist_ok=True)
    state_path = os.path.join(log_dir, "run_state.json")
    fingerprint = hashlib.sha256(
        Path(it_path).read_bytes() + json.dumps(info, sort_keys=True).encode()
    ).hexdigest()
    if os.path.exists(state_path):
        previous = json.loads(Path(state_path).read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Input/configuration changed since preprocessing began. "
                             "Use a separate output directory for the new run.")

    log_files = sorted(glob(os.path.join(log_dir, "batch_*.csv")),
                       key=lambda path: int(Path(path).stem.split("_")[-1]))
    logged_indices = [int(Path(path).stem.split("_")[-1]) for path in log_files]
    if logged_indices != list(range(len(logged_indices))) or len(log_files) > n_batches:
        raise ValueError("Batch logs are not a contiguous prefix of this run; "
                         "restore the missing logs before resuming.")
    for batch_idx, log_file in enumerate(log_files):
        logged = pd.read_csv(log_file, dtype=str, keep_default_na=False)
        begin = 0 if batch_idx == 0 else batch_end[batch_idx - 1]
        expected = it_df.iloc[begin:batch_end[batch_idx]].reset_index(drop=True)
        # Also validate legacy logs that predate run fingerprints.
        columns = list(expected.columns)
        if not set(columns).issubset(logged.columns) or not logged[columns].equals(
                expected.fillna("").astype(str)):
            raise ValueError(f"Batch {batch_idx} log does not match the current input rows.")
        batch_dfs[batch_idx] = logged
    atomic_write(state_path, json.dumps({"fingerprint": fingerprint}) + "\n")
    for batch_idx, log_file in enumerate(log_files):
        handoff_batch(batch_idx, batch_dfs[batch_idx], log_file, out_dir, project,
                      info['output']['folders'], batch_manifest_dir, post_batch)
    start_batch = len(log_files)
    print(f"Starting process at batch {start_batch}.")

    # determine the beginning video index for the first batch
    if start_batch == 0:
        beg_video_idx = 0
    else:
        beg_video_idx = batch_end[start_batch - 1]

    # iterate over batches starting at start_batch
    for idx, end_video_idx in enumerate(tqdm(batch_end[start_batch:],
                                             desc="Processing batches",
                                             total=n_batches-start_batch)):
        # actual batch index depends on start_batch
        batch_idx = idx + start_batch
        print(f"Processing batch {batch_idx} from {beg_video_idx} to {end_video_idx-1}.")
        # grab the dataframe corresponding to this batch
        batch_it_df = it_df.iloc[beg_video_idx:end_video_idx].copy()
        batch_it_df.reset_index(inplace=True, drop=True)
        # reset beginning of next batch to where this batch ends
        beg_video_idx = end_video_idx

        # grab input paths and files for this batch
        in_paths = batch_it_df['relpath'].tolist()
        in_files = batch_it_df['filename'].tolist()
        # Use file-specific tags for guiding Doppler filtering
        tags = batch_it_df['tag'].tolist()

        # create source filepaths
        in_filepaths = [os.path.join(raw_dir, x, y)
                        for x, y in zip(in_paths, in_files)]
        # stack file-specific information into DataFrame for passing to parallel function
        file_info_df = pd.DataFrame({
            'in_filepath': in_filepaths,
            'file_type': batch_it_df['file_type'].tolist(),
            'project': [info['input']['project']] * len(in_filepaths),
            'exam_dir': batch_it_df['exam_dir'].tolist(),
            'tag': tags,
            'pdx': batch_it_df['PhysicalDeltaX'].tolist()})

        # Launch job; get list of results
        if info['processing']['cores'] > 1:
            # Configure parallelization
            par_job = Parallel(
                n_jobs=info["processing"]["cores"],
                backend=parallel_backend,
                verbose=True,
            )

            # process this batch in parallel to a list of dataframes
            batch_shape_df_lst = par_job(delayed(preprocess_with_args)(f)
                                         for _, f in file_info_df.iterrows())

            # clear parallelization to conserve memory
            del par_job

            # Stack this batch's output into a dataframe to be stacked with other batches
            batch_shape_df = pd.concat(batch_shape_df_lst, axis=0)
        else:
            # non-parallelized for debugging
            dummy_df = create_video_df(
                reason="dummy",
                shape_in=(0, 0, 0, 0),
                shape_out=(0, 0, 0, 0),
                bbox=(0, 0, 0, 0),
            )
            batch_shape_df = pd.DataFrame(columns=dummy_df.columns)

            # process this batch one file at a time
            for _, f in file_info_df.iterrows():
                video_shape_df = preprocess_with_args(f)
                batch_shape_df = pd.concat((batch_shape_df, video_shape_df), axis=0)

        # concatenate columns from prototype DF and shape DF
        batch_shape_df.reset_index(inplace=True, drop=True)
        batch_df = pd.concat((batch_it_df, batch_shape_df), axis=1)

        # write this batch's dataframe to a log file
        batch_log_path = os.path.join(log_dir, f"batch_{batch_idx:03d}.csv")
        atomic_write(batch_log_path, batch_df.to_csv(index=False))
        print(f"Saved log for batch {batch_idx} to {batch_log_path}.")

        handoff_batch(batch_idx, batch_df, batch_log_path, out_dir, project,
                      info['output']['folders'], batch_manifest_dir, post_batch)

        # append this batch's dataframe to the list of batch dataframes
        batch_dfs[batch_idx] = batch_df

    # stack the batch_dfs list into a single dataframe
    output_df = pd.concat(batch_dfs, axis=0)

    # sort by 'exam_dir` column
    output_df.sort_values(by=['exam_dir', 'filename'], inplace=True)
    # reset the index
    output_df.reset_index(drop=True, inplace=True)

    # count bad videos fail_reason is not good
    # noinspection PyUnresolvedReferences
    n_bad_videos = (output_df['fail_reason'].values != GOOD_VIDEO_MSG).sum()
    print(f"Filtered out {n_bad_videos} bad videos out of {output_df.shape[0]}.")

    # Save the output_df
    out_file_path = os.path.join(meta_path, info['output']['out_file'])
    output_df.to_csv(out_file_path, index=False)
    print(f"Saved output for {output_df.shape[0]} videos to {info['output']['out_file']}.")

    if post_batch.get('enabled', False) and post_batch.get('handoff_final_outputs', True):
        final_paths = final_output_paths(meta_path)
        final_manifest_path = write_batch_manifest(
            "final",
            final_paths,
            batch_manifest_dir,
        )
        command = post_batch.get('command', [])
        if command:
            run_post_batch_command(
                command,
                "final",
                final_manifest_path,
                out_file_path,
                out_dir,
            )


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()

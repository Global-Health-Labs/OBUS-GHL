#!/usr/bin/env python3
"""Upload one preprocessing batch manifest to a GCS output prefix."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--retries", default=3, type=int)
    parser.add_argument("--retry-delay", default=10, type=int)
    return parser.parse_args()


def relative_uri(destination: str, out_dir: Path, local_path: Path) -> str:
    relative_path = local_path.relative_to(out_dir)
    return "/".join([destination.rstrip("/"), *relative_path.parts])


def run(args: list[str], retries: int = 0, retry_delay: int = 0) -> None:
    last_error = None
    for attempt in range(retries + 1):
        try:
            subprocess.run(args, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_for = retry_delay * (attempt + 1)
            print(
                f"Command failed with exit code {exc.returncode}; "
                f"retrying in {sleep_for}s ({attempt + 1}/{retries}).",
                flush=True,
            )
            time.sleep(sleep_for)
    raise last_error


def copy_file(local_path: Path,
              remote_path: str,
              retries: int,
              retry_delay: int,
              ) -> None:
    try:
        run(["gcloud", "storage", "cp", str(local_path), remote_path],
            retries=retries,
            retry_delay=retry_delay)
        return
    except subprocess.CalledProcessError:
        temp_remote_path = f"{remote_path}.uploading-{os.getpid()}-{int(time.time())}"
        print(
            "Direct upload failed after retries; uploading via temporary "
            f"object {temp_remote_path}.",
            flush=True,
        )
        run(["gcloud", "storage", "cp", str(local_path), temp_remote_path],
            retries=retries,
            retry_delay=retry_delay)
        run(["gcloud", "storage", "mv", temp_remote_path, remote_path],
            retries=retries,
            retry_delay=retry_delay)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    destination = args.destination.rstrip("/")

    with args.manifest.open() as handle:
        output_paths = [Path(line.strip()) for line in handle if line.strip()]

    for local_path in output_paths:
        remote_path = relative_uri(destination, out_dir, local_path.resolve())
        if local_path.is_dir():
            run(["gcloud", "storage", "rsync", "--recursive", str(local_path), remote_path],
                retries=args.retries,
                retry_delay=args.retry_delay)
        else:
            copy_file(local_path, remote_path, args.retries, args.retry_delay)

    log_remote_path = relative_uri(destination, out_dir, args.batch_log.resolve())
    copy_file(args.batch_log, log_remote_path, args.retries, args.retry_delay)


if __name__ == "__main__":
    main()

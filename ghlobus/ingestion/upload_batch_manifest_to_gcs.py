#!/usr/bin/env python3
"""Upload one preprocessing batch manifest to a GCS output prefix."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True)
    return parser.parse_args()


def relative_uri(destination: str, out_dir: Path, local_path: Path) -> str:
    relative_path = local_path.relative_to(out_dir)
    return "/".join([destination.rstrip("/"), *relative_path.parts])


def run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    destination = args.destination.rstrip("/")

    with args.manifest.open() as handle:
        output_paths = [Path(line.strip()) for line in handle if line.strip()]

    for local_path in output_paths:
        remote_path = relative_uri(destination, out_dir, local_path.resolve())
        if local_path.is_dir():
            run(["gcloud", "storage", "rsync", "--recursive", str(local_path), remote_path])
        else:
            run(["gcloud", "storage", "cp", str(local_path), remote_path])

    log_remote_path = relative_uri(destination, out_dir, args.batch_log.resolve())
    run(["gcloud", "storage", "cp", str(args.batch_log), log_remote_path])


if __name__ == "__main__":
    main()

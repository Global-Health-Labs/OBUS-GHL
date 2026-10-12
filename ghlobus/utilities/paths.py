"""
paths.py

Paths for the full Famli (v9) and NEJM (v4) Training and Inference DataModules.

Author: Daniel Shea
        Olivia Zahn
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os

v4_dataset_dir = "/data/ML-Project-Data/FAMLI/Datasets/v4"
v9_dataset_dir = "/data/ML-Project-Data/FAMLI/Datasets/v9"
v9_5fold_dir = "/data/ML-Project-Data/FAMLI/Datasets/v9/splits/TWIN_5fold"
GA_csv_template = "/data/ML-Project-Data/FAMLI/Datasets/v4/{}.csv"
FP_csv_template = "/data/ML-Project-Data/FAMLI/Datasets/v4/{}_w_lie.csv"


def output_file_base_name(filename: str) -> str:
    """Strip media extensions, preserving extensionless dotted DICOM UIDs."""
    name = os.path.basename(filename)
    if name.lower().endswith((".dcm", ".mp4")):
        return os.path.splitext(name)[0]
    return name

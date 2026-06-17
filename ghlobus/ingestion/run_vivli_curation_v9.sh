#! /bin/bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/workspace/code/OBUS-GHL"

cd "$(dirname "$0")"

python vivli_curate_v9.py --yaml configs/VIVLI_FAMLI3_sr_crf_v9.yaml
python vivli_curate_v9.py --yaml configs/VIVLI_FAMLI3_prototype_curated_v9.yaml

python efw_data_selection_v9.py --yaml configs/VIVLI_FAMLI3_efw_data_v9.yaml
python twin_data_selection_v9.py --yaml configs/VIVLI_FAMLI3_twin_data_v9.yaml

python vivli_task_split_v9.py --yaml configs/VIVLI_FAMLI3_ga_split_v9.yaml
python vivli_task_split_v9.py --yaml configs/VIVLI_FAMLI3_fp_split_v9.yaml
python vivli_task_split_v9.py --yaml configs/VIVLI_FAMLI3_efw_split_v9.yaml
python vivli_task_split_v9.py --yaml configs/VIVLI_FAMLI3_twin_split_v9.yaml

#!/bin/bash
set -x
ulimit -c 0



source venv/bin/activate

# generate data and then rename data
python -m datasets.encode_openx_dataset --episode_cnt 1000000 --dataset_name kaist_nonprehensile_converted_externally_to_rlds --data_split train --root_dir data
# generate data and then rename data
python -m datasets.encode_openx_dataset --episode_cnt 1000000 --dataset_name kaist_nonprehensile_converted_externally_to_rlds --data_split val --root_dir data

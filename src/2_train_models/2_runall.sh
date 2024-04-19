#!/bin/bash

# This script does *ALL* the model training.

set -e

script_dir=`dirname $0`

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_types=( "K562" "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )

mkdir -p "logs"


# train the main models -- all cell types, all folds

for cell_type in "${cell_types[@]}"; do
  "$script_dir/2_train_main_models_all_folds.sh" "$cell_type" "$GPU"
done

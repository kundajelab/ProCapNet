#!/bin/bash

# This script does *ALL* the model training.

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_types=( "K562" "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"


# train the main models -- all cell types, all folds

for cell_type in "${cell_types[@]}"; do
  ./2_train_main_models_all_folds.sh "$cell_type" "$GPU"
done

# everything below is optional if you aren't specifically
# replicating the analyses in the paper that they're for

# for promoters vs. enhancers analysis
./2_train_models_all_folds_promoters_only.sh "$GPU"

# RAMPAGE models
./2_train_models_replicates_K562_all_folds.sh "$GPU"

# replicate models trained as a baseline for these analyses
./2_train_models_rampage_K562_all_folds.sh "$GPU"

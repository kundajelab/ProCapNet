#!/bin/bash

# This script generates model predictions and performance stats
# for *ALL* the models.

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

for cell_type in "${cell_types[@]}"; do
  ./3_eval_models_main.sh "$cell_type" "$GPU"
done

# everything below is optional if you aren't specifically
# replicating the analyses in the paper that they're for

# for promoters vs. enhancers analysis
./3_eval_models_promoters_only.sh "$GPU"

# RAMPAGE models
./3_eval_models_rampage.sh "$GPU"

# replicate models trained as a baseline for these analyses
./3_eval_models_replicates.sh "$GPU"



exit 0

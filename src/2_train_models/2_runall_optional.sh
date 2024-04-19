#!/bin/bash

# everything below is optional if you aren't specifically
# replicating the analyses in the paper that they're for

set -e

script_dir=`dirname $0`

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

mkdir -p "logs"


# for promoters vs. enhancers analysis
"$script_dir/2_train_models_all_folds_promoters_only.sh" "$GPU"

# RAMPAGE models
"$script_dir/2_train_models_replicates_K562_all_folds.sh" "$GPU"

# replicate models trained as a baseline for these analyses
"$script_dir/2_train_models_rampage_K562_all_folds.sh" "$GPU"

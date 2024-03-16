#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1
cell_type="K562"
model_type="strand_merged_umap_replicate"
data_type="procap"

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"

for fold in "${folds[@]}"; do
  python train.py "$cell_type" "$model_type" "$data_type" "$fold" "$GPU" | tee "logs/${cell_type}_${fold}_replicate.log"
done

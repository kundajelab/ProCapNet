#!/bin/bash

set -e

GPU="MIG-166d7783-762d-5f61-b31c-549eb4e0fba0"
model_type="promoters_only_strand_merged_umap"
data_type="procap"

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"

for fold in "${folds[@]}"; do
  python train.py "K562" "$model_type" "$data_type" "$fold" "$GPU" | tee "logs/${cell_type}_${fold}_promoters_only.log"
done

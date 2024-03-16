#!/bin/bash

set -e

GPU="MIG-40f43250-998e-586a-ac37-d6520e92590f"
model_type="strand_merged_umap"
data_type="procap"

cell_types=( "K562" "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"

for cell_type in "${cell_types[@]}"; do
  for fold in "${folds[@]}"; do
    python train.py $cell_type "$model_type" "$data_type" "$fold" "$GPU" | tee "logs/${cell_type}_${fold}.log"
  done
done

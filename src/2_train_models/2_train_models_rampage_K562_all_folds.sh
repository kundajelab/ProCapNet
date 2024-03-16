#!/bin/bash

set -e

GPU="MIG-40f43250-998e-586a-ac37-d6520e92590f"
model_type="strand_merged_umap"
data_type="rampage"

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"

for fold in "${folds[@]}"; do
  python train.py "K562" "$model_type" "$data_type" "$fold" "$GPU" | tee "logs/${cell_type}_${fold}_rampage.log"
done



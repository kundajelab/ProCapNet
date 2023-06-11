#!/bin/bash

GPU="MIG-166d7783-762d-5f61-b31c-549eb4e0fba0"
model_type="strand_merged_umap"

#cell_types=( "K562" "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )
cell_types=( "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )

folds=( 1 2 3 4 5 6 7 )

for cell_type in "${cell_types[@]}"; do
  for fold in "${folds[@]}"; do
    python train.py $cell_type "$model_type" $fold "$GPU" | tee "logs/${cell_type}_${fold}.log"
  done
done



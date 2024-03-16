#!/bin/bash

# This script does *ALL* the model training.

# For training individual model types, see these scripts:
# 2_train_main_models_all_folds_all_cells.sh   <-- main ProCapNet models
# 2_train_models_all_folds_promoters_only.sh
# 2_train_models_rampage_K562_all_folds.sh
# 2_train_models_replicates_all_folds_K562.sh


set -e

GPU="MIG-40f43250-998e-586a-ac37-d6520e92590f"  # change

cell_types=( "K562" "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"


# train the main models -- all cell types, all folds

for cell_type in "${cell_types[@]}"; do
  for fold in "${folds[@]}"; do
    python train.py "$cell_type" "strand_merged_umap" "procap" "$fold" "$GPU" | tee "logs/${cell_type}_${fold}.log"
  done
done


# train models on only promoters (enhancers vs. promoters analysis)

for fold in "${folds[@]}"; do
  python train.py "K562" "promoters_only_strand_merged_umap" "procap" "$fold" "$GPU" | tee "logs/K562_${fold}_promoters_only.log"
done


# train "replicate" models with a diff. random seed as a baseline for reproducibility

for fold in "${folds[@]}"; do
  python train.py "K562" "strand_merged_umap_replicate" "procap" "$fold" "$GPU" | tee "logs/K562_${fold}_replicate.log"
done


# train models on RAMPAGE data

for fold in "${folds[@]}"; do
  python train.py "K562" "strand_merged_umap" "rampage" "$fold" "$GPU" | tee "logs/K562_${fold}_rampage.log"
done

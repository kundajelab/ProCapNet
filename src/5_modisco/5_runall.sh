#!/bin/bash

set -e

model_type="strand_merged_umap"

tasks=( "profile" "counts" )

mkdir -p logs

cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

for cell_type in "${cell_types[@]}"; do
  for task in "${tasks[@]}"; do
    python modisco.py "$cell_type" "$model_type" "$task" | tee "logs/${cell_type}_${task}.log"
  done
done


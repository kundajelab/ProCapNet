#!/bin/bash

cell_type="K562"
model_type="strand_merged_umap"

tasks=( "profile" "counts" )

tasks=( "counts" )

mkdir -p logs

for task in "${tasks[@]}"; do
  python modisco.py "$cell_type" "$model_type" "$task" | tee "logs/${cell_type}_${task}.log"
done

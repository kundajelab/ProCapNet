#!/bin/bash

set -e

model_type="cell_union_negs_strand_merged_umap"

tasks=( "profile" "counts" )

mkdir -p logs

cell_type="K562"

for task in "${tasks[@]}"; do
  python modisco.py "$cell_type" "$model_type" "$task" | tee "logs/${cell_type}_${task}_cell_union_negs.log"
done

echo "Done!"

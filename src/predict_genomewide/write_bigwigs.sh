#!/bin/bash

set -e

mkdir -p logs

cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

for cell_type in "${cell_types[@]}"; do
  echo "$cell_type"
  python write_preds_to_bigwigs_genomewide.py "$cell_type" | tee -a "logs/t2t_${cell_type}_write_bigwig.txt"
done

echo "End of bigwig-making script."

exit 0

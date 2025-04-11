#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Expecting chromosome, and GPU ID as input args. Exiting." && exit 1
fi

chrom=$1
GPU=$2

mkdir -p logs

cell_types=( "CACO2" "CALU3" "HUVEC" "MCF10A" )
#cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

for cell_type in "${cell_types[@]}"; do
  echo "$cell_type" "$chrom"
  python predict_whole_chromosome_lowermem.py "$cell_type" "$chrom" "$GPU" | tee -a "logs/t2t_${cell_type}_${chrom}.txt"
done

echo "End of predict script."

exit 0

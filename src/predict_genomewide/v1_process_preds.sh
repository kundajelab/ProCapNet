#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Expecting cell type and chromosome as input args. Exiting." && exit 1
fi

cell_type=$1
chrom=$2

mkdir -p logs

echo "$cell_type" "$chrom"
python v1_process_preds_whole_chromosome.py "$cell_type" "$chrom" | tee -a "logs/process_${cell_type}_${chrom}.txt"

echo "End of processing script."

exit 0

#!/bin/bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Expecting cell type, chromosome, and GPU ID as input args. Exiting." && exit 1
fi

cell_type=$1
chrom=$2
GPU=$3

mkdir -p logs

echo "$cell_type" "$chrom"
python raw_predict_whole_chromosomes.py "$cell_type" "$chrom" "$GPU" | tee -a "logs/raw_preds_${cell_type}_${chrom}.txt"

echo "End of predict script."

exit 0

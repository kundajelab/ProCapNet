#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Expecting cell_type and GPU as input args. Exiting." && exit 1
fi

cell_type=$1
GPU=$2
model_type="strand_merged_umap"
data_type="procap"

folds=( 1 2 3 4 5 6 7 )

mkdir -p "logs"

for fold in "${folds[@]}"; do
  python train.py "$cell_type" "$model_type" "$data_type" "$fold" "$GPU" | tee "logs/${cell_type}_${fold}.log"
done

echo "Done training, all folds."

exit 0

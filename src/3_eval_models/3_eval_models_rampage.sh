#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_type="K562"
model_type="strand_merged_umap"
data_type="rampage"

timestamps=( "2023-09-13_21-57-36" "2023-09-13_23-37-50" "2023-09-14_01-07-10" "2023-09-14_02-28-40" "2023-09-14_03-25-49" "2023-09-14_04-44-39" "2023-09-14_05-56-16" )

mkdir -p "logs"

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python eval.py "$cell_type" "$model_type" "$data_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/eval_${fold}_rampage.log"
done

python merge_prediction_tracks.py "$cell_type" "$model_type" "$data_type" "${timestamps[*]}"


#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_type="K562"
model_type="strand_merged_umap_replicate"
data_type="procap"

# K562 replicate model IDs
timestamps=( "2024-02-18_01-26-07" "2024-02-18_03-00-29" "2024-02-18_04-50-55" "2024-02-18_06-47-37" "2024-02-18_08-43-12" "2024-02-18_10-37-54" "2024-02-18_12-18-54" )

mkdir -p logs

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python eval.py "$cell_type" "$model_type" "$data_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}_replicate.log"
done

python merge_prediction_tracks.py "$cell_type" "$model_type" "$data_type" "${timestamps[*]}"




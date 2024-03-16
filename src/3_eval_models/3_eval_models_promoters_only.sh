#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_type="K562"
model_type="promoters_only_strand_merged_umap"
data_type="procap"

timestamps=( "2024-01-11_04-16-09" "2024-01-11_05-41-04" "2024-01-11_06-46-15" "2024-01-11_07-54-30" "2024-01-11_09-03-11" "2024-01-11_10-25-53" "2024-01-11_11-44-28" )

mkdir -p "logs"

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python eval.py "$cell_type" "$model_type" "$data_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}_promoters_only.log"
done

python merge_prediction_tracks.py "$cell_type" "$model_type" "$data_type" "${timestamps[*]}"




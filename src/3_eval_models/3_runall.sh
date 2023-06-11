#!/bin/bash

GPU="3"
cell_type="K562"
model_type="strand_merged_umap"

timestamps=( "2023-05-29_15-51-40" "2023-05-29_15-58-41" "2023-05-29_15-59-09" "2023-05-30_01-40-06" "2023-05-29_23-21-23" "2023-05-29_23-23-45" "2023-05-29_23-24-11" )

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  #python eval.py "$cell_type" "$model_type" "$j" "${timestamps[$i]}" "$GPU"
done

python merge_prediction_tracks.py "$cell_type" "$model_type" "${timestamps[*]}"




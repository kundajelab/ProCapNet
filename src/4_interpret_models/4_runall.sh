#!/bin/bash

GPU="3"
cell_type="K562"
model_type="strand_merged_umap"

timestamps=( "2023-05-29_15-51-40" "2023-05-29_15-58-41" "2023-05-29_15-59-09" "2023-05-30_01-40-06" "2023-05-29_23-21-23" "2023-05-29_23-23-45" "2023-05-29_23-24-11" )

mkdir -p logs

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  #python deepshap.py "$cell_type" "$model_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}.log"
done

python merge_deepshap_tracks.py "$cell_type" "$model_type" "${timestamps[*]}"




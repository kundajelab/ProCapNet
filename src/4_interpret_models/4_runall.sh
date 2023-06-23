#!/bin/bash

GPU="MIG-f80e9374-504a-571b-bac0-6fb00750db4c"
cell_type="CACO2"
model_type="strand_merged_umap"

# K562
#timestamps=( "2023-05-29_15-51-40" "2023-05-29_15-58-41" "2023-05-29_15-59-09" "2023-05-30_01-40-06" "2023-05-29_23-21-23" "2023-05-29_23-23-45" "2023-05-29_23-24-11" )
# A673
#timestamps=( "2023-06-11_20-11-32" "2023-06-11_23-42-00" "2023-06-12_03-29-06" "2023-06-12_07-17-43" "2023-06-12_11-10-59" "2023-06-12_14-36-40" "2023-06-12_17-26-09" )
# CACO2
timestamps=( "2023-06-12_21-46-40" "2023-06-13_01-28-24" "2023-06-13_05-06-53" "2023-06-13_08-52-39" "2023-06-13_13-12-09" "2023-06-13_16-40-41" "2023-06-13_20-08-39" )



mkdir -p logs

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python deepshap.py "$cell_type" "$model_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}.log"
done

python merge_deepshap_tracks.py "$cell_type" "$model_type" "${timestamps[*]}"




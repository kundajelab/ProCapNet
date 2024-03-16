#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Expecting cell type and GPU as input args. Exiting." && exit 1
fi

cell_type=$1
GPU=$2
model_type="strand_merged_umap"
data_type="procap"


if [ "$cell_type" == "K562" ]; then
  timestamps=( "2023-05-29_15-51-40" "2023-05-29_15-58-41" "2023-05-29_15-59-09" "2023-05-30_01-40-06" "2023-05-29_23-21-23" "2023-05-29_23-23-45" "2023-05-29_23-24-11" )
elif [ "$cell_type" == "A673" ]; then
  timestamps=( "2023-06-11_20-11-32" "2023-06-11_23-42-00" "2023-06-12_03-29-06" "2023-06-12_07-17-43" "2023-06-12_11-10-59" "2023-06-12_14-36-40" "2023-06-12_17-26-09" )
elif [ "$cell_type" == "CACO2" ]; then
  timestamps=( "2023-06-12_21-46-40" "2023-06-13_01-28-24" "2023-06-13_05-06-53" "2023-06-13_08-52-39" "2023-06-13_13-12-09" "2023-06-13_16-40-41" "2023-06-13_20-08-39" )
elif [ "$cell_type" == "CALU3" ]; then
  timestamps=( "2023-06-14_00-43-44" "2023-06-14_04-26-48" "2023-06-14_09-34-26" "2023-06-14_13-03-59" "2023-06-14_17-22-28" "2023-06-14_21-03-11" "2023-06-14_23-58-36" )
elif [ "$cell_type" == "HUVEC" ]; then
  timestamps=( "2023-06-16_21-59-35" "2023-06-17_00-20-34" "2023-06-17_02-17-07" "2023-06-17_04-27-08" "2023-06-17_06-42-19" "2023-06-17_09-16-24" "2023-06-17_11-09-38" )
elif [ "$cell_type" == "MCF10A" ]; then
  timestamps=( "2023-06-15_06-07-40" "2023-06-15_10-37-03" "2023-06-15_16-23-56" "2023-06-15_21-44-32" "2023-06-16_03-47-46" "2023-06-16_09-41-26" "2023-06-16_15-07-01" )
else
  echo "Incorrect cell type argument. Exiting." && exit 1
fi

mkdir -p logs

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python eval.py "$cell_type" "$model_type" "$data_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}.txt"
done

python merge_prediction_tracks.py "$cell_type" "$model_type" "$data_type" "${timestamps[*]}"




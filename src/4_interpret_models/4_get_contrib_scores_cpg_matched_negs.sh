#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_type="K562"
model_type="cpg_matched_negs_strand_merged_umap"
data_type="procap"

timestamps=( "2024-05-13_22-24-42" "2024-05-14_00-09-58" "2024-05-14_02-05-23" "2024-05-14_03-43-32" "2024-05-14_05-37-19" "2024-05-14_08-01-14" "2024-05-14_09-40-32" )

mkdir -p "logs"

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python deepshap.py "$cell_type" "$model_type" "$data_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}_cpg_matched_negs.log"
done

python merge_deepshap_tracks.py "$cell_type" "$model_type" "$data_type" "${timestamps[*]}"




#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expecting GPU as input arg. Exiting." && exit 1
fi

GPU=$1

cell_type="K562"
model_type="cell_union_negs_strand_merged_umap"
data_type="procap"

timestamps=( "2024-05-13_20-47-13" "2024-05-13_22-21-35" "2024-05-14_00-06-56" "2024-05-14_02-28-11" "2024-05-14_04-41-02" "2024-05-14_07-01-04" "2024-05-14_08-55-24" )

mkdir -p "logs"

for i in "${!timestamps[@]}"; do
  j=$(($i + 1))
  python deepshap.py "$cell_type" "$model_type" "$data_type" "$j" "${timestamps[$i]}" "$GPU" | tee "logs/${cell_type}_${j}_cell_union_negs.log"
done

python merge_deepshap_tracks.py "$cell_type" "$model_type" "$data_type" "${timestamps[*]}"




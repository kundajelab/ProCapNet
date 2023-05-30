#!/bin/bash

set -e

cell_types=( "K562" "A673" "CACO2" "CALU3" "MCF10A" "HUVEC" )

for cell_type in "${cell_types[@]}"; do
  echo "Processing data for ${cell_type}."

  ./1.0_process_bams.sh "$cell_type"

  ./1.1_make_pseudoreps.sh "$cell_type"

  ./1.2_process_peaks.sh "$cell_type"

  ./1.3_process_dnase_peaks.sh "$cell_type"
done



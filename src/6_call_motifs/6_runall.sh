#!/bin/bash

# this script calls motif hits for both tasks at once

cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

for cell_type in "${cell_types[@]}"; do
  python call_motifs_script.py "$cell_type"
done

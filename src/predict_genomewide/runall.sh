#!/bin/bash

set -e

### Note: don't actually run this -- this script is just
# a record of how / in what order to run the other scripts.


# needed for predicting with ProCapNet
GPU="1"

cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

chroms=( "chr1" "chr2" "chr3" "chr4" "chr5" "chr6" "chr7" "chr8" "chr9" "chr10" "chr11" "chr12" "chr13" "chr14" "chr15" "chr16" "chr17" "chr18" "chr19" "chr20" "chr21" "chr22" "chrX" "chrY")


for chrom in "${chroms[@]}"; do
    for cell_type in "${cell_types[@]}"; do
        ./predict.sh "$cell_type" "$chrom" "$GPU" || exit 1
        ./process_preds.sh "$cell_type" "$chrom"  || exit 1
        
        #./delete_raw_files.sh "$cell_type" "$chrom"
    done
done

exit 0

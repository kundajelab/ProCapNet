#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_root=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_root"


data_type="procap"
cell_type="K562"

ccres_bed="$proj_root/annotations/$cell_type/cCREs.bed.gz"

processed_data_dir="$proj_root/data/$data_type/processed/$cell_type"
peaks_bed="$processed_data_dir/peaks.bed.gz"

# make file of just promoters to filter for overlap with

promoters_bed="$proj_root/annotations/$cell_type/promoters.bed.gz"
zcat "$ccres_bed" | grep "PLS" | gzip -nc > "$promoters_bed"

# then filter peaks

peaks_tmp=` echo "$peaks_bed" | sed 's/.bed.gz/.tmp.bed.gz/' `
peaks_promoters_bed="$processed_data_dir/peaks_promoters_only.bed.gz"

zcat "$peaks_bed" | awk -v OFS="\t" '{ print $1, int(($2 + $3) / 2) - 500, int(($2 + $3) / 2) + 500, $2, $3, $4, $5, $6, $7, $8 }' | gzip -nc > "$peaks_tmp"
bedtools intersect -a "$peaks_tmp" -b "$promoters_bed" -wa -u | awk -v OFS="\t" '{ print $1, $4, $5, $6, $7, $8, $9, $10 }' | gzip -nc  > "$peaks_promoters_bed"

num_peaks_before=`zcat "$peaks_bed" | wc -l`
echo "Total peaks: $num_peaks_before"
num_peaks_after=`zcat "$peaks_promoters_bed" | wc -l`
echo "Peaks that overlap promoters: $num_peaks_after"


echo "Splitting peaks by train/val/test chroms..."

python _split_peaks_train_val_test.py "$peaks_promoters_bed"

rm "$peaks_tmp"

echo "Done."

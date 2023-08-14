#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_root=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_root"


data_type="procap"
cell_type=$1

if [[ -z "$cell_type" ]] ; then
  echo "Error: give cell type as argument" && exit 1
fi



echo "Processing DNase peaks in $cell_type..."

annots_dir="$proj_root/annotations/$cell_type"
dnase_peaks_bed="$annots_dir/DNase_peaks.bed.gz"

if [ ! -f "$dnase_peaks_bed" ]; then
  echo "Missing DNase peaks file: $dnase_peaks_bed" && exit 1
fi

processed_data_dir="$proj_root/data/$data_type/processed/$cell_type"
procap_peaks_bed="$processed_data_dir/peaks.bed.gz"

if [ ! -f "$procap_peaks_bed" ]; then
  echo "Missing train peaks file: $procap_peaks_bed" && exit 1
fi

dnase_peaks_no_procap_overlap_bed="$processed_data_dir/dnase_peaks_no_procap_overlap.bed.gz"

# First, filter out any DNase peaks overlapping PRO-cap peaks (or within 500bp of their center)
# Then, merge the remaining peaks so they're not so redundant
# Also for K562 toss out anything on Y chromosome bc K62 is female

if [[ "$cell_type" == "K562" ]]; then
  zcat "$procap_peaks_bed" | grep -v "chrM" | awk -v OFS="\t" '{ print $1, int(($2 + $3) / 2) - 500, int(($2 + $3) / 2) + 500 }' | sort -k1,1 -k2,2n | bedtools intersect -v -a "$dnase_peaks_bed" -b stdin | bedtools merge -i stdin -d 100 | grep -v "chrY" | gzip -nc > "$dnase_peaks_no_procap_overlap_bed"
else
  zcat "$procap_peaks_bed" | grep -v "chrM" | awk -v OFS="\t" '{ print $1, int(($2 + $3) / 2) - 500, int(($2 + $3) / 2) + 500 }' | sort -k1,1 -k2,2n | bedtools intersect -v -a "$dnase_peaks_bed" -b stdin | bedtools merge -i stdin -d 100 | gzip -nc > "$dnase_peaks_no_procap_overlap_bed"
fi


python _split_peaks_train_val_test.py "$dnase_peaks_no_procap_overlap_bed"

num_peaks_before=`zcat "$dnase_peaks_bed" | wc -l`
echo "Total DNase peaks: $num_peaks_before"
num_peaks_after=`zcat "$dnase_peaks_no_procap_overlap_bed" | wc -l`
echo "DNase peaks that don't overlap training set: $num_peaks_after"

exit 0


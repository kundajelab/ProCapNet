#!/bin/bash

set -e

get_proj_root () {
  # requires that file .root.txt is located in top level of project directory.
  # if you don't want to do this, hard-code or input the project dir as an arg.
  local DIR=$(pwd)
  while [ ! -z "$DIR" ] && [ ! -f "$DIR/.root.txt" ]; do DIR="${DIR%\/*}"; done
  if [ -z "$DIR" ]; then script=`basename "$0"` && echo "ERROR: could not determine project directory in script $script. Exiting." >&2 && exit 1; fi
  echo "$DIR"
}

proj_root=`get_proj_root`

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
train_peaks_bed="$processed_data_dir/peaks_uni_and_bi_train.bed.gz"

if [ ! -f "$train_peaks_bed" ]; then
  echo "Missing train peaks file: $train_peaks_bed" && exit 1
fi

dnase_peaks_no_train_overlap_bed="$processed_data_dir/dnase_peaks_no_train_peaks.bed.gz"

# First, filter out any DNase peaks overlapping PRO-cap peaks (or within 500bp of their center)
# Then, merge the remaining peaks so they're not so redundant
# Also toss out anything on Y chromosome bc some of these cell types are female anyways

if [[ "$cell_type" == "K562" ]]; then
  zcat "$train_peaks_bed" | awk -v OFS="\t" '{ print $1, int(($2 + $3) / 2) - 500, int(($2 + $3) / 2) + 500 }' | sort -k1,1 -k2,2n | bedtools intersect -v -a "$dnase_peaks_bed" -b stdin | bedtools merge -i stdin -d 100 | grep -v "chrY" | gzip -nc > "$dnase_peaks_no_train_overlap_bed"
else
  zcat "$train_peaks_bed" | awk -v OFS="\t" '{ print $1, int(($2 + $3) / 2) - 500, int(($2 + $3) / 2) + 500 }' | sort -k1,1 -k2,2n | bedtools intersect -v -a "$dnase_peaks_bed" -b stdin | bedtools merge -i stdin -d 100 | gzip -nc > "$dnase_peaks_no_train_overlap_bed"
fi

# Finally, combine DNase peaks with usual training set
# (ok actually this is not the final file... but it is the whole training set in one file)

out_bed="$processed_data_dir/peaks_uni_and_bi_train_and_DNase_peaks.bed.gz"
zcat "$dnase_peaks_no_train_overlap_bed" "$train_peaks_bed" | shuf | gzip -nc > "$out_bed"

# THIS will make the final file needed for train_multi_source.py
python _split_peaks_train_val_test.py "$dnase_peaks_no_train_overlap_bed"

num_peaks_before=`zcat "$dnase_peaks_bed" | wc -l`
echo "Total DNase peaks: $num_peaks_before"
num_peaks_after=`zcat "$dnase_peaks_no_train_overlap_bed" | wc -l`
echo "DNase peaks that don't overlap training set: $num_peaks_after"
num_peaks_final=`zcat "$out_bed" | wc -l`
echo "DNase peaks after merging overlaps/near-overlaps: $num_peaks_final"

exit 0


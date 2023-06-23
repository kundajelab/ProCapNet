#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_root=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_root"


data_type="procap"


echo "Combining peaks across cell types..."

processed_data_dir="$proj_root/data/$data_type/processed"

all_cell_type_train_peaks=`find "$processed_data_dir" -mindepth 1 -maxdepth 1 -type d | sed -e 's|$|/peaks_fold1_train.bed.gz|'`

echo "Training peak set files to combine:"
echo "$all_cell_type_train_peaks"

# First, combine the peak sets across all cell types into one file
# Notably, I drop everything from the Y chromosome because you'll get errors for models training on female data

all_peaks_tmp="$processed_data_dir/union_peaks_fold1_train.tmp.bed"
zcat $all_cell_type_train_peaks | grep -v "chrY" | sort -k1,1 -k2,2n > "$all_peaks_tmp"

# Then merge the peaks if they overlap, or if they are < 100bp apart

all_peaks="$processed_data_dir/union_peaks_fold1_train.bed.gz"
bedtools merge -i "$all_peaks_tmp" -d 100 | shuf | gzip -nc > "$all_peaks"

num_peaks_before=`wc -l < "$all_peaks_tmp"`
echo "Train peaks in union before merge: $num_peaks_before"
num_peaks_after=`zcat "$all_peaks" | wc -l`
echo "Train peaks in union after merge: $num_peaks_after"

#rm "$all_peaks_tmp"



all_cell_type_val_peaks=`find "$processed_data_dir" -mindepth 1 -maxdepth 1 -type d | sed -e 's|$|/peaks_fold1_val.bed.gz|'`

echo "Val peak set files to combine:"
echo "$all_cell_type_val_peaks"

# First, combine the peak sets across all cell types into one file
# Notably, I drop everything from the Y chromosome because you'll get errors for models training on female data

all_peaks_tmp="$processed_data_dir/union_peaks_fold1_val.tmp.bed"
zcat $all_cell_type_val_peaks | grep -v "chrY" | sort -k1,1 -k2,2n > "$all_peaks_tmp"

# Then merge the peaks if they overlap, or if they are < 100bp apart

all_peaks="$processed_data_dir/union_peaks_fold1_val.bed.gz"
bedtools merge -i "$all_peaks_tmp" -d 100 | shuf | gzip -nc > "$all_peaks"

num_peaks_before=`wc -l < "$all_peaks_tmp"`
echo "Val peaks in union before merge: $num_peaks_before"
num_peaks_after=`zcat "$all_peaks" | wc -l`
echo "Val peaks in union after merge: $num_peaks_after"

rm "$all_peaks_tmp"



exit 0


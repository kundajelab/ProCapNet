#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_root=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_root"


data_type="procap"


echo "Combining peaks across cell types..."

processed_data_dir="$proj_root/data/$data_type/processed"

all_cell_type_peaks=`find "$processed_data_dir" -mindepth 1 -maxdepth 1 -type d | sed -e 's|$|/peaks.bed.gz|'`

echo "Peak set files to combine:"
echo "$all_cell_type_peaks"

all_peaks_tmp="$processed_data_dir/union_peaks.tmp.bed"
all_peaks="$processed_data_dir/union_peaks.bed.gz"

# First, combine the peak sets across all cell types into one file
# drop everything from sex chromosomes because the samples are not all the same sex

zcat $all_cell_type_peaks | grep -v "chrY" | grep -v "chrX" | sort -k1,1 -k2,2n > "$all_peaks_tmp"

# Then merge the peaks if they overlap or are < 100bp apart

bedtools merge -i "$all_peaks_tmp" -d 100 | shuf | gzip -nc > "$all_peaks"

num_peaks_before=`wc -l < "$all_peaks_tmp"`
echo "Peaks in union before merge: $num_peaks_before"
num_peaks_after=`zcat "$all_peaks" | wc -l`
echo "Peaks in union after merge: $num_peaks_after"

rm "$all_peaks_tmp"

exit 0


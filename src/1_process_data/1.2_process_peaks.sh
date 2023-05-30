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


echo "Processing peaks for $cell_type..."

raw_data_dir="$proj_root/data/$data_type/raw/$cell_type"
processed_data_dir="$proj_root/data/$data_type/processed/$cell_type"

if [[ ! -d "$raw_data_dir" ]]; then
  echo "Raw data directory for this cell type not found, exiting: $raw_data_dir" && exit 1
fi

# we'll save all the new files we create in this directory
mkdir -p "$processed_data_dir"


# These files were downloaded from ENCODE (see download_data.sh)
raw_uni_peaks="$raw_data_dir/peaks.uni.bed.gz"
raw_bi_peaks="$raw_data_dir/peaks.bi.bed.gz"

if [ ! -f "$raw_uni_peaks" ] || [ ! -f "$raw_bi_peaks" ]; then
  echo "Missing uni- or bi-directional peak file: $raw_uni_peaks, $raw_bi_peaks" && exit 1
fi

all_peaks="$processed_data_dir/peaks.bed.gz"
  
# This script combines the lines in the two peak files,
# retaining only the info that it makes sense to retain,
# since the two files have different columns.
python _merge_uni_bi_peaks.py "$raw_uni_peaks" "$raw_bi_peaks" "$all_peaks"

#echo "Hi Kelly. This is you from the past. You need to implement train/val/test splitting for 5-fold CV. Bye <3"

#exit 0


# Split the set of all peaks into train/val/test, according to chromosome
python _split_peaks_train_val_test.py "$all_peaks"

# the script above should output the files below:
#echo "Examples in the training set:" `zcat "$processed_data_dir/peaks_uni_and_bi_train.bed.gz" | wc -l`
#echo "Examples in the validation set:" `zcat "$processed_data_dir/peaks_uni_and_bi_val.bed.gz" | wc -l`
#echo "Examples in the test set:" `zcat "$processed_data_dir/peaks_uni_and_bi_test.bed.gz" | wc -l`

# Check that the merged peak file has the same # lines as the two input files put together
if [ `zcat "$raw_uni_peaks" "$raw_bi_peaks" | wc -l` -ne `zcat "$all_peaks" | wc -l` ]; then
  echo "Error: merged uni- and bi-directional peaks not the right length." && exit 1
fi


echo "Peaks processed for cell type $cell_type."

exit 0


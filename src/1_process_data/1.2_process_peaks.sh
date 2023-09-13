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

all_peaks="$processed_data_dir/peaks.bed.gz"
  
if [[ "$data_type" == "procap" ]]; then
  echo "Processing as PRO-cap..."

  raw_uni_peaks="$raw_data_dir/peaks.uni.bed.gz"
  raw_bi_peaks="$raw_data_dir/peaks.bi.bed.gz"

  if [ ! -f "$raw_uni_peaks" ] || [ ! -f "$raw_bi_peaks" ]; then
    echo "Missing uni- or bi-directional peak file: $raw_uni_peaks, $raw_bi_peaks" && exit 1
  fi

  # This script combines the lines in the two peak files,
  # retaining only the info that it makes sense to retain,
  # since the two files have different columns.
  python _merge_uni_bi_peaks.py "$raw_uni_peaks" "$raw_bi_peaks" "$all_peaks"

  # Check that the merged peak file has the same # lines as the two input files put together
  if [ `zcat "$raw_uni_peaks" "$raw_bi_peaks" | wc -l` -ne `zcat "$all_peaks" | wc -l` ]; then
    echo "Error: merged uni- and bi-directional peaks not the right length." && exit 1
  fi

elif [[ "$data_type" == "cage" ]]; then
  echo "Processing as CAGE..."

  raw_peaks_rep1="$raw_data_dir/peaks.rep1.bed.gz"
  raw_peaks_rep2="$raw_data_dir/peaks.rep2.bed.gz"
  
  # merge across reps, filter out stuff the model code isn't expecting (genome scaffolds, chrM/EBV, etc)
  zcat "$raw_peaks_rep1" "$raw_peaks_rep2" | sort -k1,1 -k2,2n | bedtools merge -i stdin | grep -e "^chr[0-9XY]*	" | gzip -nc > "$all_peaks"

elif [[ "$data_type" == "rampage" ]]; then
  echo "Processing as RAMPAGE..."  # same as CAGE

  raw_peaks_rep1="$raw_data_dir/peaks.rep1.bed.gz"
  raw_peaks_rep2="$raw_data_dir/peaks.rep2.bed.gz"
  
  # merge across reps, filter out stuff the model code isn't expecting (genome scaffolds, chrM/EBV, etc)
  zcat "$raw_peaks_rep1" "$raw_peaks_rep2" | sort -k1,1 -k2,2n | bedtools merge -i stdin | grep -e "^chr[0-9XY]*	" | gzip -nc > "$all_peaks"

else
  echo "Unrecognized data_type argument." && exit 1
fi

echo "Splitting peaks into train/val/test sets by fold..."
# Split the set of all peaks into train/val/test, according to chromosome
python _split_peaks_train_val_test.py "$all_peaks"

echo "Peaks processed for cell type $cell_type."

exit 0


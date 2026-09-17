#!/bin/bash

set -e

# Code modified from an earlier version of Anusri's ChromBPNet's GC-matching:
# https://github.com/kundajelab/chrombpnet/tree/be0fe06cf0f2dec3147c87cf08e81ec831881fec/chrombpnet/helpers/make_gc_matched_negatives

# This script uses the CpG content values calculated genome-wide previously
# to select a set of genomic windows that are CpG-matched with a given set
# of windows, specified in a bed file.

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_root=$( dirname $( dirname $( dirname $script_dir )))  # extra one because we're in a subdir
echo "Project directory: $proj_root"


data_type="procap"
cell_type="K562"
window_size=2114

genome_fasta="$proj_root/genomes/hg38.withrDNA.fasta"
chrom_sizes="$proj_root/genomes/hg38.withrDNA.chrom.sizes"
data_dir="$proj_root/data/$data_type/processed/$cell_type"



    
# our peak set, which we want to find CpG-content-matched examples for
peaks_bed="$data_dir/peaks.bed.gz"

if [ ! -f "$peaks_bed" ]; then
  echo "Can't find peaks_bed: $peaks_bed" && exit 1
fi

### First, we need to calculate the CpG content genome-wide

# we are assuming stride of 1000 used in script below
genomewide_cpg="$proj_root/annotations/genomewide_cpg_hg38_stride_1000_inputlen_${window_size}.bed.gz"

if [ ! -f "$genomewide_cpg" ]; then
  echo "Generating genome-wide CpG bins (should only do once)..."
  python _get_genomewide_cpg_bins.py -g "$genome_fasta" -o "$genomewide_cpg" -f "$window_size"
  zcat "$genomewide_cpg" | sort -k1,1 -k2,2n | gzip -nc > "$genomewide_cpg.tmp"
  mv "$genomewide_cpg.tmp" "$genomewide_cpg"
fi


### Second, we calculate the CpG content of our peak set

# file where we will store the CpG content of the peaks
peaks_cpg_bed=`echo "$peaks_bed" | sed 's|.bed|_cpg_fracs.bed|'`

python _get_cpg_content_of_peaks.py "$genome_fasta" "$peaks_bed" "$peaks_cpg_bed" "$window_size"

# need to sort for bedtools intersect speed boost
echo "Sorting peaks..."
tmp_sorted_peaks_bed="$data_dir/tmp.peaks.sorted.bed.gz"  # will delete  
zcat "$peaks_bed" | awk -v OFS="\t" '{ print $1, $2, $3 }' | sort -k1,1 -k2,2n | gzip -nc > "$tmp_sorted_peaks_bed"


### Third, we filter out genome-wide bins overlapping any peaks

echo "Filtering candidate sites to not overlap with peaks..."
tmp_candidate_negs_bed="$data_dir/candidate_negatives.bed.gz"  # will delete
bedtools intersect -a "$genomewide_cpg" -b "$tmp_sorted_peaks_bed" -wa -v -sorted | gzip -nc > "$tmp_candidate_negs_bed"

#echo "Contents of $tmp_candidate_negs_bed:"
#head "$tmp_candidate_negs_bed"
#zcat "$tmp_candidate_negs_bed" | head


### Fourth, we select CpG-matched (or as close as possible) examples

echo "Finding regions in candidate examples bed that CpG-match wth foreground..."
out_bed=`echo "$peaks_bed" | sed 's|.bed|_cpg_matched.bed|'`
bw_prefix="$data_dir/5prime"  # needed for getting counts values
python _get_cpg_matched_negatives_with_counts.py -c "$tmp_candidate_negs_bed" --peaks_cpg_bed "$peaks_cpg_bed" -o "$out_bed" --bw "$bw_prefix" -s "$chrom_sizes"


### Finally, split the examples you've gotten back into training and validation (by chromosome)

python ../_split_peaks_train_val_test.py "$out_bed"

#train_out_bed=`echo "$out_bed" | sed 's|.bed|_train.bed|'`
#full_training_set_bed=`echo "$out_bed" | sed 's|.bed|_train_peaks_and_matched.bed|'`
#zcat "$train_out_bed" "$peaks_bed" | shuf | gzip -nc > "$full_training_set_bed"

# delete tmp files
rm "$tmp_sorted_peaks_bed" "$tmp_candidate_negs_bed" "$peaks_cpg_bed"


echo "Done."

exit 0


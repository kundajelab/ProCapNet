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

chrom_sizes="$proj_root/genomes/hg38.withrDNA.chrom.sizes"

# also need genomeCoverageBed in conda environment
source /etc/profile.d/modules.sh
module load samtools/1.15
module load ucsc_tools/latest


data_type="procap"
cell_type=$1

if [[ -z "$cell_type" ]] ; then
  echo "Error: give cell type as argument" && exit 1
fi


echo "Making pseudoreplicates for $cell_type..."

data_dir="$proj_root/data/$data_type/processed/$cell_type"

if [[ ! -d "$data_dir" ]]; then
  echo "Data directory for this cell type not found, exiting: $data_dir" && exit 1
fi


# First, merge the bam files for the replicates and convert to tagAlign format

merged_bam="$data_dir/merged.bam"
merged_sorted_bam="$data_dir/merged.sort.bam"
merged_tagalign="$data_dir/merged.tagAlign.bed.gz"

echo "Merging and sorting bams..."

samtools merge "$merged_bam" $data_dir/rep*bam
samtools sort -n "$merged_bam" -o "$merged_sorted_bam"

echo "Creating tagAlign file..."

bedtools bamtobed -i "$merged_sorted_bam" | awk 'BEGIN{OFS="\t"}{$4="N";$5="1000";print $0}' | grep -v "dm6" | gzip -nc > "$merged_tagalign"

rm "$merged_sorted_bam"


# Second, split the tagAlign file into two pseudoreplicates

pr1_tagalign="$data_dir/pseudorep1.tagAlign.gz"
pr2_tagalign="$data_dir/pseudorep2.tagAlign.gz"

# Get total number of read pairs
nlines=$( zcat "$merged_tagalign" | wc -l )
nlines=$(( (nlines + 1) / 2 ))

# Shuffle and split BED file into 2 equal parts
out_pref="$data_dir/pr"

echo "Splitting tagAlign between pseudoreps..."

# Will produce $out_pref00 and $out_pref01
zcat "$merged_tagalign" | shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$(zcat -f "$merged_tagalign" | wc -c) -nosalt </dev/zero 2>/dev/null) | split -d -l ${nlines} - "$out_pref"

# Convert reads into standard tagAlign file
gzip -nc "${out_pref}00" > "$pr1_tagalign"
rm "${out_pref}00"
gzip -nc "${out_pref}01" > "$pr2_tagalign"
rm "${out_pref}01"


# Third, convert the pseudorep tagAligns into bigWigs

echo "Creating bigWigs from tagAlign pseudoreps..."

pseudoreps=( "$pr1_tagalign" "$pr2_tagalign" )

for pseudorep in "${pseudoreps[@]}"; do
  strands=( "pos" "neg" )
  for strand in ${strands[@]}; do
    if [ "$strand" = "pos" ]; then
      strand_symbol="+"
    else
      strand_symbol="-"
    fi

    pr_pref=`echo "$pseudorep" | sed 's/.tagAlign.gz//'`
    zcat -f "$pseudorep" | LC_COLLATE=C sort -k1,1 -k2,2n | bedtools genomecov -5 -bg -strand "$strand_symbol" -g "$chrom_sizes" -i stdin > "$data_dir/tmp.bed"
    bedGraphToBigWig "$data_dir/tmp.bed" "$chrom_sizes" "${pr_pref}.${strand}.bigWig"
  done

  rm "$pseudorep" "$data_dir/tmp.bed"
done



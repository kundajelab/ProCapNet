#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_root=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_root"


# also need genomeCoverageBed in conda environment
source /etc/profile.d/modules.sh
module load samtools/1.15
module load ucsc_tools/latest


data_type="procap"
cell_type=$1

if [[ -z "$cell_type" ]] ; then
  echo "Error: give cell type as argument" && exit 1
fi


echo "Processing bams for $cell_type..."

raw_data_dir="$proj_root/data/$data_type/raw/$cell_type"
processed_data_dir="$proj_root/data/$data_type/processed/$cell_type"

if [[ ! -d "$raw_data_dir" ]]; then
  echo "Raw data directory for this cell type not found, exiting: $raw_data_dir" && exit 1
fi

# we'll save all the new files we create in this directory
mkdir -p "$processed_data_dir"

# needed for bigwig-writing code
chrom_sizes="$proj_root/genomes/hg38.withrDNA.chrom.sizes"

# whatever is between "rep#" and ".bam" in the naming scheme of the bam files
raw_file_mid="raw."




#### Loop over all replicates, one bam file each
# These files were downloaded from ENCODE (see download_data.sh)

for raw_bam in $raw_data_dir/*.${raw_file_mid}bam; do
  tmp_bam="$raw_bam.tmp"  # will delete
    
  processed_bam_basename=`basename "$raw_bam" | sed "s/${raw_file_mid}//"`
  processed_bam="$processed_data_dir/$processed_bam_basename"  # will be final processed bam file
    
  # Remove R1, leaving just R2 (where the TSS actually is for PE PRO-cap)
  echo "Filtering R1 from $raw_bam (assuming paired-end data!)..."
  samtools view -hb -f 128 "$raw_bam" -o "$tmp_bam"
    
  # Sort bam file (needed for ucsc tools to make bigwigs)
  echo "Sorting $raw_bam..."
  samtools sort "$tmp_bam" -o "$processed_bam"

  strands=( "pos" "neg" )
  for strand in ${strands[@]}; do
    if [ "$strand" = "pos" ]; then
      strand_symbol="+"
    else
      strand_symbol="-"
    fi

    tmp_bg="$processed_bam.bg"  # will delete

    rep_bigwig=`echo "$processed_bam" | sed "s|.bam|.5prime.${strand}.bigWig|"`

    echo "Converting replicate bam to bedgraph (5' ends only)..."
    genomeCoverageBed -ibam "$processed_bam" -bg -strand "$strand_symbol" -5 | grep -v "_" | LC_COLLATE=C sort -k1,1 -k2,2n > "$tmp_bg"
    echo "Converting bedgraph to bigWig (5' ends only)..."
    bedGraphToBigWig "$tmp_bg" "$chrom_sizes" "$rep_bigwig"

    rm "$tmp_bg"

  done  # end of strands loop
    
  rm "$tmp_bam"
    
done  # end of replicates loop


  
### Merge bigwigs of replicates


# if you want to do 3' ends also, add that to list
read_types=( "5prime" )

for read_type in "${read_types[@]}"; do
  strands=( "pos" "neg" )
  for strand in ${strands[@]}; do
    final_bigwig="$processed_data_dir/${read_type}.${strand}.bigWig"
      
    # need to delete past final_bigwig, or else it will be
    # among the results of the find command below (shows as 2x data in bigwig)
    # (note this error should not longer happen after final_bigwig name change,
    # but will keep this in anyways)
    if [ -f "$final_bigwig" ]; then
      rm "$final_bigwig"
    fi
      
    rep_bigwigs=`find "$processed_data_dir/" -mindepth 1 -maxdepth 1 -name "*.${read_type}.${strand}.bigWig"`
    echo "Replicate bigwigs found: $rep_bigwigs"
      
    if [ `echo "$rep_bigwigs" | wc -l` -eq "1" ]; then
      echo "One replicate found."
      cp "$rep1_bigwig" "$final_bigwig"
    else
      echo "Multiple replicates being merged..."
      rep_bigwigs_list=`echo "$rep_bigwigs" | tr '\n' ' '`
      tmp_bg="$processed_data_dir/tmp.bg"  # will delete
      tmp2_bg="$processed_data_dir/tmp2.bg"  # will delete

      bigWigMerge $rep_bigwigs_list "$tmp_bg"
      LC_COLLATE=C sort -k1,1 -k2,2n "$tmp_bg" > "$tmp2_bg"
      bedGraphToBigWig "$tmp2_bg" "$chrom_sizes" "$final_bigwig"

      rm "$tmp_bg" "$tmp2_bg"
    fi
  done  # end of strands loop
done  # end of read type loop


echo "Done processing bams for $cell_type."

exit 0


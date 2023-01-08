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

proj_dir=`get_proj_root`

echo "Project directory: $proj_dir"

genomes_dir="$proj_dir/genomes"


# Download the ENCODE version of the hg38 genome fasta

hg38_fasta="$genomes_dir/hg38.fasta"

#wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz -O "${hg38_fasta}.gz"
#gunzip "${hg38_fasta}.gz"


# Download the U13369.1 reference sequence for ribosomal DNA
# (the procap data was aligned using this in the reference, so
# we need to include it to process the data made that way)

rDNA_fasta="$genomes_dir/rDNA_human_U13369.1.fasta"

#wget https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi\?tool\=portal\&save\=file\&log$\=seqview\&db\=nuccore\&report\=fasta\&id\=555853\&conwithfeat\=on\&withparts\=on\&hide-cdd\=on -O - | sed '$ d' > "$rDNA_fasta"


# Then, combine the two fasta files

combo_fasta="$genomes_dir/hg38.withrDNA.fasta"
#cat "$hg38_fasta" "$rDNA_fasta" > "$combo_fasta"


# Download the chromosome sizes file

wget https://www.encodeproject.org/files/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv -O - | grep -v "_alt" > "$genomes_dir/hg38.chrom.sizes"

# Create a chromosome sizes file for just the rDNA
cp "$genomes_dir/hg38.chrom.sizes" "$genomes_dir/hg38.withrDNA.chrom.sizes"
rDNA_length=`grep -v ">" "$rDNA_fasta" | tr -d '\n' | wc -c`
echo "U13369.1	$rDNA_length" >> "$genomes_dir/hg38.withrDNA.chrom.sizes"


echo "Done downloading genome files."

exit 0


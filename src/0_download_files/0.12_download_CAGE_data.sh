#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_dir=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_dir"

echo "Downloading data from ENCODE portal..."

# Where we will download all data files to
raw_data_dir="$proj_dir/data/cage/raw"
mkdir -p "$raw_data_dir"

### K562 CAGE

dest_dir="$raw_data_dir/K562"
mkdir -p "$dest_dir"

# bams (filtered)
#wget https://www.encodeproject.org/files/ENCFF754FAU/@@download/ENCFF754FAU.bam -O "$dest_dir/rep1.raw.bam"
#wget https://www.encodeproject.org/files/ENCFF366MWI/@@download/ENCFF366MWI.bam -O "$dest_dir/rep2.raw.bam"

# peak calls (per-replicate, not IDR; will use overlap between replicates)
#wget https://www.encodeproject.org/files/ENCFF698DQS/@@download/ENCFF698DQS.bed.gz -O "$dest_dir/peaks.bed.gz"
wget https://www.encodeproject.org/files/ENCFF638ZUQ/@@download/ENCFF638ZUQ.bed.gz -O "$dest_dir/peaks.rep1.bed.gz"
wget https://www.encodeproject.org/files/ENCFF370YBR/@@download/ENCFF370YBR.bed.gz -O "$dest_dir/peaks.rep2.bed.gz"


### K562 RAMPAGE

raw_data_dir="$proj_dir/data/rampage/raw"
mkdir -p "$raw_data_dir"

dest_dir="$raw_data_dir/K562"
mkdir -p "$dest_dir"

# bams
#wget https://www.encodeproject.org/files/ENCFF038VNX/@@download/ENCFF038VNX.bam -O "$dest_dir/rep1.raw.bam"
#wget https://www.encodeproject.org/files/ENCFF618CKG/@@download/ENCFF618CKG.bam -O "$dest_dir/rep2.raw.bam"

# peak calls (per-replicate, not IDR; will use overlap between replicates)
wget https://www.encodeproject.org/files/ENCFF923WZG/@@download/ENCFF923WZG.bed.gz -O "$dest_dir/peaks.rep1.bed.gz"
wget https://www.encodeproject.org/files/ENCFF954ZBU/@@download/ENCFF954ZBU.bed.gz -O "$dest_dir/peaks.rep2.bed.gz"



echo "Done downloading data."
exit 0

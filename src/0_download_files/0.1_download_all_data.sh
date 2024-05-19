#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_dir=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_dir"

echo "Downloading data and annotations from ENCODE portal and such..."

# Where we will download all data files to
raw_data_dir="$proj_dir/data/procap/raw"
annots_dir="$proj_dir/annotations"
mkdir -p "$raw_data_dir"
mkdir -p "$annots_dir"


### K562 PRO-cap experiment (ENCSR261KBX)

dest_dir="$raw_data_dir/K562"
mkdir -p "$dest_dir"

# bams (filtered)
wget https://www.encodeproject.org/files/ENCFF877SQU/@@download/ENCFF877SQU.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF663UAN/@@download/ENCFF663UAN.bam -O "$dest_dir/rep2.raw.bam"

# peak calls
wget https://www.encodeproject.org/files/ENCFF819CPA/@@download/ENCFF819CPA.bed.gz -O "$dest_dir/peaks.bi.bed.gz"
wget https://www.encodeproject.org/files/ENCFF636REV/@@download/ENCFF636REV.bed.gz -O "$dest_dir/peaks.uni.bed.gz"

# annotations
dest_dir="$annots_dir/K562"
mkdir -p "$dest_dir"
wget https://www.encodeproject.org/files/ENCFF464BRU/@@download/ENCFF464BRU.bed.gz -O "$dest_dir/cCREs.bed.gz"
wget https://www.encodeproject.org/files/ENCFF274YGF/@@download/ENCFF274YGF.bed.gz -O "$dest_dir/DNase_peaks.bed.gz"

### More cell types!

dest_dir="$raw_data_dir/MCF10A"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF010CAI/@@download/ENCFF010CAI.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF966SIN/@@download/ENCFF966SIN.bam -O "$dest_dir/rep2.raw.bam"
wget https://www.encodeproject.org/files/ENCFF463RFT/@@download/ENCFF463RFT.bam -O "$dest_dir/rep3.raw.bam"
wget https://www.encodeproject.org/files/ENCFF123ORK/@@download/ENCFF123ORK.bam -O "$dest_dir/rep4.raw.bam"
wget https://www.encodeproject.org/files/ENCFF478BHR/@@download/ENCFF478BHR.bed.gz -O "$dest_dir/peaks.bi.bed.gz"
wget https://www.encodeproject.org/files/ENCFF875PER/@@download/ENCFF875PER.bed.gz -O "$dest_dir/peaks.uni.bed.gz"

# annotations
dest_dir="$annots_dir/MCF10A"
mkdir -p "$dest_dir"
wget https://www.encodeproject.org/files/ENCFF916CRB/@@download/ENCFF916CRB.bed.gz -O "$dest_dir/DNase_peaks.bed.gz"


dest_dir="$raw_data_dir/A673"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF121DPX/@@download/ENCFF121DPX.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF349OYZ/@@download/ENCFF349OYZ.bam -O "$dest_dir/rep2.raw.bam"
wget https://www.encodeproject.org/files/ENCFF728PUD/@@download/ENCFF728PUD.bam -O "$dest_dir/rep3.raw.bam"
wget https://www.encodeproject.org/files/ENCFF395TST/@@download/ENCFF395TST.bam -O "$dest_dir/rep4.raw.bam"
wget https://www.encodeproject.org/files/ENCFF268UEL/@@download/ENCFF268UEL.bed.gz -O "$dest_dir/peaks.bi.bed.gz"
wget https://www.encodeproject.org/files/ENCFF045MXI/@@download/ENCFF045MXI.bed.gz -O "$dest_dir/peaks.uni.bed.gz"

# annotations
dest_dir="$annots_dir/A673"
mkdir -p "$dest_dir"
wget https://www.encodeproject.org/files/ENCFF629STI/@@download/ENCFF629STI.bed.gz -O "$dest_dir/DNase_peaks.bed.gz"


dest_dir="$raw_data_dir/CACO2"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF477BQK/@@download/ENCFF477BQK.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF725CZI/@@download/ENCFF725CZI.bam -O "$dest_dir/rep2.raw.bam"
wget https://www.encodeproject.org/files/ENCFF396MZI/@@download/ENCFF396MZI.bed.gz -O "$dest_dir/peaks.bi.bed.gz"
wget https://www.encodeproject.org/files/ENCFF814YZW/@@download/ENCFF814YZW.bed.gz -O "$dest_dir/peaks.uni.bed.gz"

# annotations
dest_dir="$annots_dir/CACO2"
mkdir -p "$dest_dir"
wget https://www.encodeproject.org/files/ENCFF579UXQ/@@download/ENCFF579UXQ.bed.gz -O "$dest_dir/DNase_peaks.bed.gz"


dest_dir="$raw_data_dir/CALU3"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF630HAF/@@download/ENCFF630HAF.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF226MDQ/@@download/ENCFF226MDQ.bam -O "$dest_dir/rep2.raw.bam"
wget https://www.encodeproject.org/files/ENCFF982XOO/@@download/ENCFF982XOO.bed.gz -O "$dest_dir/peaks.bi.bed.gz"
wget https://www.encodeproject.org/files/ENCFF819VDW/@@download/ENCFF819VDW.bed.gz -O "$dest_dir/peaks.uni.bed.gz"

# annotations
dest_dir="$annots_dir/CALU3"
mkdir -p "$dest_dir"
wget https://www.encodeproject.org/files/ENCFF930GWQ/@@download/ENCFF930GWQ.bed.gz -O "$dest_dir/DNase_peaks.bed.gz"


dest_dir="$raw_data_dir/HUVEC"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF508GKB/@@download/ENCFF508GKB.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF394GCC/@@download/ENCFF394GCC.bam -O "$dest_dir/rep2.raw.bam"
wget https://www.encodeproject.org/files/ENCFF122NQB/@@download/ENCFF122NQB.bed.gz -O "$dest_dir/peaks.bi.bed.gz"
wget https://www.encodeproject.org/files/ENCFF931HXJ/@@download/ENCFF931HXJ.bed.gz -O "$dest_dir/peaks.uni.bed.gz"

# annotations
dest_dir="$annots_dir/HUVEC"
mkdir -p "$dest_dir"
wget https://www.encodeproject.org/files/ENCFF406AWN/@@download/ENCFF406AWN.bed.gz -O "$dest_dir/DNase_peaks.bed.gz"


### Other Annoations

wget https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_non-redundant_pfms_meme.txt -O "$annots_dir/JASPAR2022_CORE_pfms.meme"

wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz -O "$annots_dir/gencode.v41.annotation.gtf.gz"

wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw -O "$annots_dir/hg38.phyloP100way.bw"


### CAGE and RAMPAGE

### K562 CAGE

raw_data_dir="$proj_dir/data/cage/raw"
#mkdir -p "$raw_data_dir"

dest_dir="$raw_data_dir/K562"
#mkdir -p "$dest_dir"

# bams (filtered?)
#wget https://www.encodeproject.org/files/ENCFF754FAU/@@download/ENCFF754FAU.bam -O "$dest_dir/rep1.raw.bam"
#wget https://www.encodeproject.org/files/ENCFF366MWI/@@download/ENCFF366MWI.bam -O "$dest_dir/rep2.raw.bam"

# peak calls (per-replicate, not IDR; will use overlap between replicates)
#wget https://www.encodeproject.org/files/ENCFF638ZUQ/@@download/ENCFF638ZUQ.bed.gz -O "$dest_dir/peaks.rep1.bed.gz"
#wget https://www.encodeproject.org/files/ENCFF370YBR/@@download/ENCFF370YBR.bed.gz -O "$dest_dir/peaks.rep2.bed.gz"


### K562 RAMPAGE

raw_data_dir="$proj_dir/data/rampage/raw"
mkdir -p "$raw_data_dir"

dest_dir="$raw_data_dir/K562"
mkdir -p "$dest_dir"

# bams
wget https://www.encodeproject.org/files/ENCFF038VNX/@@download/ENCFF038VNX.bam -O "$dest_dir/rep1.raw.bam"
wget https://www.encodeproject.org/files/ENCFF618CKG/@@download/ENCFF618CKG.bam -O "$dest_dir/rep2.raw.bam"

# peak calls (per-replicate, not IDR; will use overlap between replicates)
wget https://www.encodeproject.org/files/ENCFF923WZG/@@download/ENCFF923WZG.bed.gz -O "$dest_dir/peaks.rep1.bed.gz"
wget https://www.encodeproject.org/files/ENCFF954ZBU/@@download/ENCFF954ZBU.bed.gz -O "$dest_dir/peaks.rep2.bed.gz"


echo "Done downloading data."
exit 0

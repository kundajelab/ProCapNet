#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_dir=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_dir"

echo "Downloading K562 histone mark data..."

cell="K562"

# Where we will download all K562 histone mark data to
# (ok I know it's not an annotation but I didn't want to make another folder just for these)
dest_dir="$proj_dir/annotations/$cell"
mkdir -p "$dest_dir"

# ENCSR668LDD
wget https://www.encodeproject.org/files/ENCFF253TOF/@@download/ENCFF253TOF.bigWig -O "$dest_dir/H3K4me3.bigWig"
# ENCSR000AKS
wget https://www.encodeproject.org/files/ENCFF834SEY/@@download/ENCFF834SEY.bigWig -O "$dest_dir/H3K4me1.bigWig"
# ENCSR000AKT
wget https://www.encodeproject.org/files/ENCFF959YJV/@@download/ENCFF959YJV.bigWig -O "$dest_dir/H3K4me2.bigWig"
# ENCSR000AKP
wget https://www.encodeproject.org/files/ENCFF381NDD/@@download/ENCFF381NDD.bigWig -O "$dest_dir/H3K27ac.bigWig"
# ENCSR000APD
wget https://www.encodeproject.org/files/ENCFF544AVW/@@download/ENCFF544AVW.bigWig -O "$dest_dir/H3K79me2.bigWig"
# ENCSR000AKV
wget https://www.encodeproject.org/files/ENCFF286WRJ/@@download/ENCFF286WRJ.bigWig -O "$dest_dir/H3K9ac.bigWig"

# also ATAC-seq which I know is not a histone mark but it'll be ok I promise
wget https://www.encodeproject.org/files/ENCFF102ARJ/@@download/ENCFF102ARJ.bigWig -O "$dest_dir/ATAC.bigWig"

echo "Done downloading histone mark data."

exit 0


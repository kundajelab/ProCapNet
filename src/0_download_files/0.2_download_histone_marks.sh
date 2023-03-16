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

# Where we will download all data files to
data_dir="$proj_dir/src/counts_residuals_analysis"
mkdir -p "$data_dir"

### K562 histone marks

dest_dir="$data_dir/K562"
mkdir -p "$dest_dir"

# ENCSR668LDD
#wget https://www.encodeproject.org/files/ENCFF253TOF/@@download/ENCFF253TOF.bigWig -O "$dest_dir/H3K4me3.bigWig"
# ENCSR000AKS
#wget https://www.encodeproject.org/files/ENCFF834SEY/@@download/ENCFF834SEY.bigWig -O "$dest_dir/H3K4me1.bigWig"
# ENCSR000AKT
#wget https://www.encodeproject.org/files/ENCFF959YJV/@@download/ENCFF959YJV.bigWig -O "$dest_dir/H3K4me2.bigWig"
# ENCSR000AKP
#wget https://www.encodeproject.org/files/ENCFF381NDD/@@download/ENCFF381NDD.bigWig -O "$dest_dir/H3K27ac.bigWig"
# ENCSR000APD
wget https://www.encodeproject.org/files/ENCFF544AVW/@@download/ENCFF544AVW.bigWig -O "$dest_dir/H3K79me2.bigWig"
# ENCSR000AKV
#wget https://www.encodeproject.org/files/ENCFF286WRJ/@@download/ENCFF286WRJ.bigWig -O "$dest_dir/H3K9ac.bigWig"

# ATAC-seq
#wget https://www.encodeproject.org/files/ENCFF102ARJ/@@download/ENCFF102ARJ.bigWig -O "$dest_dir/ATAC.bigWig"


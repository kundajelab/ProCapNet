#!/bin/bash

set -e

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
proj_dir=$( dirname $( dirname $script_dir ))
echo "Project directory: $proj_dir"

echo "Downloading histone mark data..."

cell="K562"

# Where we will download all K562 histone mark data to
# (ok I know it's not an annotation but I didn't want to make another folder just for these)
dest_dir="$proj_dir/annotations/$cell"
mkdir -p "$dest_dir"

# all of these should be fold-change over control bigwigs, not p-values

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
# ENCSR000APE
wget https://www.encodeproject.org/files/ENCFF601JGK/@@download/ENCFF601JGK.bigWig -O "$dest_dir/H3K9me3.bigWig"
# ENCSR000AKQ
wget https://www.encodeproject.org/files/ENCFF405HIO/@@download/ENCFF405HIO.bigWig -O "$dest_dir/H3K27me3.bigWig"
# ENCSR000AKW
wget https://www.encodeproject.org/files/ENCFF654SLZ/@@download/ENCFF654SLZ.bigWig -O "$dest_dir/H3K9me1.bigWig"
# ENCSR000AKR
wget https://www.encodeproject.org/files/ENCFF317VHO/@@download/ENCFF317VHO.bigWig -O "$dest_dir/H3K36me3.bigWig"


# also ATAC-seq which I know is not a histone mark but it'll be ok I promise
wget https://www.encodeproject.org/files/ENCFF102ARJ/@@download/ENCFF102ARJ.bigWig -O "$dest_dir/ATAC.bigWig"

# also DNase (this one is "read-depth-normalized signal" rather than fold-change?)
wget https://www.encodeproject.org/files/ENCFF972GVB/@@download/ENCFF972GVB.bigWig -O "$dest_dir/DNase.bigWig"


cell="A673"
dest_dir="$proj_dir/annotations/$cell"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF061CVO/@@download/ENCFF061CVO.bigWig -O "$dest_dir/H3K27me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF820QRK/@@download/ENCFF820QRK.bigWig -O "$dest_dir/H3K4me1.bigWig"
wget https://www.encodeproject.org/files/ENCFF213BSP/@@download/ENCFF213BSP.bigWig -O "$dest_dir/H3K27ac.bigWig"
wget https://www.encodeproject.org/files/ENCFF163YOI/@@download/ENCFF163YOI.bigWig -O "$dest_dir/H3K9me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF014WRA/@@download/ENCFF014WRA.bigWig -O "$dest_dir/H3K36me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF958CFK/@@download/ENCFF958CFK.bigWig -O "$dest_dir/H3K4me3.bigWig"

wget https://www.encodeproject.org/files/ENCFF107RAU/@@download/ENCFF107RAU.bigWig -O "$dest_dir/DNase.bigWig"


cell="CACO2"
dest_dir="$proj_dir/annotations/$cell"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF120MMZ/@@download/ENCFF120MMZ.bigWig -O "$dest_dir/H3K27me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF135XXE/@@download/ENCFF135XXE.bigWig -O "$dest_dir/H3K4me1.bigWig"
wget https://www.encodeproject.org/files/ENCFF904AUC/@@download/ENCFF904AUC.bigWig -O "$dest_dir/H3K9me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF920MFN/@@download/ENCFF920MFN.bigWig -O "$dest_dir/H3K36me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF027JAD/@@download/ENCFF027JAD.bigWig -O "$dest_dir/H3K4me3.bigWig"

wget https://www.encodeproject.org/files/ENCFF695QYJ/@@download/ENCFF695QYJ.bigWig -O "$dest_dir/DNase.bigWig"


cell="HUVEC"
dest_dir="$proj_dir/annotations/$cell"
mkdir -p "$dest_dir"

wget https://www.encodeproject.org/files/ENCFF778MSD/@@download/ENCFF778MSD.bigWig -O "$dest_dir/H3K27me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF254KUQ/@@download/ENCFF254KUQ.bigWig -O "$dest_dir/H3K4me1.bigWig"
wget https://www.encodeproject.org/files/ENCFF955PAU/@@download/ENCFF955PAU.bigWig -O "$dest_dir/H3K27ac.bigWig"
wget https://www.encodeproject.org/files/ENCFF846FWZ/@@download/ENCFF846FWZ.bigWig -O "$dest_dir/H3K9me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF983DWS/@@download/ENCFF983DWS.bigWig -O "$dest_dir/H3K36me3.bigWig"
wget https://www.encodeproject.org/files/ENCFF399KTR/@@download/ENCFF399KTR.bigWig -O "$dest_dir/H3K4me3.bigWig"

wget https://www.encodeproject.org/files/ENCFF289OKT/@@download/ENCFF289OKT.bigWig -O "$dest_dir/DNase.bigWig"


echo "Done downloading histone mark data."

exit 0


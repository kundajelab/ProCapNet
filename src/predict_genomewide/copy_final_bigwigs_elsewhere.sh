#!/bin/bash

set -e

proj_root="/users/kcochran/projects/procapnet"
mitra_root="/srv/www/kundaje/kcochran/nascent_RNA"

cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )
cell_types=( "K562" )

strands=( "pos" "neg" )


genome="hg38"


whereto="mitra"


if [[ "$whereto" == "mitra" ]]; then
    mkdir -p "${mitra_root}/gencode_genomewide/$genome/predictions/"
    
    for cell_type in "${cell_types[@]}"; do
        echo "$cell_type"

        for strand in "${strands[@]}"; do
            from_path="${proj_root}/src/predict_genomewide/bigwigs/$genome/${cell_type}/genomewide/${cell_type}.${strand}.bigWig"
            to_path="${mitra_root}/gencode_genomewide/$genome/predictions/${cell_type}.${strand}.bigWig"

            echo 'cp' "$from_path" "$to_path"
            cp "$from_path" "$to_path"
        done
    done
else
    mkdir -p "${proj_root}/wg_preds/$genome/bigwigs/"
    
    for cell_type in "${cell_types[@]}"; do
        echo "$cell_type"

        for strand in "${strands[@]}"; do
            from_path="${proj_root}/src/predict_genomewide/bigwigs/$genome/${cell_type}/genomewide/${cell_type}.${strand}.bigWig"
            to_path="${proj_root}/wg_preds/$genome/bigwigs/${cell_type}.${strand}.bigWig"

            echo 'cp' "$from_path" "$to_path"
            cp "$from_path" "$to_path"
        done
    done
fi

echo "Done copying bigwigs."

exit 0

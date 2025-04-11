#!/bin/bash

set -e

cell_type=$1
chrom=$2

genome="hg38"

raw_preds_dir="raw_preds/${genome}/${cell_type}/${chrom}"

fold_dirs=`ls -d $raw_preds_dir/*/ | grep -v "merged"`
echo $fold_dirs

for fold_dir in `ls -d $raw_preds_dir/*/ | grep -v "merged"`; do
  echo "${fold_dir}"
  rm -r "$fold_dir"
done

exit 0

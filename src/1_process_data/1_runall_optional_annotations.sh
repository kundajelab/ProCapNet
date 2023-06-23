#!/bin/bash

# the outputs from these scripts are only used in a few figures in the jupyter notebooks
# so if you just want trained models / predictions / contribution scores, don't bother

python make_gene_region_annotations.py
python make_tct_promoter_annotations.py

# this requires a file we got from Jesse - so will not work when other people try to run this
python make_housekeeping_promoter_annotations.py

./make_peak_union_across_cell_types.sh

exit 0

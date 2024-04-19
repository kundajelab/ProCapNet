#!/bin/bash

set -e

script_dir=`dirname $0`

# the outputs from these scripts are only used in a few figures in the jupyter notebooks
# so if you just want trained models / predictions / contribution scores, don't bother

python "$script_dir/make_gene_region_annotations.py"
python "$script_dir/make_tct_promoter_annotations.py"

# this requires a file we got from Jesse - so will not work when other people try to run this... #TODO
python "$script_dir/make_housekeeping_promoter_annotations.py"

"$script_dir/make_peak_union_across_cell_types.sh"

# for the promoters-only ProCapNet trained for the promtoers vs. enhancers analysis
"$script_dir//make_promoters_only_peak_files.sh"

exit 0

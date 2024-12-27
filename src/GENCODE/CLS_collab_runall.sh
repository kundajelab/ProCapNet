#!/bin/bash

set -e


### Note: you probably should not actually run this script,
# since that would take forever. But you can use it to see
# how to run each of the Python scripts in this folder.

### Before running this script, you need to have run the notebook CLS_collab_make_regions.ipynb



# needed for generating ProCapNet predictions
GPU="1"

# all the ProCapNet models we have
cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )

mkdir -p logs

for cell_type in "${cell_types[@]}"; do
  python CLS_collab_make_predictions_script.py "$cell_type" "$GPU" | tee "logs/CLS_collab_pred_${cell_type}.log"
  python CLS_collab_score_TSSs_from_predictions_script.py "$cell_type" | tee "logs/CLS_collab_score_${cell_type}.log"
done

python CLS_collab_merge_scores_across_cell_types_script.py | tee "logs/CLS_collab_merge_scores.log"
python CLS_collab_scores_to_labels.py | tee "logs/CLS_collab_scores_to_labels.log"


echo "Done!"

exit 0

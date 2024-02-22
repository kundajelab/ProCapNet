#!/bin/bash

set -e
  
cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )
  
for cell_type in "${cell_types[@]}"; do
  echo "$cell_type"

  if [ "$cell_type" == "K562" ]; then
    enc_id="ENCSR261KBX"
    timestamps=( "2023-05-29_15-51-40" "2023-05-29_15-58-41" "2023-05-29_15-59-09" "2023-05-30_01-40-06" "2023-05-29_23-21-23" "2023-05-29_23-23-45" "2023-05-29_23-24-11" )
  fi
  if [ "$cell_type" == "A673" ]; then
    enc_id="ENCSR046BCI"
    timestamps=( "2023-06-11_20-11-32" "2023-06-11_23-42-00" "2023-06-12_03-29-06" "2023-06-12_07-17-43" "2023-06-12_11-10-59" "2023-06-12_14-36-40" "2023-06-12_17-26-09" )
  fi
  if [ "$cell_type" == "CACO2" ]; then
    enc_id="ENCSR100LIJ"
    timestamps=( "2023-06-12_21-46-40" "2023-06-13_01-28-24" "2023-06-13_05-06-53" "2023-06-13_08-52-39" "2023-06-13_13-12-09" "2023-06-13_16-40-41" "2023-06-13_20-08-39" )
  fi
  if [ "$cell_type" == "CALU3" ]; then
    enc_id="ENCSR935RNW"
    timestamps=( "2023-06-14_00-43-44" "2023-06-14_04-26-48" "2023-06-14_09-34-26" "2023-06-14_13-03-59" "2023-06-14_17-22-28" "2023-06-14_21-03-11" "2023-06-14_23-58-36" )
  fi
  if [ "$cell_type" == "HUVEC" ]; then
    enc_id="ENCSR098LLB"
    timestamps=( "2023-06-16_21-59-35" "2023-06-17_00-20-34" "2023-06-17_02-17-07" "2023-06-17_04-27-08" "2023-06-17_06-42-19" "2023-06-17_09-16-24" "2023-06-17_11-09-38" )
  fi
  if [ "$cell_type" == "MCF10A" ]; then
    enc_id="ENCSR799DGV"
    timestamps=( "2023-06-15_06-07-40" "2023-06-15_10-37-03" "2023-06-15_16-23-56" "2023-06-15_21-44-32" "2023-06-16_03-47-46" "2023-06-16_09-41-26" "2023-06-16_15-07-01" )
  fi
  
  
  if [ -z "$enc_id" ]; then
    echo "Input cell type argument (and spell it correctly)." && exit 1
  fi
  
  
  
  data_dir="/users/kcochran/projects/procapnet/data/procap/processed/$cell_type"
  
  models_dir="/users/kcochran/projects/procapnet/models/procap/$cell_type/strand_merged_umap"
  
  train_logs_dir="/users/kcochran/projects/procapnet/src/2_train_models/logs"
  
  preds_dir="/users/kcochran/projects/procapnet/model_out/procap/$cell_type/strand_merged_umap/merged"
  
  deepshap_dir="/users/kcochran/projects/procapnet/deepshap_out/procap/$cell_type/strand_merged_umap/merged"
  
  dest_dir="/users/kcochran/oak/kcochran/procapnet_portal_uploads/to_upload/$enc_id"
  
  if [ -d "$dest_dir" ]; then
    rm -r "$dest_dir"
  fi
  
  mkdir -p "$dest_dir"
  
  cp "$data_dir/5prime.pos.bigWig" "$dest_dir/${enc_id}.reads.5primeends.plus.bigWig"
  cp "$data_dir/5prime.neg.bigWig" "$dest_dir/${enc_id}.reads.5primeends.minus.bigWig"
  
  cp "$data_dir/peaks.bed.gz" "$dest_dir/${enc_id}.peaks.all.bed.gz"
  cp "$data_dir/dnase_peaks_no_procap_overlap.bed.gz" "$dest_dir/${enc_id}.nonpeaks.all.bed.gz"
  
  
  for fold_i in $(seq 0 6); do
    # 0 indexed vs 1 indexed
    let fold_j=$((fold_i + 1))
    
    cp "$data_dir/peaks_fold${fold_j}_train.bed.gz" "$dest_dir/${enc_id}.peaks.fold${fold_i}.trainingset.bed.gz"
    cp "$data_dir/peaks_fold${fold_j}_val.bed.gz" "$dest_dir/${enc_id}.peaks.fold${fold_i}.validationset.bed.gz"
    cp "$data_dir/peaks_fold${fold_j}_test.bed.gz" "$dest_dir/${enc_id}.peaks.fold${fold_i}.testset.bed.gz"
  
    cp "$data_dir/dnase_peaks_no_procap_overlap_fold${fold_j}_train.bed.gz" "$dest_dir/${enc_id}.nonpeaks.fold${fold_i}.trainingset.bed.gz"
    cp "$data_dir/dnase_peaks_no_procap_overlap_fold${fold_j}_val.bed.gz" "$dest_dir/${enc_id}.nonpeaks.fold${fold_i}.validationset.bed.gz"
    cp "$data_dir/dnase_peaks_no_procap_overlap_fold${fold_j}_test.bed.gz" "$dest_dir/${enc_id}.nonpeaks.fold${fold_i}.testset.bed.gz"
  
  
    model_id="${timestamps[$fold_i]}"
  
    python resave_pytorch_model.py "$models_dir/${model_id}.model" "$dest_dir/${enc_id}.procapnet_model.fold${fold_i}.state_dict.torch"
  
    cp "$train_logs_dir/${cell_type}_${fold_j}.log" "$dest_dir/${enc_id}.logfile.procapnet_model_training.fold${fold_i}.txt"
  
  done # folds loop
  
  cp "$preds_dir/all_pred_profiles.pos.bigWig" "$dest_dir/${enc_id}.predictions.mergedallfolds.plus.bigWig"
  cp "$preds_dir/all_pred_profiles.neg.bigWig" "$dest_dir/${enc_id}.predictions.mergedallfolds.minus.bigWig"
  
  cp "$deepshap_dir/all_profile_deepshap.bigWig" "$dest_dir/${enc_id}.deepshap_contrib_scores.profile.mergedallfolds.bigWig"
  cp "$deepshap_dir/all_counts_deepshap.bigWig" "$dest_dir/${enc_id}.deepshap_contrib_scores.counts.mergedallfolds.bigWig"
  
done
  
echo "Done."


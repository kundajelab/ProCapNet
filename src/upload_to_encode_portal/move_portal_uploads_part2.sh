#!/bin/bash

set -e

source /etc/profile.d/modules.sh
module load ucsc_tools/latest

chrom_sizes="/users/kcochran/projects/procapnet/genomes/hg38.chrom.sizes"

tasks=( "profile" "counts" )
  
#cell_types=( "K562" "A673" "CACO2" "CALU3" "HUVEC" "MCF10A" )
cell_types=( "CACO2" )

for cell_type in "${cell_types[@]}"; do
  echo "$cell_type"

  if [ "$cell_type" == "K562" ]; then
    enc_id="ENCSR261KBX"
  fi
  if [ "$cell_type" == "A673" ]; then
    enc_id="ENCSR046BCI"
  fi
  if [ "$cell_type" == "CACO2" ]; then
    enc_id="ENCSR100LIJ"
  fi
  if [ "$cell_type" == "CALU3" ]; then
    enc_id="ENCSR935RNW"
  fi
  if [ "$cell_type" == "HUVEC" ]; then
    enc_id="ENCSR098LLB"
  fi
  if [ "$cell_type" == "MCF10A" ]; then
    enc_id="ENCSR799DGV"
  fi
  
  
  if [ -z "$enc_id" ]; then
    echo "Input cell type argument (and spell it correctly)." && exit 1
  fi
   
  modisco_dir="/users/kcochran/projects/procapnet/modisco_out/procap/$cell_type/strand_merged_umap/merged"
  
  motifs_dir="/users/kcochran/projects/procapnet/motifs_out/procap/$cell_type/strand_merged_umap/merged"

  dest_dir="/users/kcochran/oak/kcochran/procapnet_portal_uploads/to_upload/$enc_id"

  mkdir -p "$dest_dir"

  python merge_motif_hits.py "$motifs_dir/profile_hits.bed" "$motifs_dir/counts_hits.bed" "$dest_dir/${enc_id}.motif_instances.bed"
  cat "$dest_dir/${enc_id}.motif_instances.bed" | gzip -nc > "$dest_dir/${enc_id}.motif_instances.bed.gz"
  bedToBigBed "$dest_dir/${enc_id}.motif_instances.bed" "$chrom_sizes" "$dest_dir/${enc_id}.motif_instances.bb" -tab -type=bed6
  chmod ugo+r "$dest_dir/${enc_id}.motif_instances.bb"

  modisco_motifs_tar_dir="$dest_dir/tfmodisco.motifs.raw"
  modisco_reports_tar_dir="$dest_dir/tfmodisco.reports"   

  # fresh start in these dirs, for if I run this script a bunch
  if [ -d "$modisco_reports_tar_dir" ]; then
    rm -r "$modisco_reports_tar_dir"
  fi
  mkdir -p "$modisco_reports_tar_dir"
  

  for task in "${tasks[@]}"; do
    modisco_motifs_dest_dir="$modisco_motifs_tar_dir/$task"

    if [ -d "$modisco_motifs_dest_dir" ]; then
      rm -r "$modisco_motifs_dest_dir"
    fi
    mkdir -p "$modisco_motifs_dest_dir"
    
    contrib_scores_tar_dir="$dest_dir/contrib_tar/$task"
    
    if [ -d "$contrib_scores_tar_dir" ]; then
      rm -r "$contrib_scores_tar_dir"
    fi
    mkdir -p "$contrib_scores_tar_dir"
    

    # copy raw modisco hdf5
    cp "$modisco_dir/${task}_modisco_results.hd5" "$modisco_motifs_dest_dir/${enc_id}.tfmodisco.raw_output.${task}.hd5"
    
    # make all the meme files from every pattern, put in motifs dir to tar
    python modisco_results_to_meme_files.py "$modisco_motifs_dest_dir/${enc_id}.tfmodisco.raw_output.${task}.hd5" "$modisco_motifs_dest_dir/meme_files/" "${enc_id}.${task}"

    # make report pdf and put in reports dir to tar
    python generate_modisco_report_pdf.py "$cell_type" "$task" "$modisco_reports_tar_dir/${enc_id}.tfmodisco.report.${task}.pdf"
  
    # copy inputs to modisco to contrib_scores_tar_dir
    python save_modisco_inputs_to_tar.py "$cell_type" "$task" "$contrib_scores_tar_dir/$enc_id"
    
    # make tar
    
    if [ -f "$dest_dir/contrib_tar/${enc_id}.contrib_scores.raw.tfmodisco_inputs.${task}.tar.gz" ]; then
      rm "$dest_dir/contrib_tar/${enc_id}.contrib_scores.raw.tfmodisco_inputs.${task}.tar.gz"
    fi
    
    cp "/users/kcochran/oak/kcochran/procapnet_portal_uploads/to_upload/contribution_scores.tar.README.txt" "$contrib_scores_tar_dir"
    
    pushd "$dest_dir/contrib_tar/"
    tar -czvf "${enc_id}.contrib_scores.raw.tfmodisco_inputs.${task}.tar.gz" "$task"
    popd
  
  done  # end of tasks loop
  
  # delete tar, about to overwrite
  if [ -f "$dest_dir/${enc_id}.tfmodisco.motifs.raw.tar.gz" ]; then
    rm "$dest_dir/${enc_id}.tfmodisco.motifs.raw.tar.gz"
  fi
  
  # copy readme to dir to be tarred
  cp "/users/kcochran/oak/kcochran/procapnet_portal_uploads/to_upload/tfmodisco.motifs.README.txt" "$modisco_motifs_tar_dir"
  
  # tar (trust me, you need to cd over to where you're going to tar or it breaks)
  pushd "$dest_dir"
  tar -czvf "${enc_id}.tfmodisco.motifs.raw.tar.gz" "tfmodisco.motifs.raw"
  popd
  
  
  # deleted below for now because Aman can tar these himself (thanks Aman!)
  
  #if [ -f "$dest_dir/${enc_id}.tfmodisco.reports.tar.gz" ]; then
    #rm "$dest_dir/${enc_id}.tfmodisco.reports.tar.gz"
  #fi
  
  #cp "/users/kcochran/oak/kcochran/procapnet_portal_uploads/to_upload/tfmodisco.report.README.txt" "$modisco_reports_tar_dir/"
  #pushd "$dest_dir"
  #tar -czvf "${enc_id}.tfmodisco.reports.tar.gz" "tfmodisco.reports"
  #popd
  
done
  
echo "Done."


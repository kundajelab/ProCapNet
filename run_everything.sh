#!/bin/bash

# all the commands I ran to run everything

git clone https://github.com/kundajelab/nascent_RNA_models.git .

./setup_project_directory.sh

pushd src/0_download_files/
./0_runall.sh
popd

pushd src/1_process_data/
./1_runall.sh
./1_runall_optional_annotations.sh
popd

pushd src/2_train_models/
./2_runall.sh
./2_runall_optional.sh
popd

pushd src/3_eval_models/
./3_runall.sh
popd

pushd src/4_interpret_models/
./4_runall.sh
popd

pushd src/5_modisco/
./5_runall.sh
popd

pushd src/6_call_motifs/
./6_runall.sh
popd


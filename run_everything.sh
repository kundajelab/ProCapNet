#!/bin/bash

# all the commands I ran to run everything

root="/users/kcochran/projects/procapnet"

mkdir -p "$root"
cd "$root"
git clone https://github.com/kundajelab/nascent_RNA_models.git .

./setup_project_directory.sh

./src/0_download_files/0_runall.sh

./src/1_process_data/1_runall.sh
./src/1_process_data/1_runall_optional_annotations.sh

./src/2_train_models/2_runall.sh

./src/3_eval_models/3_runall.sh

./src/4_interpret_models/4_runall.sh

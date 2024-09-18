#!/bin/bash

set -e

# run this script before you run the notebooks to generate plots

# first: to create the directory these scripts are in, clone puffin github repo
# (https://github.com/jzhoulab/puffin)

# These scripts and notebooks have to live in the puffin repo directory itself
# because of a relative path encoded in the puffin model code.


# Then, run these scripts:

gpu="MIG-f80e9374-504a-571b-bac0-6fb00750db4c"

python make_puffin_train_test_data.py

python retrain_procapnet_on_puffin_data.py "$gpu"

# ...and you are good to run the notebooks



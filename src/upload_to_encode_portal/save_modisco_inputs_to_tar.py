import os
import shutil
import sys
import numpy as np

sys.path.append("../5_modisco")
from modiscolite_utils import load_sequences, load_scores

sys.path.append("../2_train_models")
from file_configs import MergedFilesConfig


assert len(sys.argv) == 4, len(sys.argv)  # expecting celltype, model_type, task
cell_type, task, dest_dir_with_file_prefix = sys.argv[1:]

assert task in ["profile", "counts"]


data_type = "procap"
model_type = "strand_merged_umap"
in_window = 2114

config = MergedFilesConfig(cell_type, model_type, data_type)
    
if task == "profile":
    scores_path = config.profile_scores_path
else:
    scores_path = config.counts_scores_path


# load what will be saved

onehot_seqs = load_sequences(config.genome_path, config.chrom_sizes,
                             config.all_peak_path,
                             config.slice, in_window=in_window)

scores = load_scores(scores_path, config.slice, in_window=in_window)


# save as numpy arrays

np.save(dest_dir_with_file_prefix + ".input.seqs.npy", onehot_seqs)
np.save(dest_dir_with_file_prefix + ".input.hyp.contrib.scores.npy", scores)

# also copy over the peaks file

shutil.copyfile(config.all_peak_path, dest_dir_with_file_prefix + ".peaks.all.bed.gz")

print("Done copying stuff for scores tar,", cell_type, task, dest_dir_with_file_prefix)

import sys
assert len(sys.argv) == 2

sys.path.append('../2_train_models')

from utils import load_json
import os
import shutil


mitra_proj_dir = "/srv/www/kundaje/kcochran/nascent_RNA/"


config_path = sys.argv[1]

config = load_json(config_path)

data_type = config["data_type"]
cell_type = config["cell_type"]
model_type = config["model_type"]
timestamp = config["timestamp"]

profile_scores_path = config["profile_scores_path"].replace(".npy", ".bigWig")
counts_scores_path = config["counts_scores_path"].replace(".npy", ".bigWig")

print("Files to copy:")
print(profile_scores_path)
print(counts_scores_path)

dest_dir = mitra_proj_dir + "/".join([data_type, cell_type, model_type])
dest_prefix = dest_dir + "/" + timestamp + "_"

profile_bw_dest_path = dest_prefix + os.path.basename(profile_scores_path)
counts_bw_dest_path = dest_prefix + os.path.basename(counts_scores_path)

print("Copying to:")
print(profile_bw_dest_path)
print(counts_bw_dest_path)

shutil.copy2(profile_scores_path, profile_bw_dest_path)
shutil.copy2(counts_scores_path, counts_bw_dest_path)

print("Done.")

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

pred_profiles_train_val_path = config["pred_profiles_train_val_path"]

pos_bw_path = pred_profiles_train_val_path.replace(".npy", ".pos.bigWig")
neg_bw_path = pred_profiles_train_val_path.replace(".npy", ".neg.bigWig")

print("Files to copy:")
print(pos_bw_path)
print(neg_bw_path)

dest_dir = mitra_proj_dir + "/".join([data_type, cell_type, model_type])
dest_prefix = dest_dir + "/" + timestamp + "_"

pos_bw_dest_path = dest_prefix + os.path.basename(pos_bw_path)
neg_bw_dest_path = dest_prefix + os.path.basename(neg_bw_path)

print("Copying to:")
print(pos_bw_dest_path)
print(neg_bw_dest_path)

shutil.copy2(pos_bw_path, pos_bw_dest_path)
shutil.copy2(neg_bw_path, neg_bw_dest_path)

print("Done.")

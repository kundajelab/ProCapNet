import os, sys
assert len(sys.argv) in [4,5], len(sys.argv)  # expecting celltype, model_type, timestamp, maybe gpu
cell_type, model_type, timestamp = sys.argv[1:4]
if len(sys.argv) == 5:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]


sys.path.append("../2_train_models")
from file_configs import DeepshapFilesConfig
config = DeepshapFilesConfig(cell_type, model_type, timestamp)

config.copy_input_files()
config.save_config()


from deepshap_utils import run_deepshap

print("Running deepshap...")

run_deepshap(config.genome_path,
             config.chrom_sizes,
             config.plus_bw_path,
             config.minus_bw_path,
             config.train_val_peak_path,
             config.model_save_path,
             config.profile_scores_path,
             config.profile_onehot_scores_path,
             config.counts_scores_path,
             config.counts_onehot_scores_path,
             in_window=config.in_window,
             out_window=config.out_window,
             stranded=config.stranded_model)

print("Done running deepshap.")
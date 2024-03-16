import os, sys
assert len(sys.argv) in [5,6], len(sys.argv)  # expecting celltype, model_type, fold, timestamp, maybe gpu
cell_type, model_type, fold, timestamp = sys.argv[1:5]
if len(sys.argv) == 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]

sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig, MergedFilesConfig

rampage_config = FoldFilesConfig(cell_type, model_type, fold, timestamp, data_type = "rampage")
procap_config = MergedFilesConfig(cell_type, model_type, data_type = "procap")

in_window, out_window = rampage_config.load_model_params()


from deepshap_utils import run_deepshap

print("Running deepshap...")

run_deepshap(rampage_config.genome_path,
             rampage_config.chrom_sizes,
             rampage_config.plus_bw_path,
             rampage_config.minus_bw_path,
             procap_config.all_peak_path, ###
             rampage_config.model_save_path,
             rampage_config.profile_scores_path.replace("all_", "all_procap_peaks_"),
             rampage_config.profile_onehot_scores_path.replace("all_", "all_procap_peaks_"),
             rampage_config.counts_scores_path.replace("all_", "all_procap_peaks_"),
             rampage_config.counts_onehot_scores_path.replace("all_", "all_procap_peaks_"),
             in_window=in_window,
             out_window=out_window,
             stranded=rampage_config.stranded_model)

print("Done running deepshap.")

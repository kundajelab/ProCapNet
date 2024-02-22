import os, sys
assert len(sys.argv) in [5,6], len(sys.argv)  # expecting celltype, model_type, fold, timestamp, maybe gpu
cell_type, model_type, fold, timestamp = sys.argv[1:5]
if len(sys.argv) == 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
data_type = "procap"

sys.path.append("../2_train_models")

if "promoters_only" in model_type:
    from file_configs_promoters_only import PromotersOnlyFoldFilesConfig as FilesConfig
else:
    from file_configs import FoldFilesConfig as FilesConfig

config = FilesConfig(cell_type, model_type, fold, timestamp, data_type = data_type)
in_window, out_window = config.load_model_params()


from deepshap_utils import run_deepshap

print("Running deepshap...")

run_deepshap(config.genome_path,
             config.chrom_sizes,
             config.plus_bw_path,
             config.minus_bw_path,
             config.all_peak_path,
             config.model_save_path,
             config.profile_scores_path,
             config.profile_onehot_scores_path,
             config.counts_scores_path,
             config.counts_onehot_scores_path,
             in_window=in_window,
             out_window=out_window,
             stranded=config.stranded_model)

print("Done running deepshap.")

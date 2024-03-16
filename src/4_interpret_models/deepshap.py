import os
import sys

from deepshap_utils import run_deepshap

sys.path.append("../2_train_models")


assert len(sys.argv) in [5,6], len(sys.argv)  # expecting celltype, model_type, fold, timestamp, maybe gpu

cell_type, model_type, data_type, fold, timestamp = sys.argv[1:6]

if len(sys.argv) == 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]

possible_cell_types = ["K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"]
assert cell_type in possible_cell_types, cell_type

model_types = ["strand_merged_umap", "promoters_only_strand_merged_umap",
               "strand_merged_umap_replicate"]
assert model_type in model_types, model_type

assert data_type in ["procap", "rampage", "cage"], data_type
assert fold in ["1", "2", "3", "4", "5", "6", "7"], fold

if "promoters_only" in model_type:
    from file_configs_promoters_only import PromotersOnlyFoldFilesConfig as FilesConfig
else:
    from file_configs import FoldFilesConfig as FilesConfig

config = FilesConfig(cell_type, model_type, fold, timestamp, data_type = data_type)
in_window, out_window = config.load_model_params()


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

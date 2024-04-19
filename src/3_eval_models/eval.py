import os
import sys

from eval_utils import run_eval

sys.path.append("../2_train_models")


assert len(sys.argv) in [6,7], len(sys.argv)  # expecting celltype, model_type, data_type, fold, timestamp, maybe gpu

cell_type, model_type, data_type, fold, timestamp = sys.argv[1:6]

if len(sys.argv) == 7:
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



print("Predicting on test set for performance metrics...")

run_eval(config.genome_path,
        config.chrom_sizes,
        config.plus_bw_path,
        config.minus_bw_path,
        config.test_peak_path,
        config.model_save_path,
        config.pred_profiles_test_path,
        config.pred_logcounts_test_path,
        config.metrics_test_path,
        config.log_test_path,
        in_window=in_window,
        out_window=out_window,
        stranded=config.stranded_model)


print("Predicting on whole peak set for downstream analysis...")


run_eval(config.genome_path,
        config.chrom_sizes,
        config.plus_bw_path,
        config.minus_bw_path,
        config.all_peak_path,
        config.model_save_path,
        config.pred_profiles_all_path,
        config.pred_logcounts_all_path,
        config.metrics_all_path,
        config.log_all_path,
        in_window=in_window,
        out_window=out_window,
        stranded=config.stranded_model)


print("Done - end of eval.py.")


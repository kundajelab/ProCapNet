import os, sys
assert len(sys.argv) in [5,6], len(sys.argv)  # expecting celltype, model_type, fold, timestamp, maybe gpu
cell_type, model_type, fold, timestamp = sys.argv[1:5]
if len(sys.argv) == 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
data_type = "procap"

sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig as FilesConfig
config = FilesConfig(cell_type, model_type, fold, timestamp, data_type = data_type)
in_window, out_window = config.load_model_params()

config.save_config() # TODO: remove

from eval_utils import run_eval

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


print("Done - end of val.py.")


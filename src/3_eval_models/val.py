import os, sys
assert len(sys.argv) in [4,5], len(sys.argv)  # expecting celltype, model_type, timestamp, maybe gpu
cell_type, model_type, timestamp = sys.argv[1:4]
if len(sys.argv) == 5:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]


sys.path.append("../2_train_models")
from file_configs import ValFilesConfig
config = ValFilesConfig(cell_type, model_type, timestamp)

config.copy_input_files()
config.save_config()


from val_utils import run_val

print("Predicting on val set for performance metrics...")

run_val(config.genome_path,
        config.chrom_sizes,
        config.plus_bw_path,
        config.minus_bw_path,
        config.val_peak_path,
        config.model_save_path,
        config.pred_profiles_val_path,
        config.pred_logcounts_val_path,
        config.metrics_val_path,
        config.log_val_path,
        in_window=config.in_window,
        out_window=config.out_window,
        stranded=config.stranded_model)


print("Predicting on train + val set for downstream analysis...")


run_val(config.genome_path,
        config.chrom_sizes,
        config.plus_bw_path,
        config.minus_bw_path,
        config.train_val_peak_path,
        config.model_save_path,
        config.pred_profiles_train_val_path,
        config.pred_logcounts_train_val_path,
        config.metrics_train_val_path,
        config.log_train_val_path,
        in_window=config.in_window,
        out_window=config.out_window,
        stranded=config.stranded_model)


print("Done - end of val.py.")


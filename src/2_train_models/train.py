import os
import sys

# expecting cell type, model_type, fold #, maybe gpu

assert len(sys.argv) in [4,5], len(sys.argv)  
cell_type, model_type, fold = sys.argv[1:4]
if len(sys.argv) == 5:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]
data_type = "procap"


from data_loading_multi_source import load_data_loader
from data_loading import extract_peaks

sys.path.append("../utils")
from misc import ensure_parent_dir_exists

from torch.optim import Adam


# Load Hyperparameters, Filepaths from Configs

from file_configs import FoldFilesConfig as FilesConfig
config = FilesConfig(cell_type, model_type, fold, data_type = data_type)


# Init Model and Params Object

if config.model_type == "strand_merged_umap":
    from BPNet_strand_merged_umap import Model
    
elif config.model_type == "stranded_umap":
    from BPNet_stranded_umap import Model
elif config.model_type == "strand_merged":
    from BPNet_strand_merged import Model
elif config.model_type == "stranded":
    from BPNet_stranded import Model
else:
    raise NotImplementedError(config.model_type + " is not a valid model type.")

from hyperparams import DefaultParams as Params
params = Params()
    
model = Model(config.model_save_path,
              n_filters=params.n_filters,
              n_layers=params.n_layers,
              trimming=params.trimming,
              alpha=params.counts_weight)


# Save Filepaths + Variables + Model Arch to Files

config.save_config()

ensure_parent_dir_exists(config.params_path)
params.save_config(config.params_path)

ensure_parent_dir_exists(config.arch_path)
model.save_model_arch_to_txt(config.arch_path)


# Load Training + Validation Data

train_data_loader = load_data_loader(config.genome_path,
                                     config.chrom_sizes,
                                     config.plus_bw_path,
                                     config.minus_bw_path,
                                     [config.train_peak_path, config.dnase_train_path],
                                     params.source_fracs,
                                     config.mask_bw_path,
                                     params.in_window,
                                     params.out_window,
                                     params.max_jitter,
                                     params.batch_size)

val_sequences, val_profiles = extract_peaks(config.genome_path,
                                            config.chrom_sizes,
                                            config.plus_bw_path,
                                            config.minus_bw_path,
                                            config.val_peak_path,
                                            in_window=params.in_window,
                                            out_window=params.out_window,
                                            max_jitter=0,
                                            verbose=True)


# Model Training

model = model.cuda()
optimizer = Adam(model.parameters(), lr=params.learning_rate)

model.fit_generator(train_data_loader, optimizer,
                    X_valid=val_sequences,
                    y_valid=val_profiles,
                    max_epochs=params.max_epochs,
                    validation_iter=params.val_iters,
                    batch_size=params.batch_size,
                    early_stop_epochs=params.early_stop_epochs)

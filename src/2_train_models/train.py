import os
import sys
from torch.optim import Adam

from data_loading_multi_source import load_data_loader
from data_loading import extract_peaks

from BPNet_strand_merged_umap import Model

sys.path.append("../utils")
from misc import ensure_parent_dir_exists


# Script inputs: expecting cell type, model_type, fold #, maybe gpu

assert len(sys.argv) in [5,6], len(sys.argv)  

cell_type, model_type, data_type, fold = sys.argv[1:5]

if len(sys.argv) == 5:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]


possible_cell_types = ["K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"]
assert cell_type in possible_cell_types, cell_type

model_types = ["strand_merged_umap", "promoters_only_strand_merged_umap",
               "strand_merged_umap_replicate"]
assert model_type in model_types, model_type

assert data_type in ["procap", "rampage", "cage"], data_type
assert fold in ["1", "2", "3", "4", "5", "6", "7"], fold


# Load Hyperparameters, Filepaths from Configs

from hyperparams import DefaultParams as Params
params = Params()


if "promoters_only" in model_type:
    # point to peak files that only have promoter-overlap peaks in them,
    # and save these models in a different directory
    print("Training model only on promoter examples.")
    from file_configs_promoters_only import PromotersOnlyFoldFilesConfig as FilesConfig
else:
    from file_configs import FoldFilesConfig as FilesConfig

config = FilesConfig(cell_type, model_type, fold, data_type = data_type)


if "replicate" in model_type:
    # if training replicate models, change the random seed
    print("Training model with different random seed than normal.")
    random_seed = 13169
else:
    random_seed = 0

    
# Init Model and Params Object

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
                                     params.batch_size,
                                     generator_random_seed=random_seed)

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

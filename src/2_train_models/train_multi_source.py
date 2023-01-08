import os
import sys

# expecting cell type, maybe gpu
assert len(sys.argv) in [3,4], len(sys.argv)  
cell_type, model_type = sys.argv[1:3]
if len(sys.argv) == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]


from data_loading import extract_peaks
from data_loading_multi_source import load_data_loader
from torch.optim import Adam


# Load Hyperparameters, Filepaths from Configs

from file_configs import TrainWithDNasePeaksFilesConfig as FilesConfig
config = FilesConfig(cell_type, model_type)


# Init Model and Params Object

if config.model_type == "strand_merged_umap":
    from BPNet_strand_merged_umap import Model
    from hyperparams import DefaultParams as Params
elif config.model_type == "stranded_umap":
    from BPNet_stranded_umap import Model
    from hyperparams import DefaultParams as Params
elif config.model_type == "strand_merged":
    from BPNet_strand_merged import Model
    from hyperparams import DefaultParams as Params
elif config.model_type == "stranded":
    from BPNet_stranded import Model
    from hyperparams import DefaultParams as Params
elif config.model_type == "EMD_strand_merged":
    from BPNet_EMD_strand_merged import Model
    from hyperparams import EMDParams as Params
elif config.model_type == "EMD_and_multinomial_strand_merged":
    from BPNet_EMD_and_multinomial_strand_merged import Model
    from hyperparams import EMDParams as Params
else:
    raise NotImplementedError(config.model_type + "is not a valid model type.")

params = Params()
    
if "EMD" in config.model_type:
    model = Model(config.model_save_path,
                  n_filters=params.n_filters,
                  n_layers=params.n_layers,
                  trimming=params.trimming,
                  alpha=params.counts_weight,
                  beta=params.emd_weight)
else:
    model = Model(config.model_save_path,
                  n_filters=params.n_filters,
                  n_layers=params.n_layers,
                  trimming=params.trimming,
                  alpha=params.counts_weight)


# Save Filepaths + Variables + Model Arch to Files

config.copy_input_files()
config.save_config()
params.save_config(config.params_path)
model.save_model_arch_to_txt(config.arch_path)


# Load Training + Validation Data

train_data_loader = load_data_loader(config.genome_path,
                                     config.chrom_sizes,
                                     config.plus_bw_path,
                                     config.minus_bw_path,
                                     config.train_peak_paths,
                                     config.source_fracs,
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

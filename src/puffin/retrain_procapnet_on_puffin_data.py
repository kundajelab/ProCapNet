# point to GPU
import os
import sys

assert len(sys.argv) == 2, len(sys.argv)
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] # expecting GPU ID


sys.path.append("../2_train_models")
from data_loading import load_data_loader
from data_loading import extract_peaks

sys.path.append("../utils")
from misc import ensure_parent_dir_exists

from datetime import datetime
from torch.optim import Adam





class FilesConfig():
    def __init__(self, timestamp = None):
        # timestamp should be None when training a new model, otherwise should use existing
        # serves as unique identifier for a model and all downstream analysis
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.timestamp = timestamp
        print("Timestamp: " + self.timestamp)
        
        self.cell_type = "puffin_data"
        self.model_type = "strand_merged_umap"
        self.data_type = "procap"
        
        self.stranded_model = False
        self.umap = "umap" in self.model_type
        
        self.proj_dir = "/".join(os.getcwd().split("/")[:-2])  # hacky -- don't move this notebook  
        
        ## Store filepaths to everything
        
        # Genome files and annotations
        
        self.genome_path = self.proj_dir + "genomes/hg38.withrDNA.fasta"
        self.chrom_sizes = self.proj_dir + "genomes/hg38.withrDNA.chrom.sizes"
        
        for filepath in [self.genome_path, self.chrom_sizes]:
            assert os.path.exists(filepath), filepath
        
        if self.umap:
            self.mask_bw_path = self.proj_dir + "/annotations/hg38.k36.multiread.umap.bigWig"
            assert os.path.exists(self.mask_bw_path), self.mask_bw_path
        else:
            self.mask_bw_path = None
            
        
        # Data files (peaks, bigWigs)
        
        self.data_dir = "./puffin_data_train_val_test_split/" # relative path from here
        
        self.train_peak_path = "puffin_data_train_val_test_split/train_set_from_ksenia.bed.gz"
        self.val_peak_path = "puffin_data_train_val_test_split/val_set_from_ksenia.bed.gz"
        self.test_peak_path = "puffin_data_train_val_test_split/test_set_from_ksenia.bed.gz"
        
        self.plus_bw_path = "data/resources/agg.plus.allprocap.bedgraph.sorted.merged.bw"
        self.minus_bw_path = "data/resources/agg.minus.allprocap.bedgraph.sorted.merged.bw"

        for filepath in [self.train_peak_path,
                         self.val_peak_path,
                         self.test_peak_path,
                         self.plus_bw_path,
                         self.minus_bw_path]:
            
            assert os.path.exists(filepath), filepath
            
        # Model save files
        
        self.model_dir = "./retrained_procapnet_models/models/" # relative path from here
        
        self.model_save_path = self.model_dir + self.timestamp + ".model"
        
        self.params_path = self.model_dir + self.timestamp + "_params.json"
        self.arch_path = self.model_dir + self.timestamp + "_model_arch.txt"
        
        
        
# Load Hyperparameters, Filepaths from Configs

config = FilesConfig()

from hyperparams import DefaultParams as Params
params = Params()


# Load Training + Validation Data

train_data_loader = load_data_loader(config.genome_path,
                                     config.chrom_sizes,
                                     config.plus_bw_path,
                                     config.minus_bw_path,
                                     config.train_peak_path,
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


### Setup Model



# had to modify these functions from the code in 2_train_models/performance_metrics.py
# because of asserts in the original code that check for whole-number counts,
# which the puffin data fails because it's aggregated + averaged

from losses import MNLLLoss
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

def calc_profile_jsds_and_corrs(profs1, profs2):
    assert profs1.shape == profs2.shape, (profs1.shape, profs2.shape)
    
    # assuming none of these profiles are in log-space

    jsds = []
    pearson_rs = []
    for prof1, prof2 in zip(profs1, profs2):
        # if multiple strands, flatten data across them into 1D array
        prof1 = prof1.flatten()
        prof2 = prof2.flatten()
        
        jsd = jensenshannon(prof1, prof2, base=2)
        jsds.append(jsd)
        
        pearson_r = np.corrcoef(prof1, prof2)[0,1]
        pearson_rs.append(pearson_r)
        
    return np.array(jsds), np.array(pearson_rs)


def compute_performance_metrics(true_profs, log_pred_profs, true_counts, log_pred_counts):
    
    assert true_profs.shape == log_pred_profs.shape, (true_profs.shape, log_pred_profs.shape)
    assert true_counts.shape == log_pred_counts.shape, (true_counts.shape, log_pred_counts.shape)
    assert true_profs.shape[0] == true_counts.shape[0], (true_profs.shape, true_counts.shape)
    assert len(true_counts.shape) == 1, true_counts.shape
    assert true_profs.shape[1:] == (2,1000), true_profs.shape
    # check if log probs are negative
    assert np.all(log_pred_profs <= 0), [n for n in log_pred_profs.flatten() if n >= 0]
        

    # Multinomial NLL
    nll = MNLLLoss(torch.Tensor(log_pred_profs), torch.Tensor(true_profs))

    # Jensen-Shannon divergence
    jsds, prof_pears = calc_profile_jsds_and_corrs(np.exp(log_pred_profs), true_profs)
    jsd = np.mean(jsds)
    prof_pears = np.mean(prof_pears)

    # Total count correlations/MSE
    count_pears = np.corrcoef(log_pred_counts, np.log(true_counts + 1))[0,1]
    count_mse = np.mean((np.log(true_counts + 1) - log_pred_counts)  ** 2)

    return {
        "nll": nll,
        "jsd": jsd,
        "profile_pearson": prof_pears,
        "count_pearson": count_pears,
        "count_mse": count_mse
    }


### Code here is modified from Jacob's Schreiber's
### implementation of BPNet, called BPNet-lite:
### https://github.com/jmschrei/bpnet-lite/

# this model definition is modified from the original in 2_train_models/
# because I decided to keep all the data loading code untouched, but just
# modify two lines inside the training loop to account for the fact that
# the aggregated puffin data is saved in log-10 + 1 in bigwigs.
# The lines I changed are marked with "#####"s

import time 
import numpy as np
import torch

from losses import MNLLLoss, log1pMSELoss
from performance_metrics import pearson_corr, multinomial_log_probs

torch.backends.cudnn.benchmark = True


class Model(torch.nn.Module):

    def __init__(self, model_save_path, n_filters=64, n_layers=8, n_outputs=2,  
        alpha=1, trimming=None):
        super(Model, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.alpha = alpha
        self.trimming = trimming or 2 ** n_layers
        self.model_save_path = model_save_path
        self.train_metrics = []

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])

        self.deconv_kernel_size = 75  # will need in forward() to crop padding
        self.fconv = torch.nn.Conv1d(n_filters, n_outputs,
                                    kernel_size=self.deconv_kernel_size)

        self.relus = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(0, self.n_layers+1)])
        self.linear = torch.nn.Linear(n_filters, 1)

    def forward(self, X):
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.relus[0](self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.relus[i+1](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        X = X[:, :, start - self.deconv_kernel_size//2 : end + self.deconv_kernel_size//2]

        y_profile = self.fconv(X)

        X = torch.mean(X, axis=2)
        y_counts = self.linear(X).reshape(X.shape[0], 1)

        return y_profile, y_counts

    def predict(self, X, batch_size=64, logits = False):
        with torch.no_grad():
            starts = np.arange(0, X.shape[0], batch_size)
            ends = starts + batch_size

            y_profiles, y_counts = [], []
            for start, end in zip(starts, ends):
                X_batch = X[start:end]

                y_profiles_, y_counts_ = self(X_batch)
                if not logits:  # apply softmax
                    y_profiles_ = self.log_softmax(y_profiles_)
                y_profiles.append(y_profiles_.cpu().detach().numpy())
                y_counts.append(y_counts_.cpu().detach().numpy())

            y_profiles = np.concatenate(y_profiles)
            y_counts = np.concatenate(y_counts)
            return y_profiles, y_counts

    def log_softmax(self, y_profile):
        y_profile = y_profile.reshape(y_profile.shape[0], -1)
        y_profile = torch.nn.LogSoftmax(dim=-1)(y_profile)
        y_profile = y_profile.reshape(y_profile.shape[0], self.n_outputs, -1)
        return y_profile

    def save_metrics(self):
        ensure_parent_dir_exists(self.model_save_path)
        
        metrics_filename = self.model_save_path.replace(".model", "_metrics.tsv")
        with open(metrics_filename, "w") as metrics_file:
            for line in self.train_metrics:
                metrics_file.write(line + "\n")
                
    def save_model_arch_to_txt(self, filepath):
        ensure_parent_dir_exists(filepath)
        print(self, file=open(filepath, "w"))

    def fit_generator(self, training_data, optimizer, X_valid=None, 
        y_valid=None, max_epochs=100, batch_size=64, 
        validation_iter=100, early_stop_epochs=10, verbose=True, save=True):

        if X_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=torch.float32).cuda()
            
            #################### because puffin data is saved in log-10 + 1, exponentiate #########################
            y_valid = np.power(10, y_valid) - 1   
                
            y_valid_counts = y_valid.sum(axis=(1,2))

        columns = "Epoch\tIteration\tTraining Time\tValidation Time\t"
        columns += "Train MNLL\tTrain Count log1pMSE\t"
        columns += "Val MNLL\tVal JSD\tVal Profile Pearson\tVal Count Pearson\tVal Count log1pMSE"
        columns += "\tSaved?"
        if verbose:
            print(columns)
            self.train_metrics.append(columns)
        if save:
            ensure_parent_dir_exists(self.model_save_path)

        start = time.time()
        iteration = 0
        best_loss = float("inf")

        for epoch in range(max_epochs):
            tic = time.time()
            time_to_early_stop = False

            for X, y, mask in training_data:
                #################### because puffin data is saved in log-10 + 1, exponentiate #########################
                y = torch.pow(10, y) - 1
                
                X, y, mask = X.cuda(), y.cuda(), mask.cuda()
                
                
                # Clear the optimizer and set the model to training mode
                optimizer.zero_grad()
                self.train()

                # Run forward pass
                y_profile, y_counts = self(X)
                
                # Calculate the profile and count losses
                # Here we apply the mask to the profile task only
                
                profile_loss = 0
                for y_profile_i, y_i, mask_i in zip(y_profile, y, mask):
                    # Apply mask before softmax, so masked bases do not influence softmax
                    y_profile_i = torch.masked_select(y_profile_i, mask_i)[None,...]
                    y_profile_i = torch.nn.LogSoftmax(dim=-1)(y_profile_i)
                    
                    y_i = torch.masked_select(y_i, mask_i)[None,...]
                    
                    profile_loss_i = MNLLLoss(y_profile_i, y_i)
                    profile_loss += profile_loss_i
                    
                # divide by batch size to get the mean loss for this batch
                profile_loss = profile_loss / y.shape[0]
                
                # Calculate the counts loss
                count_loss = log1pMSELoss(y_counts, y.sum(dim=(1, 2)).reshape(-1, 1))

                # Extract the losses for logging
                profile_loss_ = profile_loss.item()
                count_loss_ = count_loss.item()

                # Mix losses together and update the model
                loss = profile_loss + self.alpha * count_loss
                loss.backward()
                optimizer.step()

                # Report measures if desired
                if verbose and iteration % validation_iter == 0 and X_valid is not None and y_valid is not None:
                    train_time = time.time() - start

                    with torch.no_grad():
                        self.eval()

                        tic = time.time()
                        y_profile, y_counts = self.predict(X_valid)
                        valid_time = time.time() - tic

                        measures = compute_performance_metrics(y_valid,
                                                               y_profile,
                                                               y_valid_counts.squeeze(),
                                                               y_counts.squeeze())

                        line = "{}\t{}\t{:4.4}\t{:4.4}\t".format(epoch, iteration,
                            train_time, valid_time)

                        line += "{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}".format(
                            profile_loss_, count_loss_,
                            measures['nll'].mean(), 
                            measures['jsd'].mean(),
                            measures['profile_pearson'].mean(),
                            measures['count_pearson'].mean(), 
                            measures['count_mse'].mean()
                        )

                        valid_loss = measures['nll'].mean() + self.alpha * measures['count_mse'].mean()
                        line += "\t{}".format(valid_loss < best_loss)

                        print(line, flush=True)
                        self.train_metrics.append(line)

                        start = time.time()

                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            epoch_of_best_loss = epoch

                            if save:
                                self = self.cpu()
                                torch.save(self, self.model_save_path)
                                self = self.cuda()
                        else:
                            if epoch_of_best_loss <= epoch - early_stop_epochs:
                                time_to_early_stop = True
                                break
                iteration += 1

            if time_to_early_stop:
                break

        if save:
            self.save_metrics()
            
            
# remember to re-init config for each new model

model = Model(config.model_save_path,
              n_filters=params.n_filters,
              n_layers=params.n_layers,
              trimming=params.trimming,
              alpha=params.counts_weight)


# Save Filepaths + Variables + Model Arch to Files

ensure_parent_dir_exists(config.params_path)
params.save_config(config.params_path)

ensure_parent_dir_exists(config.arch_path)
model.save_model_arch_to_txt(config.arch_path)

model = model.cuda()
optimizer = Adam(model.parameters(), lr=params.learning_rate)

model.fit_generator(train_data_loader, optimizer,
                    X_valid=val_sequences,
                    y_valid=val_profiles,
                    max_epochs=params.max_epochs,
                    validation_iter=params.val_iters,
                    batch_size=params.batch_size,
                    early_stop_epochs=params.early_stop_epochs)


print("Done training ProCapNet.")
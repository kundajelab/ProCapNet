# set the GPU_ID and filepaths below:

# GPU_ID is the NVIDIA string identifier for the GPU on your machine
GPU_ID = "MIG-166d7783-762d-5f61-b31c-549eb4e0fba0"



class FilePaths():
    # an object that holds all the hard-coded filepaths needed.
    # edit the variables below to point to your files
    
    def __init__(self):
        # filepath for where the trained model will be saved eventually
        self.model_save_path = "test_05312025.model"
        
        # filepath for fasta for reference genome
        self.genome_path = '/mnt/lab_data2/kcochran/procapnet/genomes/hg38.withrDNA.fasta'
        # filepath for text file with chromosome sizes for reference genome
        self.chrom_sizes = '/mnt/lab_data2/kcochran/procapnet/genomes/hg38.withrDNA.chrom.sizes'
        
        # filepaths for experimental PRO-cap data bigwigs (both strands):
        # what the model will be trained to predict
        self.plus_bw_path = '/mnt/lab_data2/kcochran/procapnet/data/procap/processed/K562/5prime.pos.bigWig'
        self.minus_bw_path = '/mnt/lab_data2/kcochran/procapnet/data/procap/processed/K562/5prime.neg.bigWig'
        
        # filepath for RO-cap peak regions to train the model on:
        # bed file with 3+ columns in format (chrom, region_start_coord, region_end_coord)
        self.train_peak_path = '/mnt/lab_data2/kcochran/procapnet/data/procap/processed/K562/peaks_fold1_train.bed.gz'
        # filepath for PRO-cap peak regions to eval the model on, mid-training:
        self.val_peak_path = '/mnt/lab_data2/kcochran/procapnet/data/procap/processed/K562/peaks_fold1_val.bed.gz'
        
        # filepath for regions not overlapping PRO-cap peaks to train with, as "background":
        # bed file with 3+ columns in format (chrom, region_start_coord, region_end_coord)
        self.dnase_train_path = '/mnt/lab_data2/kcochran/procapnet/data/procap/processed/K562/dnase_peaks_no_procap_overlap_fold1_train.bed.gz'
        
        # filepath for a bigwig containing 0-to-1 float values,
        # corresponding to how mappable each base is, according to Michael Hoffman's Umap tracks
        # (bases with values < 1 will not be included in profile task loss function calculations)
        self.mask_bw_path = '/mnt/lab_data2/kcochran/procapnet//annotations/hg38.k36.multiread.umap.bigWig'
        
        
config = FilePaths()
        
        
### Load Hyperparameters

class Params():
    # you probably don't want to change any of these
    
    def __init__(self):
        self.n_filters = 512
        self.n_layers = 8
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.counts_weight = 100
        self.max_epochs = 500
        self.val_iters = 100
        self.early_stop_epochs = 10

        self.in_window = 2114
        self.out_window = 1000
        self.trimming = (self.in_window - self.out_window) // 2
        self.max_jitter = 200
        
        # what fraction of each batch is peaks vs. background DHSs
        # (list of 2 floats, which should sum to 1)
        self.source_fracs = [0.875, 0.125]
        
        self.random_seed = 0

        
params = Params()



####################### Everything below here should just work #######################


# these should all be available in the conda environment:
import torch
from torch.optim import Adam
from torch.utils.data import Sampler

import pyBigWig
from pyfaidx import Fasta

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from math import ceil
import time 
import os
import sys

 

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


### Loss functions

def MNLLLoss(logps, true_counts):
    logps = logps.reshape(logps.shape[0], -1)
    true_counts = true_counts.reshape(true_counts.shape[0], -1)

    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return -torch.mean(log_fact_sum - log_prod_fact + log_prod_exp)


def log1pMSELoss(log_predicted_counts, true_counts):
    log_true = torch.log(true_counts+1)
    return torch.nn.MSELoss()(log_predicted_counts, log_true)



### Performance metrics


train_metrics_cols = ["Epoch", "Iteration", "Training Time", "Validation Time",
                      "Train MNLL", "Train JSD", "Train Profile Pearson",
                      "Train Count Pearson", "Train Count log1pMSE",
                      "Val MNLL", "Val JSD", "Val Profile Pearson",
                      "Val Count Pearson", "Val Count log1pMSE", "Saved?"]

def print_metrics_cols(train_metrics_cols = train_metrics_cols):
    print("\t".join([str(num) for num in train_metrics_cols]))

def format_metrics_str(prof_metrics, counts_metrics):
    prof_metric_names = ["mnll", "jsd", "pearson_r"]
    prof_metric_list = [prof_metrics[metric_name] for metric_name in prof_metric_names]
    to_str = "\t".join([str(num) for num in prof_metric_list])
    
    counts_metric_names = ["logcounts_pearson_r", "logMSE"]
    counts_metric_list = [counts_metrics[metric_name] for metric_name in counts_metric_names]
    to_str += "\t" + "\t".join([str(num) for num in counts_metric_list])
    
    return to_str

    
def save_metrics(model, train_metrics_cols = train_metrics_cols):
    model_save_path = model.model_save_path
    metrics_save_path = model_save_path.replace(".model", "_metrics.tsv")
    
    with open(metrics_save_path, "w") as f:
        f.write("\t".join(train_metrics_cols) + "\n")
        for line in model.train_metrics:
            f.write(line + "\n")

    

def calc_counts_metrics(pred_logcounts, true_counts):
    pred_logcounts = pred_logcounts.squeeze()
    true_counts = true_counts.squeeze()
    assert pred_logcounts.shape == true_counts.shape, (pred_logcounts.shape, true_counts.shape)
    
    # if we're looking at data for each strand
    if 2 in true_counts.shape:
        pred_logcounts = pred_logcounts.flatten()
        true_counts = true_counts.flatten()
    
    metrics = dict()
    
    metrics["logcounts_pearson_r"] = np.corrcoef(pred_logcounts, np.log1p(true_counts))[0,1]
    metrics["counts_pearson_r"] = np.corrcoef(np.exp(pred_logcounts), true_counts)[0,1]
    
    metrics["logMSE"] = log1pMSELoss(torch.from_numpy(pred_logcounts),
                                     torch.from_numpy(true_counts)).item()
    
    return metrics


def calc_profile_metrics(pred_profs, true_profs):
    assert pred_profs.shape == true_profs.shape, (pred_profs.shape, true_profs.shape)
    
    # assuming pred profiles are in log-probs space
    #assert np.all(pred_profs < 0), pred_profs
    
    mnlls = MNLLLoss(torch.from_numpy(pred_profs),
                     torch.from_numpy(true_profs)).numpy()
    
    # ... then convert to not-log for the rest of this function
    pred_profs = np.exp(pred_profs)

    jsds = []
    pearson_rs = []
    for pred_prof, true_prof in zip(pred_profs, true_profs):
        # if multiple strands, flatten data across them into 1D array
        pred_prof = pred_prof.flatten()
        true_prof = true_prof.flatten()

        if true_prof.sum() == 0:  # if you add this line, no nans
            continue
        
        # doesn't change result of JSD or Pearson calc,
        # but matters for CCC
        pred_prof = pred_prof / pred_prof.sum(keepdims=True)
        true_prof = true_prof / true_prof.sum(keepdims=True)
        
        jsd = jensenshannon(pred_prof, true_prof, base=2)
        jsds.append(jsd)
        
        pearson_r = np.corrcoef(pred_prof, true_prof)[0,1]
        pearson_rs.append(pearson_r)
        
    assert np.all(~np.isnan(jsds)), jsds
    assert np.all(~np.isnan(pearson_rs)), pearson_rs
    assert np.all(~np.isnan(mnlls)), mnlls
        
    metrics = {"jsd" : np.nanmean(jsds),
               "pearson_r" : np.nanmean(pearson_rs),
               "mnll" : np.nanmean(mnlls)}
    return metrics
    
    
    
class Model(torch.nn.Module):

    def __init__(self, model_save_path,
                 n_filters = 512,
                 n_layers = 8,
                 n_outputs = 1,  # whether the model is stranded or not stranded
                 alpha = 1,
                 trimming = (2114 - 1000) // 2,
                 umap_mask = True):
        
        super(Model, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.alpha = alpha
        self.trimming = trimming or 2 ** n_layers
        self.model_save_path = model_save_path
        self.train_metrics = []

        self.umap_mask = umap_mask
        
        
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])

        self.deconv_kernel_size = 75  # will need in forward() to crop padding
        
        # should always use 2 here, regardless of model loss scheme
        # (we always want to output a prediction of 2 strands)
        self.fconv = torch.nn.Conv1d(n_filters, 2,
                                    kernel_size=self.deconv_kernel_size)

        self.relus = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(0, self.n_layers+1)])
        self.linear = torch.nn.Linear(n_filters, n_outputs)
        
        
    def forward(self, X):
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.relus[0](self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.relus[i+1](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        X = X[:, :, start - self.deconv_kernel_size//2 : end + self.deconv_kernel_size//2]

        y_profile = self.fconv(X)

        X = torch.mean(X, axis=2)
        y_counts = self.linear(X).reshape(X.shape[0], self.n_outputs).squeeze()

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
        # take the softmax over both strands at once
        y_profile = y_profile.reshape(y_profile.shape[0], -1)
        y_profile = torch.nn.LogSoftmax(dim=-1)(y_profile)
        y_profile = y_profile.reshape(y_profile.shape[0], 2, -1)
        return y_profile

    
    def do_midtrain_metrics(self, train_time, epoch, iteration, best_loss,
                            X_val, y_val, X_train_eval, y_train_eval):
        tic = time.time()
        
        do_train_eval = X_train_eval is not None and y_train_eval is not None
        
        with torch.no_grad():
            self.eval()

            if do_train_eval:
                y_profile_train, y_counts_train = self.predict(X_train_eval)
                
                y_train_eval_counts = y_train_eval.sum(axis=-1)
                if self.n_outputs == 1:
                    y_train_eval_counts = y_train_eval_counts.sum(axis=-1)
                
                train_prof_metrics = calc_profile_metrics(y_profile_train, y_train_eval)
                train_counts_metrics = calc_counts_metrics(y_counts_train, y_train_eval_counts)
                train_metrics_str = format_metrics_str(train_prof_metrics, train_counts_metrics)
            else:
                train_metrics_str = ""
                
            y_profile, y_counts = self.predict(X_val)
            
            y_val_counts = y_val.sum(axis=-1)
            if self.n_outputs == 1:
                y_val_counts = y_val_counts.sum(axis=-1)
            
            val_prof_metrics = calc_profile_metrics(y_profile, y_val)
            val_counts_metrics = calc_counts_metrics(y_counts, y_val_counts)
            val_metrics_str = format_metrics_str(val_prof_metrics, val_counts_metrics)

            # get numbers / strings to print out / save
            
            val_time = time.time() - tic
            to_print = "{}\t{}\t{:4.4}\t{:4.4}\t".format(epoch, iteration,
                                                         train_time, val_time)
            to_print += "\t" + train_metrics_str
            to_print += "\t" + val_metrics_str

            val_prof_loss = val_prof_metrics["mnll"]
            val_counts_loss = val_counts_metrics["logMSE"]
            val_loss = val_prof_loss + self.alpha * val_counts_loss
            to_print += "\t{}".format(val_loss < best_loss)

            print(to_print, flush=True)
            self.train_metrics.append(to_print)
            
        return val_loss
    
    
    def do_forward_pass(self, sequences, y_profiles, y_counts):
        # Get predictions
        y_pred_profiles, y_pred_logcounts = self(sequences)
        y_pred_profiles = self.log_softmax(y_pred_profiles)

        # Calculate the profile and count losses
        profile_loss = MNLLLoss(y_pred_profiles, y_profiles)
        count_loss = log1pMSELoss(y_pred_logcounts, y_counts)
        
        return profile_loss, count_loss
    
    
    def do_forward_pass_masked(self, sequences, y_profiles, y_counts, masks):
        # Get predictions
        y_pred_logits, y_pred_logcounts = self(sequences)

        # Calculate the profile and count losses
        # Here we apply the mask to the profile task only

        profile_loss = 0
        for y_pred_logit, y_prof, mask in zip(y_pred_logits, y_profiles, masks):
            # Apply mask before softmax, so masked bases do not influence softmax
            y_pred_logit = torch.masked_select(y_pred_logit, mask)[None,...]
            y_pred_prof = torch.nn.LogSoftmax(dim=-1)(y_pred_logit)

            y_prof = torch.masked_select(y_prof, mask)[None,...]

            
            assert y_pred_prof.shape == y_prof.shape, (y_pred_prof.shape, y_prof.shape)
            profile_loss += MNLLLoss(y_pred_prof, y_prof)

        # divide by batch size to get the mean loss for this batch
        profile_loss = profile_loss / sequences.shape[0]

        # Calculate the counts loss
        count_loss = log1pMSELoss(y_pred_logcounts, y_counts)
        
        return profile_loss, count_loss
    

    def fit_generator(self, training_data, optimizer,
                      X_val, y_val,
                      X_train_eval=None, y_train_eval=None,
                      max_epochs=100, batch_size=64, 
                      validation_iter=100, early_stop_epochs=10,
                      verbose=True, save=True):

        X_val = torch.tensor(X_val, dtype=torch.float32).cuda()

        if X_train_eval is not None:
            X_train_eval = torch.tensor(X_train_eval, dtype=torch.float32).cuda()
                
        if verbose:
            print_metrics_cols()


        start = time.time()
        iteration = 0
        best_loss = float("inf")

        for epoch in range(max_epochs):
            tic = time.time()
            time_to_early_stop = False

            for batch in training_data:
                if self.umap_mask:
                    X, y, masks = batch
                    X, y, masks = X.cuda(), y.cuda(), masks.cuda()
                else:
                    X, y = batch
                    X, y = X.cuda(), y.cuda()
                    masks = None

                y_counts = y.sum(axis=-1)
                if self.n_outputs == 1:
                    y_counts = y_counts.sum(axis=-1)
                    
                # Clear the optimizer and set the model to training mode
                optimizer.zero_grad()
                self.train()

                if self.umap_mask:
                    profile_loss, count_loss = self.do_forward_pass_masked(X, y, y_counts, masks)
                else:
                    profile_loss, count_loss = self.do_forward_pass(X, y, y_counts)

                # Extract the losses for logging
                profile_loss_ = profile_loss.item()
                count_loss_ = count_loss.item()

                # Mix losses together and update the model
                loss = profile_loss + self.alpha * count_loss
                loss.backward()
                optimizer.step()

                # Report measures if desired, save best-so-far model
                if verbose and iteration % validation_iter == 0:
                    with torch.no_grad():
                        train_time = time.time() - start
                        
                        val_loss = self.do_midtrain_metrics(train_time, epoch, iteration,
                                                            best_loss, X_val, y_val,
                                                            X_train_eval, y_train_eval)
                        start = time.time()

                        if val_loss < best_loss:
                            best_loss = val_loss
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
            save_metrics(self)
            
            
            
            
            
### Init Model

model = Model(config.model_save_path,
              n_filters=params.n_filters,
              n_layers=params.n_layers,
              trimming=params.trimming,
              alpha=params.counts_weight)



### Load Data


def one_hot_encode(sequence, alphabet=['A','C','G','T'], dtype='int8', 
    desc=None, verbose=False, **kwargs):
    
    # these characters will be encoded as all-zeros
    ambiguous_nucs = ["Y", "R", "W", "S", "K", "M", "D", "V", "H", "B", "X", "N"]

    d = verbose is False

    sequence = sequence.upper()
    if isinstance(sequence, str):
        sequence = list(sequence)

    alphabet = alphabet or np.unique(sequence)
    alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

    ohe = np.zeros((len(sequence), len(alphabet)), dtype=dtype)
    for i, char in tqdm(enumerate(sequence), disable=d, desc=desc, **kwargs):
        if char in alphabet:
            idx = alphabet_lookup[char]
            ohe[i, idx] = 1
        else:
            assert char in ambiguous_nucs, char

    return ohe


def load_chrom_names(chrom_sizes, filter_out = ["_", "M", "Un", "EBV"], filter_in = ["chr"]):
    with open(chrom_sizes) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    chroms = [line[0] for line in lines]
    
    if filter_out and len(filter_out) > 0:
        chroms = [c for c in chroms if all([filt not in c for filt in filter_out])]
    if filter_in and len(filter_in) > 0:
        chroms = [c for c in chroms if all([filt in c for filt in filter_in])]
    return chroms


def read_fasta(filename, chrom_sizes=None, include_chroms=None, verbose=True):
    if include_chroms is None:
        if chrom_sizes is None:
            print("Assuming human chromosomes in read_fasta.")
            include_chroms = ["chr" + str(i + 1) for i in range(22)]
            include_chroms.extend(["chrX", "chrY"])
        else:
            include_chroms = load_chrom_names(chrom_sizes)

    chroms = {}
    print("Loading genome sequence from " + filename)
    fasta_index = Fasta(filename)
    for chrom in tqdm(include_chroms, disable=not verbose, desc="Reading FASTA"):
        chroms[chrom] = fasta_index[chrom][:].seq.upper()
    return chroms


class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, sequences, signals, masks=None, in_window=2114, out_window=1000,
        reverse_complement=True, random_state=None):
        self.in_window = int(in_window)
        self.out_window = int(out_window)

        self.reverse_complement = reverse_complement
        self.random_state = np.random.RandomState(random_state)

        self.signals = signals
        self.sequences = sequences
        self.masks = masks
        
        self.max_jitter = (sequences.shape[-1] - self.in_window) // 2

        assert len(signals) == len(sequences), (len(signals), len(sequences))
        if masks is not None:
            assert len(signals) == len(masks), (len(signals), len(masks))

        assert signals.shape[-1] == self.out_window + 2 * self.max_jitter, (signals.shape, self.out_window, self.max_jitter)
        assert sequences.shape[-1] == self.in_window + 2 * self.max_jitter, (sequences.shape, self.in_window, self.max_jitter)
        if masks is not None:
            assert masks.shape[-1] == self.out_window + 2 * self.max_jitter, (signals.shape, self.out_window, self.max_jitter)

        assert sequences.shape[1] == 4, sequences.shape
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert np.max(sequences.sum(axis=(1,2))) == self.in_window + 2 * self.max_jitter, np.max(sequences.sum(axis=(1,2)))
        assert set(np.sum(sequences, axis=1).flatten()).issubset(set([0,1]))
        
        to_print = "Data generator loaded " + str(len(sequences)) + " sequences of len " + str(self.in_window)
        to_print += ", profile len " + str(self.out_window) + ", with max_jitter " + str(self.max_jitter)
        to_print += ".\nRC enabled? " + str(self.reverse_complement)
        to_print += "\nMask loaded? " + str(self.masks is not None)
        print(to_print)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.max_jitter == 0:
            j = 0
        else:
            j = self.random_state.randint(self.max_jitter*2)

        X = self.sequences[idx][:, j:j+self.in_window]
        y = self.signals[idx][:, j:j+self.out_window]
        if self.masks is not None:
            m = self.masks[idx][:, j:j+self.out_window]

        if self.reverse_complement and np.random.choice(2) == 1:
            X = X[::-1][:, ::-1]
            y = y[::-1][:, ::-1]
            if self.masks is not None:
                m = m[:, ::-1]  # one strand

        X = torch.tensor(X.copy(), dtype=torch.float32)
        y = torch.tensor(y.copy())
        if self.masks is not None:
            m = torch.tensor(m.copy(), dtype=torch.bool)
            return X, y, m
        return X, y


def extract_peaks(sequences, chrom_sizes, plus_bw_path, minus_bw_path, peak_path,
                  mask_bw_path=None, in_window=2114, out_window=1000, max_jitter=0, verbose=False):

    seqs, signals, masks = [], [], []
    in_width, out_width = in_window // 2, out_window // 2

    if isinstance(sequences, str):
        assert os.path.exists(sequences), sequences
        sequences = read_fasta(sequences, chrom_sizes, verbose=verbose)

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    assert os.path.exists(plus_bw_path), plus_bw_path
    assert os.path.exists(minus_bw_path), minus_bw_path
    plus_bw = pyBigWig.open(plus_bw_path, "r")
    minus_bw = pyBigWig.open(minus_bw_path, "r")
    if mask_bw_path is not None:
        assert os.path.exists(mask_bw_path), mask_bw_path
        mask_bw = pyBigWig.open(mask_bw_path, "r")

    desc = "Loading Peaks"
    d = not verbose
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        start = mid - out_width - max_jitter
        end = mid + out_width + max_jitter
        assert start > 0, start

        sequence = sequences[chrom]

        # Load plus strand signal
        plus_sig = plus_bw.values(chrom, start, end, numpy=True)
        plus_sig = np.abs(np.nan_to_num(plus_sig))

        # Load minus strand signal
        minus_sig = minus_bw.values(chrom, start, end, numpy=True)
        minus_sig = np.abs(np.nan_to_num(minus_sig))

        # Append signal to growing signal list
        assert len(plus_sig) == end - start, (len(plus_sig), start, end)
        assert len(minus_sig) == end - start, (len(minus_sig), start, end)
        signals.append(np.array([plus_sig, minus_sig]))

        if mask_bw_path is not None:
            try:
                mask = mask_bw.values(chrom, start, end, numpy=True)
            except:
                print("Mask not loading for all examples.", chrom, start, end)
                mask = np.zeros((end - start,))

            # binarize to be only 1s and 0s
            mask = np.nan_to_num(mask).astype(int)
            assert len(mask) == end - start, (len(mask), start, end)
            assert set(mask.flatten()).issubset(set([0,1])), set(mask.flatten())
            # double-save the mask to broadcast to strand axis
            masks.append(np.array([mask, mask]))

        # Append sequence to growing sequence list
        s = mid - in_width - max_jitter
        e = mid + in_width + max_jitter

        if isinstance(sequence, str):
            seq = one_hot_encode(sequence[s:e]).T
        else:
            seq = sequence[s:e].T

        assert seq.shape == (4, e - s), (seq.shape, s, e)
        assert set(seq.flatten()) == set([0,1]), set(seq.flatten())
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert set(seq.sum(axis=0)).issubset(set([0, 1])), set(seq.sum(axis=0))
        assert seq.sum() <= e - s, seq
        seqs.append(seq)

    seqs = np.array(seqs)
    signals = np.array(signals)
    
    assert len(seqs) == len(signals), (seqs.shape, signals.shape)
    assert seqs.shape[1] == 4 and seqs.shape[2] == in_window + 2 * max_jitter, seqs.shape
    assert signals.shape[1] == 2 and signals.shape[2] == out_window + 2 * max_jitter, signals.shape
    
    to_print = "== In Extract Peaks =="
    to_print += "\nPeak filepath: " + peak_path
    to_print += "\nSequence length (with jitter): " + str(seqs.shape[-1])
    to_print += "\nProfile length (with jitter): " + str(signals.shape[-1])
    to_print += "\nMax jitter applied: " + str(max_jitter)
    to_print += "\nNum. Examples: " + str(len(seqs))
    to_print += "\nMask loaded? " + str(mask_bw_path is not None)
    print(to_print)
    sys.stdout.flush()

    if mask_bw_path is not None:
        masks = np.array(masks)
        assert masks.shape[1] == 2 and masks.shape[2] == out_window + 2 * max_jitter, masks.shape
        return seqs, signals, masks

    return seqs, signals


def extract_sequences(sequences, chrom_sizes, peak_path, in_window=2114, verbose=False):
    seqs = []
    in_width = in_window // 2

    if isinstance(sequences, str):
        assert os.path.exists(sequences), sequences
        sequences = read_fasta_fast(sequences, chrom_sizes, verbose=verbose)

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    desc = "Loading Peaks"
    d = not verbose
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        s = mid - in_width
        e = mid + in_width
        assert s > 0, start

        sequence = sequences[chrom]

        if isinstance(sequence, str):
            seq = one_hot_encode(sequence[s:e]).T
        else:
            seq = sequence[s:e].T

        assert seq.shape == (4, e - s), (seq.shape, s, e)
        assert set(seq.flatten()) == set([0,1]), set(seq.flatten())
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert set(seq.sum(axis=0)).issubset(set([0, 1])), set(seq.sum(axis=0))
        assert seq.sum() <= e - s, seq
        seqs.append(seq)

    seqs = np.array(seqs)
    assert seqs.shape[1] == 4 and seqs.shape[2] == in_window, seqs.shape
    
    to_print = "== In Extract Sequences =="
    to_print += "\nPeak filepath: " + peak_path
    to_print += "\nSequence length: " + str(seqs.shape[-1])
    to_print += "\nNum. Examples: " + str(len(seqs))
    print(to_print)
    sys.stdout.flush()

    return seqs


def extract_observed_profiles(plus_bw_path, minus_bw_path, peak_path,
                              out_window=1000, verbose=False):
    signals = []
    out_width = out_window // 2

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    assert os.path.exists(plus_bw_path), plus_bw_path
    assert os.path.exists(minus_bw_path), minus_bw_path
    plus_bw = pyBigWig.open(plus_bw_path, "r")
    minus_bw = pyBigWig.open(minus_bw_path, "r")

    desc = "Loading Profiles"
    d = not verbose
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        start = mid - out_width
        end = mid + out_width
        assert start > 0, start
        
        # Load plus strand signal
        plus_sig = plus_bw.values(chrom, start, end, numpy=True)
        plus_sig = np.nan_to_num(plus_sig)

        # Load minus strand signal
        minus_sig = minus_bw.values(chrom, start, end, numpy=True)
        minus_sig = np.nan_to_num(minus_sig)

        # Append signal to growing signal list
        assert len(plus_sig) == end - start, (len(plus_sig), start, end)
        assert len(minus_sig) == end - start, (len(minus_sig), start, end)
        signals.append(np.array([plus_sig, minus_sig]))

    signals = np.array(signals)
    assert signals.shape[1] == 2 and signals.shape[2] == out_window, signals.shape
    
    to_print = "== In Extract Profiles =="
    to_print += "\nPeak filepath: " + peak_path
    to_print += "\nProfile length: " + str(signals.shape[-1])
    to_print += "\nNum. Examples: " + str(len(signals))
    print(to_print)
    sys.stdout.flush()

    return signals


class MultiSourceSampler(Sampler):
    
    def __init__(self, train_generator, source_totals, source_fracs,
                 batch_size):
        
        source_batch_sizes = [ceil(batch_size * frac) for frac in source_fracs]
        # if the int truncation above doesn't create nums
        # that add up to the desired total ...
        if sum(source_batch_sizes) != batch_size:
            source_batch_sizes[0] += batch_size - sum(source_batch_sizes)
        self.source_batch_sizes = source_batch_sizes
        
        self.source_totals = source_totals
        
        self.range_ends = []
        total_so_far = 0
        for source_total in source_totals:
            total_so_far += source_total
            self.range_ends.append(total_so_far)
            
        self.range_starts = [0] + self.range_ends[:-1]
        
        # going to assume source 0 is the real peaks,
        # which we want to sample (close to) 100% of 
        self.num_batches = source_totals[0] // self.source_batch_sizes[0]
        self.len = self.num_batches * source_totals[0]
        
    def __len__(self):
        return self.len
    
    def __iter__(self):
        peak_indices_permuted = torch.randperm(self.range_ends[0]).tolist()

        for batch_i in range(self.num_batches):
            # first source (actual peaks)
            source_0_batch_size = self.source_batch_sizes[0]
            yield from peak_indices_permuted[batch_i * source_0_batch_size : (batch_i + 1) * source_0_batch_size]
            
            for source_i, source_batch_size in enumerate(self.source_batch_sizes):
                if source_i == 0:
                    continue  # we did the 0th source separately above
                    
                start = self.range_starts[source_i]
                end = self.range_ends[source_i]

                yield from torch.randint(low = start, high = end,
                                         size = (source_batch_size,),
                                         dtype = torch.int64).tolist()

                
def load_data_loader(genome_path, chrom_sizes, plus_bw_path, minus_bw_path,
                     peak_paths, source_fracs, mask_bw_path=None,
                     in_window=2114, out_window=1000, max_jitter=0,
                     batch_size=64, generator_random_seed=0):
    
    sequences_all_sources = []
    profiles_all_sources = []
    
    if mask_bw_path:
        masks_all_sources = []
    
        for peak_path in peak_paths:
            sequences, profiles, masks = extract_peaks(genome_path,
                                                       chrom_sizes,
                                                       plus_bw_path,
                                                       minus_bw_path,
                                                       peak_path,
                                                       mask_bw_path=mask_bw_path,
                                                       in_window=in_window,
                                                       out_window=out_window,
                                                       max_jitter=max_jitter,
                                                       verbose=True)
            sequences_all_sources.append(sequences)
            profiles_all_sources.append(profiles)
            masks_all_sources.append(masks)
    
        train_masks = np.concatenate(masks_all_sources)
    
    else:
        for peak_path in peak_paths:
            sequences, profiles = extract_peaks(genome_path,
                                                chrom_sizes,
                                                plus_bw_path,
                                                minus_bw_path,
                                                peak_path,
                                                in_window=in_window,
                                                out_window=out_window,
                                                max_jitter=max_jitter,
                                                verbose=True)
            sequences_all_sources.append(sequences)
            profiles_all_sources.append(profiles)
    
        train_masks = None
            
    train_sequences = np.concatenate(sequences_all_sources)
    train_signals = np.concatenate(profiles_all_sources)

    gen = DataGenerator(sequences=train_sequences,
                        signals=train_signals,
                        masks=train_masks,
                        in_window=in_window,
                        out_window=out_window,
                        random_state=generator_random_seed)

    multi_sampler = MultiSourceSampler(gen,
                                       [a.shape[0] for a in sequences_all_sources],
                                       source_fracs, batch_size)

    train_data = torch.utils.data.DataLoader(gen,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             sampler=multi_sampler)
    return train_data



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
                                     generator_random_seed=params.random_seed)


### Load Validation Data

val_sequences, val_profiles = extract_peaks(config.genome_path,
                                            config.chrom_sizes,
                                            config.plus_bw_path,
                                            config.minus_bw_path,
                                            config.val_peak_path,
                                            in_window=params.in_window,
                                            out_window=params.out_window,
                                            max_jitter=0,
                                            verbose=True)

# for computing metrics on the whole train set during training
train_sequences, train_profiles = extract_peaks(config.genome_path,
                                            config.chrom_sizes,
                                            config.plus_bw_path,
                                            config.minus_bw_path,
                                            config.train_peak_path,
                                            in_window=params.in_window,
                                            out_window=params.out_window,
                                            max_jitter=0,
                                            verbose=True)



### Go: Model Training

model = model.cuda()
optimizer = Adam(model.parameters(), lr=params.learning_rate)

model.fit_generator(train_data_loader, optimizer,
                    X_val=val_sequences,
                    y_val=val_profiles,
                    X_train_eval=train_sequences,
                    y_train_eval=train_profiles,
                    max_epochs=params.max_epochs,
                    validation_iter=params.val_iters,
                    batch_size=params.batch_size,
                    early_stop_epochs=params.early_stop_epochs)

print("Done model training.")
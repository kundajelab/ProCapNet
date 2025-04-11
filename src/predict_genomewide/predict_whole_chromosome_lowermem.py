import sys
assert len(sys.argv) == 4, sys.argv  # expecting cell_type, chromosome name, GPU ID
cell_type, which_chromosome, gpu = sys.argv[1:]

possible_cell_types = ["K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"]
assert cell_type in possible_cell_types, cell_type

possible_chroms = ["chr" + str(i+1) for i in range(22)] + ["chrX", "chrY"]
assert which_chromosome in possible_chroms, which_chromosome

import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


import numpy as np
import torch
from tqdm import tqdm

from pyfaidx import Fasta
from tangermeme.utils import one_hot_encode

sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig
from BPNet_strand_merged_umap import Model


# the input sequence length and output prediction window length
in_window = 2114
out_window = 1000

# for each cell type, we trained multiple models across folds
num_folds = 7

which_genome = "t2t"

fasta_filepath = which_genome + "/" + which_genome + ".fasta"
chrom_sizes_filepath = which_genome + "/" + which_genome + ".chrom.sizes"


# load chromosome info

def load_chrom_size(chrom_sizes_filepath, which_chromosome = which_chromosome):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        
    for line in chrom_sizes_lines:
        if line[0] == which_chromosome:
            return int(line[1])
    return np.nan

chrom_size = load_chrom_size(chrom_sizes_filepath)



# For each cell type, we have a set of ProCapNet models trained across 7 folds,
# which are ID'd by their training timestamps below

if cell_type == "K562":
    timestamps = ["2023-05-29_15-51-40", "2023-05-29_15-58-41", "2023-05-29_15-59-09",
                  "2023-05-30_01-40-06", "2023-05-29_23-21-23", "2023-05-29_23-23-45",
                  "2023-05-29_23-24-11"]
elif cell_type == "A673":
    timestamps = ["2023-06-11_20-11-32","2023-06-11_23-42-00", "2023-06-12_03-29-06",
                  "2023-06-12_07-17-43", "2023-06-12_11-10-59", "2023-06-12_14-36-40",
                  "2023-06-12_17-26-09"]
elif cell_type == "CACO2":
    timestamps = ["2023-06-12_21-46-40", "2023-06-13_01-28-24", "2023-06-13_05-06-53",
                  "2023-06-13_08-52-39", "2023-06-13_13-12-09", "2023-06-13_16-40-41",
                  "2023-06-13_20-08-39"]
elif cell_type == "CALU3":
    timestamps = ["2023-06-14_00-43-44", "2023-06-14_04-26-48", "2023-06-14_09-34-26",
              "2023-06-14_13-03-59", "2023-06-14_17-22-28", "2023-06-14_21-03-11",
              "2023-06-14_23-58-36"]
elif cell_type == "HUVEC":
    timestamps = ["2023-06-16_21-59-35", "2023-06-17_00-20-34", "2023-06-17_02-17-07",
                  "2023-06-17_04-27-08", "2023-06-17_06-42-19", "2023-06-17_09-16-24",
                  "2023-06-17_11-09-38"] 
elif cell_type == "MCF10A":
    timestamps = ["2023-06-15_06-07-40", "2023-06-15_10-37-03", "2023-06-15_16-23-56",
                  "2023-06-15_21-44-32", "2023-06-16_03-47-46", "2023-06-16_09-41-26",
                  "2023-06-16_15-07-01"]
else:
    print("Misspelled cell type?")
    
    
    
### Model loading

def get_model_path(fold, timestamps = timestamps, cell_type = cell_type,
                   model_type = "strand_merged_umap"):
    
    # fold input should be 0-indexed
    config = FoldFilesConfig(cell_type, model_type, str(fold + 1),
                             timestamps[fold], "procap")
    return config.model_save_path


def load_model(model_save_path):
    model = torch.load(model_save_path)
    model = model.eval()
    return model



### where we will save all of the predictions generated

def get_raw_preds_save_dir(model_fold, cell_type = cell_type,
                           which_genome = which_genome,
                           which_chromosome = which_chromosome):
    
    dir_prefix = "/".join(["raw_preds", which_genome, cell_type, which_chromosome])
    save_dir = dir_prefix + "/fold_" + str(model_fold) + "/"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_preds_save_prefix(model_fold, cell_type = cell_type,
                          which_genome = which_genome,
                          which_chromosome = which_chromosome):
    
    preds_out_dir = get_raw_preds_save_dir(model_fold, cell_type = cell_type,
                           which_genome = which_genome, which_chromosome = which_chromosome)
    
    return preds_out_dir + "pred_"


def save_raw_preds(model_fold, pred_profiles, pred_logcounts):
    save_prefix = get_preds_save_prefix(model_fold)
    np.save(save_prefix + "profiles.npy", pred_profiles)
    np.save(save_prefix + "logcounts.npy", pred_logcounts)

def check_if_preds_exist(model_fold):
    save_prefix = get_preds_save_prefix(model_fold)
    profs_saved = os.path.exists(save_prefix + "profiles.npy")
    counts_saved = os.path.exists(save_prefix + "logcounts.npy")
    return profs_saved and counts_saved

def load_raw_preds(model_fold):
    save_prefix = get_preds_save_prefix(model_fold)
    pred_profiles = np.load(save_prefix + "profiles.npy")
    pred_logcounts = np.load(save_prefix + "logcounts.npy")
    return pred_profiles, pred_logcounts

def delete_raw_preds(model_fold):
    save_prefix = get_preds_save_prefix(model_fold)
    os.remove(save_prefix + "profiles.npy")
    os.remove(save_prefix + "logcounts.npy") 
    
    
def get_merged_preds_path(genome = which_genome,
                          chrom = which_chromosome,
                          cell_type = cell_type):
                                      
    merged_preds_dir = "/".join(["raw_preds", genome, cell_type, chrom, "merged"])
    os.makedirs(merged_preds_dir, exist_ok=True)
    return merged_preds_dir + "/preds.npy"




### Sequence loading functions

def read_chromosome_from_fasta(fasta_filepath, which_chromosome = which_chromosome):
    print("Loading genome sequence from " + fasta_filepath + " for " + which_chromosome)
    fasta_index = Fasta(fasta_filepath)
    return fasta_index[which_chromosome][:].seq.upper()


def make_onehot_seqs(chrom_seq_str, stride = 250, chunk_size = in_window):
    print("Loading chromosome sequence, converting to one-hot arrays.")
    
    # chunk up the chromosome sequence to make a tensor of seqs of size
    # [num_seqs, 4, chunk_size] that can go into the model for prediction
    
    # we will convert all non-ACGT characters to all-zeros (like Ns)
    chrom_onehot_seq = one_hot_encode(chrom_seq_str,
                                      ignore=list("NRYWSKMBDVH")) # not needed for T2T
    
    assert chrom_onehot_seq.shape[0] == 4, chrom_onehot_seq.shape
    
    # now, chunk up the sequence into lengths the model expects as input
    
    onehot_seqs = []
    for seq_start in np.arange(0, len(chrom_seq_str) - chunk_size + 1, stride):
        seq_end = seq_start + chunk_size
        onehot_seqs.append(chrom_onehot_seq[:, seq_start : seq_end])
        
    # we'll almost always need to make an extra prediction window at the end,
    # that's offset from the penultimate window by a weird amount,
    # to cover the last bases at the end of the chromosome
    onehot_seqs.append(chrom_onehot_seq[:, - chunk_size :])
        
    return torch.stack(onehot_seqs)



### Predicting functions

def _model_predict(model, X, batch_size=256, logits = False):
    # modified so it can take massive X tensors in
    
    X = X.type(torch.float32)
    
    model = model.cuda()
    with torch.no_grad():
        starts = np.arange(0, X.shape[0], batch_size)
        ends = starts + batch_size

        y_profiles, y_counts = [], []
        for start, end in tqdm(zip(starts, ends)):
            X_batch = X[start:end].cuda()

            y_profiles_, y_counts_ = model(X_batch)
            if not logits:  # apply softmax
                y_profiles_ = model.log_softmax(y_profiles_)
            y_profiles.append(y_profiles_.cpu().detach().numpy())
            y_counts.append(y_counts_.cpu().detach().numpy())

        y_profiles = np.concatenate(y_profiles)
        y_counts = np.concatenate(y_counts)
        return y_profiles, y_counts

    
def _model_predict_with_rc(model, onehot_seqs):
    # this function makes a prediction for 1 model, for 1 sequence
    
    # it gets a prediction for both the original sequence and its reverse-complement,
    # then averages the two, which tends to improve accuracy
    
    with torch.no_grad():
        pred_profiles, pred_logcounts = _model_predict(model, onehot_seqs)
        rc_pred_profiles, rc_pred_logcounts = _model_predict(model, torch.flip(onehot_seqs, [-1, -2]))
    
    # reverse-complement (strand-flip) BOTH the profile and counts
    rc_pred_profiles = rc_pred_profiles[:, ::-1, ::-1]
    rc_pred_logcounts = rc_pred_logcounts[:, ::-1]
    
    # take the average prediction across the fwd and RC sequences
    # (profile average is in raw probs space, not logits; counts average is in log counts space)
    merged_pred_profiles = np.log(np.array([np.exp(pred_profiles),
                                            np.exp(rc_pred_profiles)]).mean(axis=0))
    merged_pred_logcounts = np.array([pred_logcounts, rc_pred_logcounts]).mean(axis=0)
    
    return merged_pred_profiles, merged_pred_logcounts


def predict_one_fold(model_fold, chrom_seq):
    print("===", "Predicting, model fold: ", model_fold, "===")
    
    # This function makes predictions across a whole chromosome, for one model
    
    model = load_model(get_model_path(model_fold))

    onehot_chunk_seqs = make_onehot_seqs(chrom_seq)

    print("onehot_seqs shape:", onehot_chunk_seqs.shape)
    
    pred_profiles, pred_logcounts = _model_predict_with_rc(model, onehot_chunk_seqs)

    # these are the raw-est possible predictions; will merge these and then delete later
    save_raw_preds(model_fold, pred_profiles, pred_logcounts)

       



def _merge_preds_across_folds(num_folds = num_folds):
    # load model predictions across all folds, then average across the folds
    
    pred_profs_across_folds = []
    pred_logcounts_across_folds = []
    for model_fold in range(num_folds):
        _pred_profiles, _pred_logcounts = load_raw_preds(model_fold)
        pred_profs_across_folds.append(_pred_profiles)
        pred_logcounts_across_folds.append(_pred_logcounts)

    # exponentiating profiles here
    pred_profs_across_folds = np.exp(np.array(pred_profs_across_folds))
    pred_logcounts_across_folds = np.array(pred_logcounts_across_folds)

    # average across folds
    pred_profs = pred_profs_across_folds.mean(axis=0)
    pred_logcounts = pred_logcounts_across_folds.mean(axis=0)
    return pred_profs, pred_logcounts
        
        
        
def merge_preds(chrom_size = chrom_size, stride = 250,
                chunk_size = in_window, out_window = out_window,
                num_folds = num_folds):
    
    print("Merging preds across folds, across chunks.")
    
    # load and merge preds across model folds for this chunk
    pred_profs, pred_logcounts = _merge_preds_across_folds()

    print("Preds merged across folds.")
    
    # combine counts and profile predictions
    pred_profs_scaled = pred_profs * np.exp(pred_logcounts)[..., None]
    

    # then, get average of scaled profiles tiled across the whole chromosome 
    
    # make list of chunk starts, so you can tell when a base was part of a chunk
    
    # this count includes the last chunk, which might be offset differently from the rest
    chrom_end_offset = (chunk_size - out_window) // 2
    effective_chrom_size = chrom_size - 2 * chrom_end_offset
    
    num_chunks = ((chrom_size - chunk_size) // stride) + 1
    print("Chunks to merge:", num_chunks + 1)
    
    chunk_starts = list(stride * np.arange(0, num_chunks) + chrom_end_offset) 
    if chrom_size > chunk_size:
        chunk_starts.append(- (chunk_size - chrom_end_offset))
    
    
    # then, for each base, get avg pred across all tiles that overlapped it
    
    # we will take average by taking sum, then dividing by the # of sums
    # (most bases will have the same # of sums, but the edge cases won't)

    
    # first: just sum all the tiled predictions into one long vector,
    # keeping track of relative positioning of tiles
    
    sum_preds = np.zeros((2, chrom_size))
    
    for chunk_i, chunk_start in enumerate(chunk_starts):
        sum_preds[:, chunk_start : chunk_start + out_window] += pred_profs_scaled[chunk_i]
    
    
    # second: count the number of sums that will happen at each base
    
    assert out_window % stride == 0 # ONLY WORKS IF out_window IS A MULTIPLE OF stride!!
    num_overlaps_default = out_window // stride

    # set default number of tiles overlapping a base to be [out_window // stride]
    num_sums = np.ones((1, chrom_size,)) * num_overlaps_default
    
    # then just check the edge cases, where the default won't be true, and adjust them

    for base in range(chrom_size):
        # if not on the far left edge or far right edge of this chunk
        # (where there are fewer tiles covering the bases than usual)
        if not (base <= out_window or base >= chrom_size - out_window - 2):
            continue

        num_sums_this_base = 0
        for chunk_i, chunk_start in enumerate(chunk_starts):
            # if not on the far left or far right edge tiles
            if not (chunk_i <= num_overlaps_default or chunk_i >= - num_overlaps_default - 1):
                continue 
                
            # check if this tile actually overlaps this base
            relative_position_of_base = base - chunk_start
            if relative_position_of_base >= 0 and relative_position_of_base < out_window:
                num_sums_this_base += 1
                
        num_sums[:, base] = num_sums_this_base
    
    # finally: turn sum into mean by dividing
    
    # hacky way to not divide by zero (setting numerator to 0 instead)
    sum_preds[:, num_sums.squeeze() == 0] = 0
    num_sums[num_sums == 0] = 1
    avg_preds = sum_preds / num_sums
    
    # save to file
    save_path = get_merged_preds_path()
    print("Saving merged preds to ", save_path)
    np.save(save_path, avg_preds)
    
    print("Deleting raw prediction files (after merging).")
    for model_fold in range(num_folds):
        delete_raw_preds(model_fold)
        
        
        
    
### Go!

def main():
    chrom_seq = read_chromosome_from_fasta(fasta_filepath)

    for model_fold in range(len(timestamps)):
        if check_if_preds_exist(model_fold):
            print("File found, skipping fold " + str(model_fold))
        else:
            predict_one_fold(model_fold, chrom_seq)

    merge_preds()
     
    print("Done!")
    
    
if __name__ == "__main__":
    main()


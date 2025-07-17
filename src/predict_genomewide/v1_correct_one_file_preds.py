import sys
assert len(sys.argv) == 7, sys.argv
cell_type, which_chromosome, which_fold, chunk_start, chunk_end, gpu  = sys.argv[1:]
which_fold = int(which_fold)
chunk_start = int(chunk_start)
chunk_end = int(chunk_end)

print("Cell type:", cell_type)
print("Chromsome:", which_chromosome)
print("Fold:", which_fold)
print("chunk_start:", chunk_start)
print("chunk_end:", chunk_end)


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



in_window = 2114
out_window = 1000

which_genome = "hg38"

fastas_by_genome = {"hg38" : "hg38/hg38.fasta"}

fasta_filepath = fastas_by_genome[which_genome]
chrom_sizes_filepath = which_genome + "/" + which_genome + ".chrom.sizes"


def load_chrom_sizes(chrom_sizes_filepath):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
    return chrom_sizes

chrom_sizes = load_chrom_sizes(chrom_sizes_filepath)



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
    
    
    

def get_model_path(fold, timestamps = timestamps, cell_type = cell_type):
    # fold input should be 0-indexed above
    config = FoldFilesConfig(cell_type, "strand_merged_umap", str(fold + 1),
                             timestamps[fold], "procap")
    return config.model_save_path

def load_model(model_save_path):
    model = torch.load(model_save_path)
    model = model.eval()
    return model


def read_chromosome_from_fasta(fasta_filepath, chrom):
    print("Loading genome sequence from " + fasta_filepath + " for " + chrom)
    fasta_index = Fasta(fasta_filepath)
    return fasta_index[chrom][:].seq.upper()



def get_contiguous_windows(list_of_ints):
    windows = []
    
    # start state: we know the first window starts at list_of_ints[0]
    curr_window_start = list_of_ints[0]
    # will just always store the num from the prev. iteration of loop
    prev_num = list_of_ints[0]
    
    for num in list_of_ints[1:]:
        # whenever we reach the end of a contiguous section, note and reset
        if not num - 1 == prev_num:
            # this window definition is inclusive on both ends
            # (like (5,5) means 5 was by itself, (5,6) means 5 and 6)
            windows.append((curr_window_start, prev_num))
            curr_window_start = num
            
        prev_num = num
        
    # the loop never does the last window for you, so add it on manually
    windows.append((curr_window_start, list_of_ints[-1]))
    
    return windows


def get_nonN_chromosome_chunks(chrom_seq_str, stride = 250, chunk_size = in_window):
    chr_len = len(chrom_seq_str)
    
    num_possible_chunks = chr_len // stride + 1
    # these go 0,1,2,... but will really mean positions 0, stride, 2 * stride, ...
    possible_chunk_indexes = np.arange(0, num_possible_chunks)
    
    # go through each chunk of the chromosome and check if the sequence is usable
    usable_chunk_indexes = []
    for chunk_index in possible_chunk_indexes:
        chunk_start = stride * chunk_index
        chunk_end = chunk_start + chunk_size
        seq_chunk = chrom_seq_str[chunk_start : chunk_end]
        
        # we only want sequences that are less than half Ns
        if not seq_chunk.count("N") > len(seq_chunk) // 2:
            usable_chunk_indexes.append(chunk_index)
        
    usable_chunk_windows_indexes = get_contiguous_windows(usable_chunk_indexes)
    
    # convert these "chunk indexes" to actual chrom positions
    usable_chunk_windows = []
    for window_start, window_end in usable_chunk_windows_indexes:
        usable_chunk_windows.append((stride * window_start, stride * window_end))
        
    return usable_chunk_windows


def make_ohe_seqs_for_chunk(chrom_seq_str, first_chunk_start, last_chunk_start,
                    stride = 250, chunk_size = in_window):
    
    # makes tensor of seqs of size [num_seqs, 4, chunk_size]
    # that can go directly into the model for prediction
    
    chrom_seq_whole_chunk = chrom_seq_str[first_chunk_start : last_chunk_start + chunk_size]
    
    chrom_seq_whole_chunk_ohe = one_hot_encode(chrom_seq_whole_chunk,
                                               ignore=["N", "R", "Y", "W", "S", "K", "M", "B", "D", "V", "H"])
    
    assert chrom_seq_whole_chunk_ohe.shape[0] == 4, chrom_seq_whole_chunk_ohe.shape
    target_chunk_len = last_chunk_start + chunk_size - first_chunk_start
    assert chrom_seq_whole_chunk_ohe.shape[1] == target_chunk_len, (chrom_seq_whole_chunk_ohe.shape,
                                                                    len(chrom_seq_whole_chunk))
    
    chunk_seqs_ohe = []
    
    # range starts at 0 bc we will index into just the whole-chunk seq above
    # (whereas "first_chunk_start" etc are positions w.r.t. the entire chromosome)
    for seq_start in np.arange(0, last_chunk_start - first_chunk_start + 1, stride):
        seq_end = seq_start + chunk_size
        chunk_seqs_ohe.append(chrom_seq_whole_chunk_ohe[:, seq_start:seq_end])
        
    return torch.stack(chunk_seqs_ohe)


def _model_predict(model, X, batch_size=256, logits = False):
    # modified so it can take massive X tensors in
    
    X = X.type(torch.float32)
    
    model = model.cuda()
    with torch.no_grad():
        starts = np.arange(0, X.shape[0], batch_size)
        ends = starts + batch_size

        y_profiles, y_counts = [], []
        for start, end in zip(starts, ends):
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
    with torch.no_grad():
        pred_profiles, pred_logcounts = _model_predict(model, onehot_seqs)
        rc_pred_profiles, rc_pred_logcounts = _model_predict(model, torch.flip(onehot_seqs, [-1, -2]))
    
    # reverse-complement (strand-flip) BOTH of the predictions
    rc_pred_profiles = rc_pred_profiles[:, ::-1, ::-1]
    rc_pred_logcounts = rc_pred_logcounts[:, ::-1]
    
    # take the average prediction across the fwd and RC sequences
    merged_pred_profiles = np.log(np.array([np.exp(pred_profiles), np.exp(rc_pred_profiles)]).mean(axis=0))
    merged_pred_logcounts = np.array([pred_logcounts, rc_pred_logcounts]).mean(axis=0)
    
    return merged_pred_profiles, merged_pred_logcounts


def get_raw_preds_save_dir(model_fold, cell_type = cell_type,
                              which_genome = which_genome,
                              which_chromosome = which_chromosome):
    
    return "raw_preds/" + which_genome + "/" + cell_type + "/" + which_chromosome + "/fold_" + str(model_fold) + "/"


def predict_one_fold(model_fold, chrom_seq, first_chunk_start, last_chunk_start):
    print("Predicting, model fold: ", model_fold)
    preds_out_dir = get_raw_preds_save_dir(model_fold)
    os.makedirs(preds_out_dir, exist_ok=True)
    
    model = load_model(get_model_path(model_fold))

    print("Chunk: ", first_chunk_start, last_chunk_start)

    save_prefix = preds_out_dir + "chunk_" + str(first_chunk_start) + "_" + str(last_chunk_start) + "_pred_"

    chunk_seqs_ohe = make_ohe_seqs_for_chunk(chrom_seq, first_chunk_start, last_chunk_start)

    pred_profiles, pred_logcounts = _model_predict_with_rc(model, chunk_seqs_ohe)

    np.save(save_prefix + "profiles.npy", pred_profiles)
    np.save(save_prefix + "logcounts.npy", pred_logcounts)

        




chrom_seq = read_chromosome_from_fasta(fasta_filepath, which_chromosome)

predict_one_fold(which_fold, chrom_seq, chunk_start, chunk_end)
    
print("Done!")


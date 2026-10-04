import numpy as np
import pandas as pd
import os
import torch
from collections import defaultdict

import sys
sys.path.append("../2_train_models")
from file_configs import MergedFilesConfig
from data_loading import read_fasta_fast

from puffin import *

# tangermeme!!! from Jacob, https://github.com/jmschrei/tangermeme
from tangermeme_puffin_compatible import saturation_mutagenesis
from tangermeme.utils import one_hot_encode as tangermeme_ohe



def extract_sequences_not_ohe(sequences, chrom_sizes, peak_path,
                              in_window, verbose=True):
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
    for _, (chrom, start, end) in peaks.iterrows():
        mid = start + (end - start) // 2
        s = mid - in_width
        e = mid + in_width
        assert s > 0, start

        seq = sequences[chrom][s:e]

        assert len(seq) == e - s, (len(seq), s, e)
        seqs.append(seq)

    return seqs

def puffin_predict_once(puffin_model, seq):
    raw_pred_df = puffin_model.predict(seq)
    # select for the PRO-cap + and - strands from all outputs;
    # since Puffin's outputs are in natural-log-plus-one scale,
    # also convert to raw (aggregate dataset) counts scale
    pred = np.exp(np.array([np.array(raw_pred_df)[6],
                     np.array(raw_pred_df)[-1]]).astype(float)) - 1
    return pred



# Load the config object for when model outputs were merged across all folds
# (just to pull some filepaths from)
cell_type = "K562"
model_type = "strand_merged_umap"
data_type = "procap"
merged_config = MergedFilesConfig(cell_type, model_type, data_type)
# these paths aren't specific to any model / fold, cell type, or data_type
genome_path = merged_config.genome_path
chrom_sizes = merged_config.chrom_sizes


# Load puffin model
puffin = Puffin(use_cuda=False)

# generate puffin prediction for example locus
locus_i = 20938

# puffin's API wants sequences as strings, not numpy arrays
    # and to make 1kb prediction, it requires 1650bp of input sequence
seq_notohe = extract_sequences_not_ohe(genome_path, chrom_sizes,
                                     merged_config.all_peak_path, in_window=1650)[locus_i]

puffin_pred_prof = puffin_predict_once(puffin, seq_notohe)

print("puffin_pred_prof.shape", puffin_pred_prof.shape)

np.save("pred_prof_puffin_example.npy", puffin_pred_prof)







def load_sequence_at_locus(coord, genome_path):
    chrom, start, end = coord
    
    # load (forward strand) genomic sequence between start and end coords
    genome = read_fasta_fast(genome_path, include_chroms = [chrom])
    # seq is a string
    seq = genome[chrom][start:end]
    
    # one-hot encode the sequence string (returns numpy array)
    onehot_seq = one_hot_encode(seq).T
    
    # after transposing, onehot_seq is shape (4, end - start)
    return seq, onehot_seq


def trim_seq_to_puffin_len(seq, puffin_len=1650):
    # puffin has a smaller "in_window" than ProCapNet (1650 instead of 2114),
    # so this function trims the flanks of sequences to fit puffin's size
    
    assert type(seq) == str, type(seq)
    
    to_trim_total = len(seq) - puffin_len
    assert to_trim_total % 2 == 0, to_trim_total
    
    to_trim_side = to_trim_total // 2
    
    trimmed = seq[to_trim_side : - to_trim_side]
    
    assert len(trimmed) == puffin_len, len(trimmed)
    return trimmed


def reverse_ohe(seq_onehot):
    # turn a sequence that's been onehot-encoded back into a string
    assert len(seq_onehot.shape) == 2 and seq_onehot.shape[0] == 4, seq_onehot.shape
    seq_onehot = seq_onehot.T
    
    ohe_to_str = defaultdict(lambda : "N")
    ohe_to_str[(1,0,0,0)] = "A"
    ohe_to_str[(0,1,0,0)] = "C"
    ohe_to_str[(0,0,1,0)] = "G"
    ohe_to_str[(0,0,0,1)] = "T"
    
    seq_str = "".join([ohe_to_str[tuple([int(num) for num in base])] for base in seq_onehot])
    return seq_str


class PuffinWrapperProcap(torch.nn.Module):
    # tangermeme needs a model-wrapper to do ISM with -- this one is for puffin
    
    def __init__(self, model):
        super(PuffinWrapperProcap, self).__init__()
        self.model = model

    def logits_to_a_number_deepshappy(self, logits):
        # more complicated way of collapsing a profile prediction to just one scalar;
        # I tried this and the easier way below, results were qualitatively the same
        logits = torch.Tensor(logits).reshape(-1)
        mean_norm_logits = logits - torch.mean(logits, axis = -1, keepdims = True)
        softmax_probs = torch.nn.Softmax(dim=-1)(mean_norm_logits.detach())
        final = (mean_norm_logits * softmax_probs).sum(axis=-1)
        return final
    
    def logits_to_a_number(self, logits):
        # simplest way of collapsing a profile prediction to just one scalar
        pred = np.exp(logits)
        final = pred.sum()
        return final
        
    def forward_one_seq(self, seq):
        # take in one sequence and return a scalar, for ISM
        
        # Puffin's API needs strings, not one-hot encodings, so convert back
        if type(seq) != str:
            seq = seq.squeeze()
            assert len(seq.shape) == 2, seq.shape
            seq = reverse_ohe(seq)

        # get just the logits for the PRO-cap + and - strand preds
        all_preds = self.model.predict(seq)
        pred = np.array([np.array(all_preds)[6],
                         np.array(all_preds)[-1]]).astype(float)
        
        # convert logits to a single scalar
        return self.logits_to_a_number(pred)
        
    def forward(self, X):
        # doesn't need to be differentiable if we just do ISM
        
        # puffin's API does one sequence at a time, so...
        model_outputs = []
        for seq in X:
            model_outputs.append(self.forward_one_seq(seq))
        return torch.Tensor(np.array(model_outputs))

    
def get_puffin_ism(seq_onehot, puffin_model):
    assert seq_onehot.shape[-2] == 4, seq_onehot.shape
    seq_len = seq_onehot.shape[-1]

    wrapper = PuffinWrapperProcap(puffin_model)
    
    # the start and end here are hard-coded for the example I wanted to plot
    # (they let you just do ISM on the bases you're going to plot)
    X_attr = saturation_mutagenesis(wrapper, seq_onehot.squeeze()[None,...],
                                    start = seq_len//2 - 100, end = seq_len//2 + 250,
                                    device='cpu', verbose=True, dtype=torch.bfloat16)
    return X_attr


seq_onehot = tangermeme_ohe(seq_notohe)

print(seq_onehot)

ism_out = get_puffin_ism(seq_onehot, puffin).squeeze()

print("ism_out.shape", ism_out.shape)

np.save("ism_puffin_example.npy", ism_out)

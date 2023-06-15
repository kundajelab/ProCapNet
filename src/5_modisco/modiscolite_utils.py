import sys
import numpy as np
import random

import modiscolite
import modiscolite.tfmodisco
import modiscolite.util
import viz_sequence

sys.path.append("../2_train_models")
from data_loading import extract_sequences, extract_observed_profiles
sys.path.append("../utils")
from misc import ensure_parent_dir_exists

random.seed(0)
np.random.seed(0)
  
    
def load_sequences(genome_path, chrom_sizes, peak_path, slice_len, in_window=2114):
    onehot_seqs = extract_sequences(genome_path, chrom_sizes, peak_path, in_window)
    
    in_width = in_window // 2
    slice_width = slice_len // 2
    
    onehot_seqs = onehot_seqs.swapaxes(1,2)
    onehot_seqs = onehot_seqs[:, (in_width - slice_width):(in_width + slice_width), :]
    assert onehot_seqs.shape[1] == slice_len and onehot_seqs.shape[2] == 4, onehot_seqs.shape

    return onehot_seqs


def load_observed_profiles(pos_bw_path, neg_bw_path, peak_path, slice_len, out_window=1000):
    profs = extract_observed_profiles(pos_bw_path, neg_bw_path, peak_path, out_window=out_window)
    
    out_width = out_window // 2
    slice_width = slice_len // 2
    
    profs = profs[..., (out_width - slice_width):(out_width + slice_width)]
    assert profs.shape[-1] == slice_len, profs.shape

    return profs    


def load_scores(scores_path, slice_len, in_window=2114):
    in_width = in_window // 2
    slice_width = slice_len // 2
    
    hyp_scores = np.load(scores_path).swapaxes(1,2)
    hyp_scores = hyp_scores[:, (in_width - slice_width):(in_width + slice_width), :]
    assert hyp_scores.shape[1] == slice_len and hyp_scores.shape[2] == 4, hyp_scores.shape

    return hyp_scores

    
def _run_modisco(onehot_seqs, scores, max_seqlets=1000000):
    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=scores, one_hot=onehot_seqs,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=20,
        flank_size=5,
        target_seqlet_fdr=0.05,
        n_leiden_runs=50)
    return pos_patterns, neg_patterns

    
def modisco(genome_path, chrom_sizes, peak_path, scores_path,
            slice_len, in_window, results_save_path, save=True):
    
    print("Running modisco(lite).\n")
    print("genome_path:", genome_path)
    print("chrom_sizes:", chrom_sizes)
    print("peak_path:", peak_path)
    print("scores_path:", scores_path)
    print("slice_len:", slice_len)
    print("in_window:", in_window, "\n")
    print("results_save_path:", results_save_path)

    
    
    onehot_seqs = load_sequences(genome_path, chrom_sizes, peak_path,
                                 slice_len, in_window=in_window)

    scores = load_scores(scores_path, slice_len, in_window=in_window)

    pos_patterns, neg_patterns = _run_modisco(onehot_seqs, scores)

    if save:
        ensure_parent_dir_exists(results_save_path)
        modiscolite.io.save_hdf5(results_save_path, pos_patterns, neg_patterns)
    else:
        return pos_patterns, neg_patterns
    

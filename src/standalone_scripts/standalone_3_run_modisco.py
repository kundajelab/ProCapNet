# set the filepaths below:


class FilePaths():
    # an object that holds all the hard-coded filepaths needed.
    # edit the variables below to point to your files

    def __init__(self):
        # filepath for fasta for reference genome
        self.genome_path = '/mnt/lab_data2/kcochran/procapnet/genomes/hg38.withrDNA.fasta'
        # filepath for text file with chromosome sizes for reference genome
        self.chrom_sizes = '/mnt/lab_data2/kcochran/procapnet/genomes/hg38.withrDNA.chrom.sizes'

        # filepath for PRO-cap peak regions you ran DeepSHAP on:
        # bed file with 3+ columns in format (chrom, region_start_coord, region_end_coord)
        self.all_peak_path = 'peaks_subset_for_testing.bed.gz'

        # filepath you saved deepshap scores to as a numpy array - not-one-hot, just one task at a time
        # these scores must have been created using the same peak file as self.all_peak_path here
        self.scores_path = 'profile_deepshap_scores.npy'
        
        # filepath to save modisco results to
        self.modisco_results_path = 'modisco_results.hdf5'


config = FilePaths()


in_window = 2114
out_window = 1000
slice_len = 1000


####################### Everything below here should just work #######################

# these should all be available in the conda environment:

from pyfaidx import Fasta

from tqdm import tqdm, trange
import numpy as np
import pandas as pd

import os
import sys
from collections import defaultdict
import gzip
import random

import h5py

import modiscolite
import modiscolite.tfmodisco
import modiscolite.util



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


def extract_sequences(sequences, chrom_sizes, peak_path, in_window=in_window, verbose=False):
    seqs = []
    in_width = in_window // 2

    if isinstance(sequences, str):
        assert os.path.exists(sequences), sequences
        sequences = read_fasta(sequences, chrom_sizes, verbose=verbose)

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





### Modisco stuff
  
    
def load_sequences(genome_path, chrom_sizes, peak_path, slice_len=slice_len, in_window=in_window):
    onehot_seqs = extract_sequences(genome_path, chrom_sizes, peak_path, in_window)
    
    in_width = in_window // 2
    slice_width = slice_len // 2
    
    onehot_seqs = onehot_seqs.swapaxes(1,2)
    onehot_seqs = onehot_seqs[:, (in_width - slice_width):(in_width + slice_width), :]
    assert onehot_seqs.shape[1] == slice_len and onehot_seqs.shape[2] == 4, onehot_seqs.shape

    return onehot_seqs    


def load_scores(scores_path, slice_len=slice_len, in_window=in_window):
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
            results_save_path, slice_len=slice_len, in_window=in_window, save=True):
    
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
        modiscolite.io.save_hdf5(results_save_path, pos_patterns, neg_patterns, slice_len)
    else:
        return pos_patterns, neg_patterns
    

def load_modisco_results(tfm_results_path):
    return h5py.File(tfm_results_path, "r")








print("Running modisco...")

modisco(config.genome_path,
        config.chrom_sizes,
        config.all_peak_path,
        config.scores_path,
        config.modisco_results_path,
        slice_len,
        in_window)

print("Done running modisco.")
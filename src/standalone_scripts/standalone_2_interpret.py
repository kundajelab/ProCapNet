# set the GPU_ID and filepaths below:

# GPU_ID is the NVIDIA string identifier for the GPU on your machine
GPU_ID = "MIG-f80e9374-504a-571b-bac0-6fb00750db4c"


class FilePaths():
    # an object that holds all the hard-coded filepaths needed.
    # edit the variables below to point to your files

    def __init__(self):
        # filepath for where the trained model is saved
        self.model_save_path = "test_05312025.model"

        # filepath for fasta for reference genome
        self.genome_path = '/mnt/lab_data2/kcochran/procapnet/genomes/hg38.withrDNA.fasta'
        # filepath for text file with chromosome sizes for reference genome
        self.chrom_sizes = '/mnt/lab_data2/kcochran/procapnet/genomes/hg38.withrDNA.chrom.sizes'

        # filepath for PRO-cap peak regions to apply DeepSHAP to:
        # bed file with 3+ columns in format (chrom, region_start_coord, region_end_coord)
        self.all_peak_path = 'peaks_subset_for_testing.bed.gz'

        # filepaths to save deepshap scores to (as numpy arrays):
        # both tasks will save scores both as one-hot (zeros except for 1 base at each position) and not-one-hot;
        # they will also save to bigwigs with the same names but different extensions
        self.profile_scores_path = 'profile_deepshap_scores.npy'
        self.profile_onehot_scores_path = 'profile_deepshap_scores_onehot.npy'
        self.counts_scores_path = 'counts_deepshap_scores.npy'
        self.counts_onehot_scores_path = 'counts_deepshap_scores_onehot.npy'
        


config = FilePaths()


in_window = 2114
out_window = 1000



####################### Everything below here should just work #######################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# these should all be available in the conda environment:
import torch

import pyBigWig
from pyfaidx import Fasta

from tqdm import tqdm, trange
import numpy as np
import pandas as pd


import sys
from collections import defaultdict
import gzip

from captum.attr import DeepLiftShap


class Model(torch.nn.Module):

    def __init__(self, model_save_path,
                 n_filters = 512,
                 n_layers = 8,
                 n_outputs = 1,  # whether the model is stranded or not stranded
                 alpha = 1,
                 trimming = (2114 - 1000) // 2):

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





##### DeepSHAP Stuff


class ProfileModelWrapper(torch.nn.Module):
    # this wrapper assumes:
    # 1) the model's profile head outputs pre-softmax logits
    # 2) the profile output has the last axis as the profile-length dimension
    # 3) the softmax should be applied over both strands at the same time
    #      (a ala Jacob's bpnetlite implementation of BPNet)
    # 4) the profile head is the first of two model outputs

    def __init__(self, model):
        super(ProfileModelWrapper, self).__init__()
        self.model = model

    def forward(self, X):
        logits, _ = self.model(X)
        logits = logits.reshape(logits.shape[0], -1)
        mean_norm_logits = logits - torch.mean(logits, axis = -1, keepdims = True)
        softmax_probs = torch.nn.Softmax(dim=-1)(mean_norm_logits.detach())
        return (mean_norm_logits * softmax_probs).sum(axis=-1)


class CountsModelWrapper(torch.nn.Module):
    # this wrapper assumes the counts head is the second of two model outputs

    def __init__(self, model):
        super(CountsModelWrapper, self).__init__()
        self.model = model

    def forward(self, X):
        _, logcounts = self.model(X)
        return logcounts




# Code borrowed, modified from Jacob Schreiber
# https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/attributions.py

import numba

@numba.jit('void(int64, int64[:], int64[:], int32[:, :], int32[:,], int32[:, :], float32[:, :, :])')
def _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, counters, shuffled_sequences):
    """An internal function for fast shuffling using numba."""

    for i in range(n_shuffles):
        for char in chars:
            n = next_idxs_counts[char]

            next_idxs_ = np.arange(n)
            next_idxs_[:-1] = np.random.permutation(n-1)  # Keep last index same
            next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

        idx = 0
        shuffled_sequences[i, idxs[idx], 0] = 1
        for j in range(1, len(idxs)):
            char = idxs[idx]
            count = counters[i, char]
            idx = next_idxs[char, count]

            counters[i, char] += 1
            shuffled_sequences[i, idxs[idx], j] = 1


def dinuc_shuffle(sequence, n_shuffles=25, random_state=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    chars, idxs = torch.unique(sequence.argmax(axis=0), return_inverse=True)
    chars, idxs = chars.numpy(), idxs.numpy()

    next_idxs = np.zeros((len(chars), sequence.shape[1]), dtype=np.int32)
    next_idxs_counts = np.zeros(max(chars)+1, dtype=np.int32)

    for char in chars:
        next_idxs_ = np.where(idxs[:-1] == char)[0]
        n = len(next_idxs_)

        next_idxs[char][:n] = next_idxs_ + 1
        next_idxs_counts[char] = n

    shuffled_sequences = np.zeros((n_shuffles, *sequence.shape), dtype=np.float32)
    counters = np.zeros((n_shuffles, len(chars)), dtype=np.int32)

    _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, 
        counters, shuffled_sequences)

    shuffled_sequences = torch.from_numpy(shuffled_sequences)
    return shuffled_sequences





'''
Code here is based very loosely off of Surag's script:
https://github.com/kundajelab/surag-scripts/blob/master/bpnet-pipeline/importance/importance_hdf5_to_bigwig.py

'''

def load_chrom_sizes(chrom_sizes_filepath):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
    return chrom_sizes


def make_track_values_dict(all_values, coords, chrom):
    track_values = defaultdict(lambda : [])

    for values, (coord_chrom, start, end) in zip(all_values, coords):
        # subset to only peaks/values for the chromosome we're looking at now
        if coord_chrom == chrom:
            assert values.shape[0] == end - start, (values.shape, start, end, end - start)
            
            for position, value in enumerate(values):
                position_offset = position + start
                track_values[position_offset] = track_values[position_offset] + [value]
    
    # take the mean at each position, so that if there was ovelap, the average value is used
    track_values = { key : sum(vals) / len(vals) for key, vals in track_values.items() }
    return track_values


def load_coords(peaks_file, window_len):
    lines = []
    if peaks_file.endswith(".gz"):
        with gzip.open(peaks_file) as f:
            lines = [line.decode().split()[:3] for line in f]
    else:
        with open(peaks_file) as f:
            lines = [line.split()[:3] for line in f]
            
    coords = []
    for line in lines:
        chrom, start, end = line[0], int(line[1]), int(line[2])
        mid = start + (end - start) // 2
        coord = (chrom, mid - window_len // 2, mid + window_len // 2)
        coords.append(coord)
    return coords
        
        
def write_scores_to_bigwigs(scores,
                            peaks_file,
                            save_filepath,
                            chrom_sizes_filepath):
    
    scores = scores.astype("float64")  # pybigwig will throw error if you don't do this
    assert len(scores.shape) == 2, scores.shape  # should be one-hot and flattened
    window_len = scores.shape[-1]  # assuming last axis is profile len axis
    
    coords = load_coords(peaks_file, window_len)
    assert len(coords) == scores.shape[0]
    
    save_filepath = save_filepath.replace(".npy", "")
    if not (save_filepath.endswith(".bigWig") or save_filepath.endswith(".bw")):
        bw_filename = save_filepath + ".bigWig"
    else:
        bw_filename = save_filepath
          
    print("Writing attributions to bigwig.")
    print("Peaks: " + peaks_file)
    print("Save path: " + bw_filename)
    print("Chrom sizes file: " + chrom_sizes_filepath)
    print("Window length: " + str(window_len))

    # bigwigs need to be written in order -- so we have to go chromosome by chromosome,
    # and the chromosomes need to be numerically sorted (i.e. chr9 then chr10)
    chrom_names = load_chrom_names(chrom_sizes_filepath)
    chrom_order = {chrom : i for i, chrom in enumerate(chrom_names)}
    chromosomes = sorted(list({coord[0] for coord in coords}),
                         key = lambda chrom_str : chrom_order[chrom_str])

    bw = pyBigWig.open(bw_filename, 'w')
    # bigwigs need headers before they can be written to
    # the header is just the info you'd find in a chrom.sizes file
    bw.addHeader(load_chrom_sizes(chrom_sizes_filepath))

    for chrom in chromosomes:
        # convert arrays of scores for each peak into dict of base position : score
        # this function will average together scores at the same position from different called peaks
        track_values_dict = make_track_values_dict(scores, coords, chrom)
        num_entries = len(track_values_dict)

        starts = sorted(list(track_values_dict.keys()))
        ends = [position + 1 for position in starts]
        scores_to_write = [track_values_dict[key] for key in starts]

        assert len(scores_to_write) == len(starts) and len(scores_to_write) == len(ends)

        bw.addEntries([chrom for _ in range(num_entries)], 
                       starts, ends = ends, values = scores_to_write)

    bw.close()

    
    
    
    
    
    




def get_attributions(sequences, model, num_shufs = 25):
    assert len(sequences.shape) == 3 and sequences.shape[1] == 4, sequences.shape
    prof_attrs = []
    count_attrs = []

    with torch.no_grad():
        for i in trange(len(sequences)):
            prof_explainer = DeepLiftShap(ProfileModelWrapper(model))
            count_explainer = DeepLiftShap(CountsModelWrapper(model))

            # use a batch of 1 so that reference is generated for each seq 
            seq = torch.tensor(sequences[i : i + 1]).float()

            # create a reference of dinucleotide shuffled sequences
            ref_seqs = dinuc_shuffle(seq[0], num_shufs).float().cuda()

            seq = seq.cuda()
            # calculate attributions according to profile task (fwd and rev strands)
            prof_attrs_fwd = prof_explainer.attribute(seq, ref_seqs).cpu()
            prof_attrs_rev = prof_explainer.attribute(torch.flip(seq, [1,2]),
                                                      torch.flip(ref_seqs, [1,2])).cpu()

            prof_attrs_rev = torch.flip(prof_attrs_rev, [1,2])

            prof_attrs_batch = np.array([prof_attrs_fwd.numpy(), prof_attrs_rev.numpy()])
            prof_attrs.append(prof_attrs_batch.mean(axis=0))

            # calculate attributions according to counts task (fwd and rev strands)

            count_attrs_fwd = count_explainer.attribute(seq, ref_seqs).cpu()
            count_attrs_rev = count_explainer.attribute(torch.flip(seq, [1,2]),
                                                        torch.flip(ref_seqs, [1,2])).cpu()

            count_attrs_rev = torch.flip(count_attrs_rev, [1,2])

            count_attrs_batch = np.array([count_attrs_fwd.numpy(), count_attrs_rev.numpy()])
            count_attrs.append(count_attrs_batch.mean(axis=0))

    prof_attrs = np.concatenate(prof_attrs)
    count_attrs = np.concatenate(count_attrs)
    return prof_attrs, count_attrs











def save_deepshap_results(onehot_seqs, scores, peak_path,
                          scores_path, onehot_scores_path,
                          chrom_sizes):
    assert len(onehot_seqs.shape) == 3 and onehot_seqs.shape[1] == 4, onehot_seqs.shape
    assert len(scores.shape) == 3 and scores.shape[1] == 4, scores.shape

    # save profile attributions
    scores_onehot = scores * onehot_seqs
    np.save(scores_path, scores)
    np.save(onehot_scores_path, scores_onehot)

    # write scores to bigwigs -- flatten the one-hot encoding of scores
    write_scores_to_bigwigs(np.sum(scores_onehot, axis = 1),
                            peak_path, scores_path, chrom_sizes)




def run_deepshap(genome_path, chrom_sizes, peak_path, model_path,
                 prof_scores_path, prof_onehot_scores_path,
                 count_scores_path, count_onehot_scores_path,
                 in_window=in_window, out_window=out_window, save=True):

    print("Running deepSHAP.\n")
    print("genome_path:", genome_path)
    print("chrom_sizes:", chrom_sizes)
    print("peak_path:", peak_path)
    print("model_path:", model_path)

    print("prof_scores_path:", prof_scores_path)
    print("prof_onehot_scores_path:", prof_onehot_scores_path)
    print("count_scores_path:", count_scores_path)
    print("count_onehot_scores_path:", count_onehot_scores_path)

    print("in_window:", in_window)
    print("out_window:", out_window, "\n")


    onehot_seqs = extract_sequences(genome_path, chrom_sizes,
                                    peak_path,
                                    in_window=in_window,
                                    verbose=True)

    model = torch.load(model_path, weights_only=False)
    model.eval()
    model = model.cuda()


    prof_attrs, count_attrs = get_attributions(onehot_seqs, model)

    if save:
        save_deepshap_results(onehot_seqs, prof_attrs, peak_path,
                              prof_scores_path, prof_onehot_scores_path,
                              chrom_sizes)

        save_deepshap_results(onehot_seqs, count_attrs, peak_path,
                              count_scores_path, count_onehot_scores_path,
                              chrom_sizes)
    else:
        return prof_attrs, count_attrs







print("Running deepshap...")

run_deepshap(config.genome_path,
             config.chrom_sizes,
             config.all_peak_path,
             config.model_save_path,
             config.profile_scores_path,
             config.profile_onehot_scores_path,
             config.counts_scores_path,
             config.counts_onehot_scores_path,
             in_window=in_window,
             out_window=out_window)

print("Done running deepshap.")
from puffin import *

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os

import sys
sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig, MergedFilesConfig
from data_loading import read_fasta_fast


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
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        s = mid - in_width
        e = mid + in_width
        assert s > 0, start

        seq = sequences[chrom][s:e]

        assert len(seq) == e - s, (len(seq), s, e)
        seqs.append(seq)

    return seqs


def puffin_predict(genome_path, chrom_sizes, puffin_model, peak_path):
    # puffin's API wants sequences as strings, not numpy arrays
    # and to make 1kb prediction, it requires 1650bp of input sequence
    seqs = extract_sequences_not_ohe(genome_path, chrom_sizes,
                                     peak_path, in_window=1650)
    preds = []
    for seq in tqdm(seqs):
        raw_pred_df = puffin_model.predict(seq)
        # select for the PRO-cap + and - strands from all outputs;
        # since Puffin's outputs are in natural-log-plus-one scale,
        # also convert to raw (aggregate dataset) counts scale
        pred = np.exp(np.array([np.array(raw_pred_df)[6],
                         np.array(raw_pred_df)[-1]]).astype(float)) - 1
        preds.append(pred)
        
    return np.array(preds) 




cell_type = "K562"
model_type = "strand_merged_umap"
data_type = "procap"
merged_config = MergedFilesConfig(cell_type, model_type, data_type)
genome_path = merged_config.genome_path
chrom_sizes = merged_config.chrom_sizes
puffin_test_set_bed = "puffin_data_train_val_test_split/test_set_from_ksenia.bed.gz"


puffin_model = Puffin(use_cuda=False)
preds = puffin_predict(genome_path, chrom_sizes, puffin_model, puffin_test_set_bed)

np.save("pred_profs_puffin_puffintestset.npy", preds)

print("Done")

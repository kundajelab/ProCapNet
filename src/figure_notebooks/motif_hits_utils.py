import os, sys
sys.path.append("../2_train_models")
from file_configs import MergedFilesConfig
from common_functions import load_coords

import numpy as np
import pandas as pd
import gzip



def import_motif_hits(motif_hits_path):
    return pd.read_csv(motif_hits_path, sep="\t", header=None, index_col=False,
        names=["chrom", "start", "end", "motif", "peak_index", "strand", "motif_index"])


def get_peak_hits(peak_table, hit_table):
    """
    For each peak, extracts the set of motif hits that fall in that peak.
    Returns a list mapping peak index to a subtable of `hit_table`. The index
    of the list is the index of the peak table.
    
    Note that this function is updated relative to older versions of itself,
    to match the format of the hits table.
    """
    peak_hits = [pd.DataFrame(columns=list(hit_table))] * len(peak_table)

    peak_indexes = peak_table["index"].values
    actual_indexes = list(peak_table.index)
    peak_index_to_actual_index = dict(zip(peak_indexes, actual_indexes))

    for peak_index, matches in hit_table.groupby("peak_index"):
        if peak_index in peak_index_to_actual_index:
            peak_hits[peak_index_to_actual_index[peak_index]] = matches
    return peak_hits


def get_peak_motif_counts(peak_hits, motif_keys):
    """
    From the peak hits (as returned by `get_peak_hits`), computes a count
    array of size N x M, where N is the number of peaks and M is the number of
    motifs. Each entry represents the number of times a motif appears in a peak.
    `motif_keys` is a list of motif keys as they appear in `peak_hits`; the
    order of the motifs M matches this list.
    """
    motif_inds = {motif_keys[i] : i for i in range(len(motif_keys))}
    counts = np.zeros((len(peak_hits), len(motif_keys)), dtype=int)
    for i in range(len(peak_hits)):
        hits = peak_hits[i]
        for key, num in zip(*np.unique(hits["motif_index"], return_counts=True)):
            counts[i][motif_inds[key]] = num
    return counts


def make_peak_table(peak_path, in_window):
    coords = load_coords(peak_path, in_window)

    peak_table = pd.DataFrame(coords, columns = ["peak_chrom", "peak_start", "peak_end"])
    peak_table["summit_offset"] = in_window // 2
    peak_table = peak_table.reset_index()
    
    return peak_table


def load_motif_hits(cell_type, model_type, data_type, in_window):
    hits = dict()
    peak_hits = dict()
    peak_hit_counts = dict()
    
    config = MergedFilesConfig(cell_type, model_type, data_type)
    all_peak_path = config.all_peak_path

    hits["profile"] = import_motif_hits(config.profile_hits_path)
    hits["counts"] = import_motif_hits(config.counts_hits_path)

    peak_table = make_peak_table(all_peak_path, in_window)
    
    for task in hits.keys():
        peak_hits[task] = get_peak_hits(peak_table, hits[task])
        motif_keys = list(set(hits[task]["motif_index"]))
        peak_hit_counts[task] = get_peak_motif_counts(peak_hits[task], motif_keys)

    return hits, peak_hits, peak_hit_counts


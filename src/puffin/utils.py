import numpy as np
import pandas as pd
import os
import torch
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

from tqdm import tqdm

import matplotlib.pyplot as plt

import sys
sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig, MergedFilesConfig
from data_loading import extract_peaks, extract_observed_profiles, extract_sequences
from data_loading import read_fasta_fast

sys.path.append("../figure_notebooks")
from load_annotations_utils import load_coords_with_summits
from common_functions import load_coords


def load_fold_configs(cell_type, timestamps,
                      model_type = "strand_merged_umap", data_type = "procap"):
    
    # Load the config objects (filepaths holders) for each fold a model was trained on

    fold_configs = []
    for fold_i, timestamp in enumerate(timestamps):
        # folds are 1-indexed, and config constructor is expecting a string
        fold = str(fold_i + 1)
        
        # load the config object for this specific fold / model
        config = FoldFilesConfig(cell_type, model_type, fold, timestamp, data_type)
        fold_configs.append(config)
        
    return fold_configs

def get_pseudorep_filepaths(pseudorep, pos_or_neg, data_dir):
    # Get paths to bigwigs for the pseudoreplicates of an experiment
    #  - pseudorep should be either an int (1-indexed) or an integer string
    
    assert pos_or_neg in ["pos", "neg"], pos_or_neg
    return os.path.join(data_dir, "pseudorep" + str(pseudorep) + "." + pos_or_neg + ".bigWig")


def load_pseudoreplicate_profiles(pseudorep, peaks_path, data_dir, out_window=1000):
    pos_bw_path = get_pseudorep_filepaths(pseudorep, "pos", data_dir)
    neg_bw_path = get_pseudorep_filepaths(pseudorep, "neg", data_dir)
    
    profs = extract_observed_profiles(pos_bw_path, neg_bw_path, peaks_path,
                                      out_window=out_window, verbose=True)
    return profs


def get_sort_order_test_sets(merged_config, timestamps, in_window=2114, cell_type="K562"):
    # Each model / fold has a mutually exclusive test set; the order of the examples
    # in each test set differs from the order in the merged file of all examples.
    # So, this function figures out how to reorder the test set model predictions
    # so that they can be compared to the observed data.
    
    # First, load in the test set coordinates for each fold's test set
    fold_configs = load_fold_configs(cell_type, timestamps)
    
    test_coords = []
    for config in fold_configs:
        test_coords.extend(load_coords(config.test_peak_path, in_window))
        
    # Second, load in the test set coordinates for the merged file
    # (the "correct order")
    
    all_coords = load_coords(merged_config.all_peak_path, in_window)
    
    # Third, figure out the re-ordering needed to arrange the test coords
    # so they match the ordering in the merged file.
    # (then use that ordering later to fix the order of model predictions)
    
    sort_order = [test_coords.index(coord) for coord in all_coords]
    assert np.all(np.array(all_coords) == np.array(test_coords)[sort_order])
    
    return sort_order


def load_procapnet_test_data(merged_config, timestamps, cell_type="K562", out_window=1000):
    true_profs = []
    pseudorep1_profs = []
    pseudorep2_profs = []
    log_pred_profs = []
    pred_logcounts = []
    
    fold_configs = load_fold_configs(cell_type, timestamps)
    
    for config in fold_configs:
        # Load observed data: replicate-merged PRO-cap signal
        obs_profs = extract_observed_profiles(config.plus_bw_path,
                                              config.minus_bw_path,
                                              config.test_peak_path,
                                              out_window=out_window,
                                              verbose=False)
        true_profs.extend(obs_profs)

        # Load PRO-cap signal for each pseudoreplicate individually (to see reproducibility)
        pseudorep1_profs.extend(load_pseudoreplicate_profiles(1, config.test_peak_path, config.data_dir))
        pseudorep2_profs.extend(load_pseudoreplicate_profiles(2, config.test_peak_path, config.data_dir))

        # Load model predictions
        log_pred_profs.extend(np.load(config.pred_profiles_test_path))
        pred_logcounts.extend(np.load(config.pred_logcounts_test_path).squeeze())
        
        # everything above should have loaded in the same number of examples/loci
        
        assert len(pseudorep1_profs) == len(pseudorep2_profs)
        assert len(pseudorep1_profs) == len(true_profs)
        assert len(log_pred_profs) == len(pred_logcounts)
        assert len(pred_logcounts) == len(true_profs)
        
    sort_order = get_sort_order_test_sets(merged_config, timestamps)
    
    true_profs = np.array(true_profs)[sort_order]
    pseudorep1_profs = np.array(pseudorep1_profs)[sort_order]
    pseudorep2_profs = np.array(pseudorep2_profs)[sort_order]
    log_pred_profs = np.array(log_pred_profs)[sort_order]
    pred_logcounts = np.array(pred_logcounts)[sort_order]
    
    pred_profs = np.exp(log_pred_profs)
    
    return true_profs, pseudorep1_profs, pseudorep2_profs, pred_profs, pred_logcounts



def get_fold_label(chrom):
    # returns *0-indexed* fold num that this chromosome was in the test set for
    
    # Fold assignment (below) needs to match what was in
    # 1_process_data/_split_peaks_train_val_test.py
    FOLDS = [["chr1", "chr4"],
         ["chr2", "chr13", "chr16"],
         ["chr5", "chr6", "chr20", "chr21"],
         ["chr7", "chr8", "chr9"],
         ["chr10", "chr11", "chr12"],
         ["chr3", "chr14", "chr15", "chr17"],
         ["chr18", "chr19", "chr22", "chrX", "chrY"]]
    
    for fold_i, fold_chroms in enumerate(FOLDS):
            if chrom in fold_chroms:
                return fold_i
    return -1


def make_fold_labels(merged_config, in_window=2114):
    # To stratify performance across folds, we need to label
    # each test example with which fold it was in the test set for.
    
    # Make list of len [num_peaks] / length of all_peak_path file,
    # where entry i is the fold that example i was in the test set for.
    
    coords = load_coords_with_summits(merged_config.all_peak_path, in_window)
    
    # coord[0] is the chromosome (first column from bed file)
    fold_labels = [get_fold_label(coord[0]) for coord in coords]
    
    # check we assigned a fold for every single example/peak
    assert all([label > -1 for label in fold_labels]), fold_labels
    return fold_labels


def get_avg_train_obs_profile_over_folds(true_profs, fold_labels):
    # For each fold, calculate the average PRO-cap profile across all examples
    # in the training or validation (not-test) sets (to use as a baseline).
    # Then, build a list the same length as fold_labels, where the ith entry
    # is the average train-val profile for the fold that example i belonged to.
    
    folds = sorted(list(set(fold_labels)))
    
    avg_profiles = []
    for fold in folds:
        # convert numeric fold labels to booleans
        in_train_val_fold = np.array([fold_label != fold for fold_label in fold_labels])
        
        # subset to profiles that were in the train-val sets for this fold
        true_profs_fold = true_profs[in_train_val_fold]
        
        # calculate mean per-base, keeping strands separate
        avg_profiles.append(np.mean(true_profs_fold, axis=0))
        
    avg_profiles_tiled = []
    for fold_label in fold_labels:
        avg_profiles_tiled.append(avg_profiles[fold_label])
    return np.array(avg_profiles_tiled)


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


def normalize_profile_metric(metrics_pred_vs_obs, metrics_upper_bound, metrics_lower_bound,
                             bigger_is_worse = False):
    assert metrics_pred_vs_obs.shape == metrics_upper_bound.shape
    assert metrics_upper_bound.shape == metrics_lower_bound.shape
    
    # For each locus/peak/example, min-max normalize the performance metric
    # using the replicate performance as the upper bound and the "average profile"
    # baseline as the lower bound.
    
    norm_metrics = (metrics_pred_vs_obs - metrics_lower_bound) / (metrics_upper_bound - metrics_lower_bound)
    
    # If bigger values for the original metric mean worse performance,
    # we want the normalized metric to follow the same pattern,
    # so we need to flip the direction of the norm metric 
    if bigger_is_worse:
        norm_metrics = 1 - norm_metrics
    
    norm_metrics = np.clip(norm_metrics, 0, 1)
    return norm_metrics


def stratify_profile_metrics_over_folds(prof_metrics, fold_labels):
    folds = sorted(list(set(fold_labels)))
    
    aggregated_metric_per_fold = []
    for fold in folds:
        # convert numeric fold labels to booleans
        in_fold = np.array([fold_num == fold for fold_num in fold_labels])
        
        prof_metrics_fold = prof_metrics[in_fold]
        
        aggregated_metric_per_fold.append(np.mean(prof_metrics_fold))
        
    print("Metric Average Within Each Fold's Test Set:")
    print(aggregated_metric_per_fold)
    
    print("\nMetric Average, Averaged Across Folds:")
    print(np.mean(aggregated_metric_per_fold))
    
    print("\nStandard Deviation of Metric Average Across Folds:")
    print(np.std(aggregated_metric_per_fold), "\n")
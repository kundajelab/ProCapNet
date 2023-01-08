import os, sys

import numpy as np
import pandas as pd
import h5py
import gzip

import sklearn.cluster
import scipy.cluster.hierarchy
import scipy.stats

import pomegranate

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import viz_sequence

import vdom.helpers as vdomh
from IPython.display import display
import tqdm

from file_configs import ModiscoFilesConfig, ValFilesConfig

BACKGROUND_FREQS = np.array([0.25, 0.25, 0.25, 0.25])


def import_motif_hits(motif_hits_path):
    """
    Imports the TF-MoDISco hits as a single Pandas DataFrame.
    The `key` column is the name of the originating PFM, and `peak_index` is the
    index of the peak file from which it was originally found.
    """
    with open(motif_hits_path, "r") as f:
        cols = next(f).split("\t")
    assert len(cols) == 14
    
    hit_table = pd.read_csv(
        motif_hits_path, sep="\t", header=None, index_col=False,
        names=[
            "chrom", "start", "end", "key", "strand", "peak_index",
            "imp_total_score", "imp_frac_score", "agg_sim", "mod_delta",
            "mod_precision", "mod_percentile", "fann_perclasssum_perc",
            "fann_perclassavg_perc"
        ]
    )

    # Sort by aggregate similarity and drop duplicates (by strand)
    hit_table = hit_table.sort_values("agg_sim")
    hit_table = hit_table.drop_duplicates(["chrom", "start", "end", "peak_index"], keep="last")
    return hit_table


def estimate_mode(x_values, bins=200, levels=1):
    """
    Estimates the mode of the distribution using `levels`
    iterations of histograms.
    """
    hist, edges = np.histogram(x_values, bins=bins)
    bin_mode = np.argmax(hist)
    left_edge, right_edge = edges[bin_mode], edges[bin_mode + 1]
    if levels <= 1:
        return (left_edge + right_edge) / 2
    else:
        return estimate_mode(
            x_values[(x_values >= left_edge) & (x_values < right_edge)],
            bins=bins,
            levels=(levels - 1)
        )
    
    
def fit_tight_exponential_dist(x_values, mode=0, percentiles=np.arange(0.05, 1, 0.05)):
    """
    Given an array of x-values and a set of percentiles of the distribution,
    computes the set of lambda values for an exponential distribution if the
    distribution were fit to each percentile of the x-values. Returns an array
    of lambda values parallel to `percentiles`. The exponential distribution
    is assumed to have the given mean/mode, and all data less than this mode
    is tossed out when doing this computation.
    """
    assert np.min(percentiles) >= 0 and np.max(percentiles) <= 1
    x_values = x_values[x_values >= mode]
    per_x_vals = np.percentile(x_values, percentiles * 100)
    return -np.log(1 - percentiles) / (per_x_vals - mode)

def exponential_pdf(x_values, lamb):
    return lamb * np.exp(-lamb * x_values)

def exponential_cdf(x_values, lamb):
    return 1 - np.exp(-lamb * x_values)


def get_peak_hits(peak_table, hit_table):
    """
    For each peak, extracts the set of motif hits that fall in that peak.
    Returns a list mapping peak index to a subtable of `hit_table`. The index
    of the list is the index of the peak table.
    """
    peak_hits = [pd.DataFrame(columns=list(hit_table))] * len(peak_table)
    for peak_index, matches in tqdm.notebook.tqdm(hit_table.groupby("peak_index")):
        # Check that all of the matches are indeed overlapping the peak
        peak_row = peak_table.iloc[peak_index]
        chrom, start, end = peak_row["chrom"], peak_row["peak_start"], peak_row["peak_end"]
        assert np.all(matches["chrom"] == chrom)
        assert np.all((matches["start"] < end) & (start < matches["end"]))
        
        peak_hits[peak_index] = matches
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
        for key, num in zip(*np.unique(hits["key"], return_counts=True)):
            counts[i][motif_inds[key]] = num
    return counts


def filter_peak_hits_by_fdr(hit_table, score_column="fann_perclasssum_perc", fdr_cutoff=0.05, plot=False):
    """
    Filters the table of peak hits by the score defined by
    `score_column` by fitting a mixture model to the score
    distribution, taking the exponential component, and then fitting a
    percentile-tightened exponential distribution to this component.
    p-values are computed using this null, and then the FDR-cutoff is applied
    using Benjamini-Hochberg.
    Returns a reduced hit table of the same format, and a tuple of figures for
    the score distribution and the FDR cutoffs.
    """
    scores = hit_table[score_column].values
    scores_finite = scores[np.isfinite(scores)]
    
    mode = estimate_mode(scores_finite)

    # Fit mixture of models to scores (mode-shifted)
    over_mode_scores = scores_finite[scores_finite >= mode] - mode
    mixed_model = pomegranate.GeneralMixtureModel.from_samples(
        [
            pomegranate.ExponentialDistribution,
            pomegranate.NormalDistribution,
            pomegranate.NormalDistribution
        ],
        3, over_mode_scores[:, None]
    )
    mixed_model = mixed_model.fit(over_mode_scores)
    mixed_model_exp_dist = mixed_model.distributions[0]
    
    # Obtain a distribution of scores that belong to the exponential distribution
    exp_scores = over_mode_scores[mixed_model.predict(over_mode_scores[:, None]) == 0]
    
    # Fit a tight exponential distribution based on percentiles
    lamb = np.max(fit_tight_exponential_dist(exp_scores))
    
    if plot:
        # Plot score distribution and fit
        score_fig, ax = plt.subplots(nrows=3, figsize=(20, 20))

        x = np.linspace(np.min(scores_finite), np.max(scores_finite), 200)[1:]  # Skip first bucket (it's usually very large
        mix_dist_pdf = mixed_model.probability(x)
        mixed_model_exp_dist_pdf = mixed_model_exp_dist.probability(x)

        perc_dist_pdf = exponential_pdf(x, lamb)
        perc_dist_cdf = exponential_cdf(x, lamb)

        # Plot mixed model
        ax[0].hist(over_mode_scores + mode, bins=500, density=True, alpha=0.3)
        ax[0].axvline(mode)
        ax[0].plot(x + mode, mix_dist_pdf, label="Mixed model")
        ax[0].plot(x + mode, mixed_model_exp_dist_pdf, label="Exponential component")
        ax[0].legend()

        # Plot fitted PDF
        ax[1].hist(exp_scores, bins=500, density=True, alpha=0.3)
        ax[1].plot(x + mode, perc_dist_pdf, label="Percentile-fitted")

        # Plot fitted CDF
        ax[2].hist(exp_scores, bins=500, density=True, alpha=1, cumulative=True, histtype="step")
        ax[2].plot(x + mode, perc_dist_cdf, label="Percentile-fitted")

        ax[0].set_title("Motif hit scores")
        plt.show()
    
    # Compute p-values
    score_range = np.linspace(np.min(scores_finite), np.max(scores_finite), 1000000)
    inverse_cdf = 1 - exponential_cdf(score_range, lamb)
    assignments = np.digitize(scores - mode, score_range, right=True)
    assignments[~np.isfinite(scores)] = 0  # If score was NaN, give it a p-value of ~1
    pvals = inverse_cdf[assignments]
    pvals_sorted = np.sort(pvals)
    ranks = np.arange(1, len(pvals_sorted) + 1)
    
    if plot:
        # Plot FDR cut-offs of various levels
        fdr_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        pval_threshes = []
        fdr_fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(ranks, pvals_sorted, color="black", label="p-values")
        for fdr in fdr_levels:
            bh_crit_vals = ranks / len(ranks) * fdr
            ax.plot(ranks, bh_crit_vals, label=("Crit values (FDR = %.2f)" % fdr))
            inds = np.where(pvals_sorted <= bh_crit_vals)[0]
            if not len(inds):
                pval_threshes.append(-1)
            else:
                pval_threshes.append(pvals_sorted[np.max(inds)])
        ax.set_title("Step-up p-values and FDR corrective critical values")
        plt.legend()
        plt.show()
    
        # Show table of number of hits at each FDR level
        header = vdomh.thead(
            vdomh.tr(
                vdomh.th("FDR level", style={"text-align": "center"}),
                vdomh.th("Number of hits kept", style={"text-align": "center"}),
                vdomh.th("% hits kept", style={"text-align": "center"})
            )
        )
        rows = []
        for i, fdr in enumerate(fdr_levels):
            num_kept = np.sum(pvals <= pval_threshes[i])
            frac_kept = num_kept / len(pvals)
            rows.append(vdomh.tr(
                vdomh.td("%.2f" % fdr), vdomh.td("%d" % num_kept), vdomh.td("%.2f%%" % (frac_kept * 100))
            ))
        body = vdomh.tbody(*rows)
        display(vdomh.table(header, body))

    # Perform filtering
    bh_crit_vals = fdr_cutoff * ranks / len(ranks)
    inds = np.where(pvals_sorted <= bh_crit_vals)[0]
    if not len(inds):
        pval_thresh = -1
    else:
        pval_thresh = pvals_sorted[np.max(inds)]
        
    if plot:
        return hit_table.iloc[pvals <= pval_thresh].reset_index(drop=True), (score_fig, fdr_fig)
    return hit_table.iloc[pvals <= pval_thresh].reset_index(drop=True)



def load_coords(peak_bed, in_window):
    if peak_bed.endswith(".gz"):
        with gzip.open(peak_bed) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(peak_bed) as f:
            lines = [line.split() for line in f]

    coords = []
    for line in lines:
        chrom, peak_start, peak_end = line[0], int(line[1]), int(line[2])
        mid = (peak_start + peak_end) // 2
        window_start = mid - in_window // 2
        window_end = mid + in_window // 2
        coords.append((chrom, window_start, window_end))
    return coords


def make_peak_table(peak_path, in_window):
    coords = load_coords(peak_path, in_window)

    peak_table = pd.DataFrame(coords, columns = ["peak_chrom", "peak_start", "peak_end"])
    peak_table["summit_offset"] = in_window // 2
    peak_table = peak_table.reset_index()
    
    return peak_table


def fix_peak_table(train_val_peak_table, val_peak_table):
    del val_peak_table["index"]
    del val_peak_table["summit_offset"]
    
    return pd.merge(val_peak_table, train_val_peak_table, how="inner",
                              on=["peak_chrom", "peak_start", "peak_end"])


def get_peak_hits_with_fix(peak_table, hit_table):
    """
    For each peak, extracts the set of motif hits that fall in that peak.
    Returns a list mapping peak index to a subtable of `hit_table`. The index
    of the list is the index of the peak table.
    """
    peak_hits = [pd.DataFrame(columns=list(hit_table))] * len(peak_table)

    peak_indexes = peak_table["index"].values
    actual_indexes = list(peak_table.index)
    peak_index_to_actual_index = dict(zip(peak_indexes, actual_indexes))

    for peak_index, matches in hit_table.groupby("peak_index"):
        if peak_index in peak_index_to_actual_index:
            peak_hits[peak_index_to_actual_index[peak_index]] = matches
    return peak_hits


def load_motif_hits(cell_type, timestamp, model_type, data_type,
                    task = "profile", fdr=0.05):
    
    # load in the motif hits file as a table
    modisco_config = ModiscoFilesConfig(cell_type, model_type,
                                        timestamp, task, data_type)
    modisco_results_dir = os.path.dirname(modisco_config.results_save_path)
    modisco_hits_path = os.path.join(modisco_results_dir, "motif_hits.bed")
    hits = import_motif_hits(modisco_hits_path)
    motif_keys = list(set(hits["key"]))

    # Filter motif hit table by p-value using FDR estimation
    filtered_hits = filter_peak_hits_by_fdr(hits,
                            score_column="fann_perclasssum_perc",
                            fdr_cutoff=fdr)

    # Match peaks to motif hits
    train_val_peak_table = make_peak_table(modisco_config.train_val_peak_path,
                                           modisco_config.in_window)

    # We need to subset to just the peaks in the val set (TODO: change to test set)
    val_config = ValFilesConfig(cell_type, model_type,
                                timestamp, data_type)
    val_peak_table = make_peak_table(val_config.val_peak_path,
                                     modisco_config.in_window)
    peak_table = fix_peak_table(train_val_peak_table, val_peak_table)

    peak_hits = get_peak_hits_with_fix(peak_table, filtered_hits)

    # Count hits of each motif in each peak
    peak_hit_counts = get_peak_motif_counts(peak_hits, motif_keys)
    
    return filtered_hits, peak_hits, peak_hit_counts


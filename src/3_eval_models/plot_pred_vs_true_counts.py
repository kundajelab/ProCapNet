import sys
assert len(sys.argv) == 2

sys.path.append('../2_train_models')

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_json
from data_loading import extract_peaks


# load filepaths and other info from config file

config_path = sys.argv[1]

config = load_json(config_path)

in_window = config["in_window"]
out_window = config["out_window"]

genome_path = config["genome_path"]
chrom_sizes = config["chrom_sizes"]

val_peak_path = config["val_peak_path"]

plus_bw_path = config["plus_bw_path"]
minus_bw_path = config["minus_bw_path"]

pred_logcounts_path = config["pred_logcounts_val_path"]
pred_profiles_path = config["pred_profiles_val_path"]

stranded = config["stranded_model"]


# where we will save the plots made by this script

out_dir = "plots/"
os.makedirs(out_dir, exist_ok=True)

out_base = config_path.split("model_out/")[1].replace("/", "__").replace("config.json", "") 
out_prefix = out_dir + out_base + "predvtruecounts__"


# load predicted read counts (made by val.py)

pred_logcounts = np.load(pred_logcounts_path).squeeze()
pred_counts = np.exp(pred_logcounts)

# determine if model was trained to output counts per strand
stranded = len(pred_counts.shape) == 2 and pred_counts.shape[-1] == 2

if stranded:
    pred_counts_stranded = pred_counts
    pred_counts_strand_merged = pred_counts.sum(axis=-1)
else:
    pred_counts_strand_merged = pred_counts
    
    # need to determine stranded counts indirectly for non-stranded models
    
    # load predicted profiles (made by val.py)
    pred_profiles = np.exp(np.load(pred_profiles_path))

    pred_profile_sums = pred_profiles.sum(axis=-1)
    pred_profile_strand_fracs = pred_profile_sums / pred_profile_sums.sum(axis=-1, keepdims=True)
    
    pred_counts_stranded = pred_profile_strand_fracs * pred_counts[...,None]


# load observed read counts

_, true_profs = extract_peaks(genome_path, chrom_sizes,
                              plus_bw_path, minus_bw_path,
                              val_peak_path,
                              in_window=in_window,
                              out_window=out_window,
                              max_jitter=0, verbose=True)

# don't collapse along strand axis, only sequence axis
true_counts_stranded = true_profs.sum(axis=-1)

# collapse along strand and sequence axis
true_counts_strand_merged = true_profs.sum(axis=(-1,-2))
    
    
def plot_pred_vs_true_counts(pred_counts, true_counts,
                             title = None, save_path = None):

    pred_counts = pred_counts.squeeze().flatten()
    true_counts = true_counts.squeeze().flatten()
    
    assert pred_counts.shape == true_counts.shape, (pred_counts.shape, true_counts.shape)
    
    pearson_r = np.corrcoef(np.log1p(pred_counts),
                            np.log1p(true_counts))[0,1]
    
    print("Pearson r (" + title + "):", pearson_r)
    
    plot_params = {
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    }
    plt.rcParams.update(plot_params)

    plt.figure(figsize=(7,6))

    plt.scatter(pred_counts, true_counts,
                alpha = 0.2, s = 8, color="blue")

    plt.semilogy()
    plt.semilogx()

    max_lim = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])
    min_lim = min(plt.gca().get_xlim()[0], plt.gca().get_ylim()[0])
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)

    plt.xlabel("Predicted Reads", fontsize=16)
    plt.ylabel("Observed Reads", fontsize=16)

    if pearson_r is not None:
        plt.text(min_lim * 1.7, max_lim * 0.4,
                 r'pearson $r = %0.2f$' % pearson_r,
                 fontsize=16)

    ax = plt.gca()
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 

    if title is not None:
        plt.title(title, fontsize=17, y=1.05)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        
    plt.show()
    
    

plot_pred_vs_true_counts(pred_counts_stranded, true_counts_stranded,
                         title = "Predicted vs. Observed Strand-Separate Reads",
                             save_path = out_prefix + "stranded.png")

plot_pred_vs_true_counts(pred_counts_strand_merged, true_counts_strand_merged,
                         title = "Predicted vs. Observed Strand-Merged Reads",
                         save_path = out_prefix + "strand_merged.png")

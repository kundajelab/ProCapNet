import numpy as np
from collections import defaultdict
import pandas as pd
import math
import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys

sys.path.append("../2_train_models")
from file_configs import MergedFilesConfig
from data_loading import extract_observed_profiles

sys.path.append("../5_modisco")
from report_utils import load_modisco_results

sys.path.append("../figure_notebooks")
from other_motif_utils import compute_per_position_ic


assert len(sys.argv) == 4, len(sys.argv)
cell_type = sys.argv[1]
task = sys.argv[2]
pdf_to_write_path = sys.argv[3]

assert cell_type in ["K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"], cell_type
assert task in ["profile", "counts"], task

data_type = "procap"
model_type = "strand_merged_umap"

config = MergedFilesConfig(cell_type, model_type, data_type)

out_window = 1000

true_profs = extract_observed_profiles(config.plus_bw_path,
                                       config.minus_bw_path,
                                       config.all_peak_path,
                                       out_window=out_window)

if task == "profile":
    modisco_results_path = config.modisco_profile_results_path
else:
    modisco_results_path = config.modisco_counts_results_path

modisco_results = load_modisco_results(modisco_results_path)



motif_names_to_bold = ["BRE/SP", "CA-Inr", "ETS", "NFY", "NRF1", "ATF1", "TATA",
                       "THAP11", "YY1", "AP1", "TA-Inr", "CTCF", "ZBTB33", "TCT", "TATATA",
                       "ATF4", "HNF1A/B", "TEAD", "FOX", "HNF4A/G", "EWS-FLI", "IRF/STAT",
                       "RFX", "CEBP", "SNAI", "GRHL1"]

if cell_type == "K562":
    if task == "profile":
        pattern_labels = ["BRE/SP", "CA-Inr", "ETS", "NFY", "NRF1",
                          "ATF1", "TATA", "THAP11", "YY1", "AP1",
                          "DPR_CGG", "DPR_CGG", "DPR_CGG", "TA-Inr", "DPR_CGG",
                          "CTCF", "DPR_CGG", "NRF1-like", "DPR_CGG", "ZBTB33",
                          "DPR_CGG", "TCT", "BRE/SP_TE", "TATATA", "CA-Inr_TE",
                          "DPR_CGG", "DPR_CGG", "TATA_TE", "CA-Inr_dimer", "DPR_CGG", "DPR_CGG",
                          "ZBTBT33_TATA_TE", "TE", "GC-rich", "CA-Inr_TE", "YY1-like",
                          "NFY_TE", "ZBTB33_TE", "ETS_TE", "ETS_dimer", "NFY_C-rich",
                          "ETS_CA-Inr_dimer", "YY1-like",
                          "BRE/SP_neg", "ETS_neg", "NFY_neg"]
    else:
        pattern_labels = ["ETS", "BRE/SP", "NRF1", "ETS-like", "NFY", "ATF1",
                          "CpG", "AP1", "CpG_spacing", "CpG_spacing", "THAP11",
                          "CpG_spacing", "CpG_spacing", "CpG_spacing",
                          "ZBTB33", "CpG_spacing", "BRE/SP_TE", "CpG_spacing", "CpG_spacing",
                          "THAP11-like", "CpG_spacing", "BRE/SP_TE", "ZBTB33_TE", "CpG_spacing",
                          "BRE/SP_TE", "TE", "BRE/SP_TE", "TE", "TE",
                          "ETS-like", "BRE/SP_TE", "Unknown", "NFY_TE"] + ["Unknown"] * 16
elif cell_type == "A673":
    if task == "profile":
        pattern_labels = ["CA-Inr", "DPR_GC-rich","CA-Inr-like", "BRE/SP", "NFY",
                          "ETS", "TATA", "ATF1", "DPR_GC-rich", "DPR_GC-rich",
                          "DPR_GC-rich", "NRF1", "AP1", "YY1", "C-rich",
                          "ATF4", "TA-Inr", "THAP11","TATA_TA-Inr-like", "TCT",
                          "CTCF", "TATA_TA-Inr-like", "ZBTB33", "TATA_TA-Inr-like", "DPR_GC-rich",
                          "CA-Inr_TE", "NRF1_Inr", "CA-Inr_TE", "TE", "ZBTB33_TE",
                          "DPR", "DPR", "DPR", "Unknown", "CA-Inr-like",
                          "CA-Inr-like_TE", "Unknown", "NFY_TE", "YY1-like", "TE",
                          "BRE/SP_TE", "YY1-like", "YY1-like"]
    else:
        pattern_labels = ["ETS", "BRE/SP", "NFY", "ATF1", "AP1",
                          "NRF1", "YY1", "ATF4", "THAP11", "ZBTB33",
                          "EWS-FLI", "ETS_ATF_dimer","ATF1-like", "NRF1-half", "TATA_TATATA",
                          "ETS_dimer", "TATA-TA-Inr-like", "BRE/SP_TE", "TCT", "NFY_TE",
                          "TATA-like", "NFY_TE", "ATF4-like", "BRE/SP-like", "CTCF_neg",
                          "SNAI_neg"]
elif cell_type == "CACO2":
    if task == "profile":
        pattern_labels = ["CA-Inr", "BRE/SP", "DPR_GC-rich", "NFY", "ATF1",
                          "ETS", "NRF1", "TATA", "TEAD", "DPR_GC-rich",
                          "YY1", "TA-Inr", "AP1", "HNF1A/B", "THAP11",
                          "CTCF", "HNF4A/G", "ZBTB33", "C-rich", "ATF4",
                          "YY1_TE", "TA-Inr-like", "TATATA", "DPR_GC-rich", "NRF1_Inr",
                          "TCT", "Inr-like", "YY1-Inr", "TEAD_HNF4_dimer", "BRE/SP_TE",
                          "CA-Inr-like", "FOX", "CA-Inr_TE", "BRE/SP-like", "CA-Inr_TE",
                          "NFY_TE", "CA-Inr_TE", "THAP11-like", "DPR", "Unknown",
                          "FOX-like", "TEAD-like", "YY1_CA-Inr_dimer"]
    else:
        pattern_labels = ["BRE/SP", "TEAD", "ATF1", "ETS", "NFY", "NRF1",
                          "AP1", "HNF4A/G", "HNF1A/B", "YY1", "ATF4",
                          "THAP11", "ETS-like", "ZBTB33", "CpG", "AP1-like",
                          "ETS_ATF1_dimer", "TEAD_dimer", "FOX", "TEAD-like_Inr_TCT-like", "FOX-like",
                          "THAP11", "TE", "TE", "BRE_TATA_dimer", "CpG",
                          "ZBTB33", "GRHL1", "Unknown", "TEAD_TE", "ETS_dimer",
                          "CEBP", "NFY_TE", "TEAD-like_TE", "TE", "Unknown",
                          "TE", "FOX-like", "BRE/SP_ETS-like_dimer", "TEAD_TE", "THAP11-like_TE",
                          "CTCF"]
elif cell_type == "CALU3":
    if task == "profile":
        pattern_labels = ["CA-Inr", "BRE/SP", "DPR_G-rich", "ETS", "NFY",
                          "AP1", "ATF1",  "DPR_GCC", "DPR_GC-rich", "NRF1", 
                          "YY1", "DPR_GC-rich", "G-rich", "TATA-TATATA", "THAP11", 
                          "HNF1A/B", "NRF1-like", "ZBTB33", "CTCF","ETS_dimer",
                          "TATA-TA-Inr-like", "ATF1-like", "NRF1_Inr", "YY1_Inr", "CTCF-like_DPR",
                         "YY1_TE", "Inr-like", "TA-Inr", "ETS_dimer", "BRE/SP_CA-Inr_dimer",
                          "DPR_GC-rich", "CA-Inr-like", "ETS_dimer", "YY1-like", "RFX",
                          "NRF1-like"]
    else:
        pattern_labels = ["ETS", "BRE/SP", "AP1", "ATF1", "NFY",
                          "NRF1", "YY1", "HNF1A/B", "ATF1-like", "THAP11",
                          "ZBTB33", "ETS_ATF1_dimer", "IRF/STAT", "BRE/SP_ETS_dimer", "THAP11-like",
                          "RFX", "IRF/STAT", "BRE/SP_ETS_dimer", "NFY_ATF1_dimer", "IRF/STAT-like",
                          "YY1_TE", "TATATA", "RFX-like", "THAP11-like", "CTCF"]
elif cell_type == "HUVEC":
    if task == "profile":
        pattern_labels = ["CA-Inr", "ETS", "BRE/SP", "NFY", "AP1",
                          "DPR_GC-rich", "ATF1", "DPR_GC-rich", "NRF1", "DPR_GC-rich",
                          "DPR_GC-rich", "DPR_GC-rich", "TATA", "YY1","BRE/SP_G-rich",
                          "TA-Inr", "THAP11", "TA-Inr-like", "ZBTB33","ATF1-like",
                          "TATATA", "DPR_GC-rich", "DPR_GC-rich", "CTCF", "AP1_Inr-like",
                          "NRF1_Inr", "YY1_Inr", "YY1_Inr", "DPR_GC-rich", "DPR_GC-rich",
                          "DPR_GC-rich"]
    else:
        pattern_labels = ["ETS-like", "ETS", "AP1", "CpG", "ATF1",
                          "BRE/SP", "NFY", "NRF1", "BRE/SP-like", "Unknown",
                          "ATF1", "ETS_ATF1_dimer", "ZBTB33", "YY1", "THAP11",
                          "AP1_ETS_dimer", "GC-rich", "ETS_half-dimer", "ETS_BRE/SP_dimer", "ETS-like",
                          "BRE/SP-like", "CpG", "ZBTB33-like", "ETS_BRE/SP_dimer", "Unknown",
                          "ETS-like"]
elif cell_type == "MCF10A":
    if task == "profile":
        pattern_labels = ["CA-Inr", "AP1", "BRE/SP", "DPR_C-rich", "NFY",
                          "ATF1", "ETS", "DPR_CCG", "NRF1", "DPR_GC-rich",
                          "ATF4", "TA-Inr","C-rich", "DPR_GC-rich", "YY1",
                          "TATA", "THAP11", "DPR_C-rich", "CTCF", "ZBTB33",
                          "DPR_C-rich", "ETS-like_Inr", "NRF1_Inr", "DPR_CCG","TA-Inr-like",
                         "DPR_C-rich", "ATF1-like", "CEBP", "YY1_Inr","AP1_Inr",
                          "RFX", "BRE/SP-like", "TCT-TA-Inr-like", "RFX-like", "YY1_Inr", "NRF1_Inr_dimer"]
    else:
        pattern_labels = ["AP1", "ETS", "ATF1", "BRE/SP", "BRE/SP-like",
                          "ATF4", "NFY", "CpG", "ETS-like", "NRF1",
                          "YY1", "AP1-like", "CpG_spacing", "CpG_spacing", "AP1-like",
                          "THAP11", "ETS_ATF1_dimer", "GC-rich", "ZBTB33", "CpG_spacing",
                          "CEBP", "NRF1-like", "CpG_spacing", "ATF1-like", "CpG_spacing",
                          "ATF4-like", "CpG_spacing", "ETS_BRE/SP_dimer", "IRF/STAT", "THAP11-like",
                          "RFX", "BRE/SP_ETS_dimer", "CpG_spacing", "BRE_TATA_dimer", "CpG_ATF1-like",
                          "T-rich", "BRE/SP_ETS_dimer", "ZBTB33-like", "ETS-like"]

    
def load_ppms_cwms(modisco_results):
    motifs = defaultdict(lambda : [])
    for pattern_group in ['pos_patterns', 'neg_patterns']:
        if pattern_group not in modisco_results.keys():
            continue

        metacluster = modisco_results[pattern_group]
        for pattern_i in range(len(metacluster.keys())):
            pattern_name = "pattern_" + str(pattern_i)
            pattern = metacluster[pattern_name]
            ppm = np.array(pattern['sequence'][:])
            cwm = np.array(pattern["contrib_scores"][:])
            
            pwm = ppm * compute_per_position_ic(ppm)[:, None]

            motifs[pattern_group].append((ppm, pwm, cwm))
    return motifs


def plot_pattern_on_ax(ax, array):
    assert len(array.shape) == 2 and array.shape[-1] == 4, array.shape
    # reformat pwm to what logomaker expects
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'

    # plot motif ("baseline_width=0" removes the y=0 line)
    crp_logo = logomaker.Logo(df, ax=ax, font_name='Arial Rounded', baseline_width=0)
    crp_logo.style_spines(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
def make_title_row(axes, fig, row_i, title):
    ax_row = axes[row_i]
    for ax in ax_row:
        ax.set_axis_off()

    gs = ax_row[0].get_gridspec()

    title_row = fig.add_subplot(gs[row_i, :])
    title_row.spines[["top", "right", "bottom", "left"]].set_visible(False)
    title_row.set_xticks([])
    title_row.set_yticks([])
    title_row.patch.set_alpha(0.)
    
    title_row.text(0.5, 0.4, title,
                   ha="center", va="center",
                   fontsize=9, weight="bold")
                
                
def extract_profs_at_seqlets(seqlets, profiles,
                             out_window=out_window, target_prof_width=200):

    # note that this function won't work correctly if your out_window
    # doesn't match what your slice_len was set to when you ran modisco!
    # (seqlet coordinates will be off)
    
    assert len(profiles.shape) == 3 and profiles.shape[-1] == out_window, profiles.shape
    assert target_prof_width <= out_window // 2
    
    # extract coordinates for each seqlet
    peak_indexes = seqlets["example_idx"][:]
    seqlet_starts = seqlets["start"][:]
    seqlet_ends = seqlets["end"][:]
    seqlet_rcs = seqlets["is_revcomp"][:]
    
    # get coordinate centered on the seqlet center to extract the data profile around
    seqlet_mids = (seqlet_starts + seqlet_ends) // 2
    
    
    # For each seqlet, fetch the true/predicted profiles
    
    seqlet_profs = []
    for peak_index, seqlet_mid, is_seqlet_rc in zip(peak_indexes, seqlet_mids, seqlet_rcs):
        prof_start = seqlet_mid - target_prof_width
        prof_end = prof_start + target_prof_width * 2
        
        if prof_start < 0 or prof_end > out_window:
            # the seqlet is too close to the edge of the profile,
            # so we will pad with nans (ignored during averaging later)
        
            adjusted_prof_start = max(0, prof_start)
            adjusted_prof_end = min(out_window, prof_end)
            
            subset_of_prof = profiles[peak_index, :, adjusted_prof_start:adjusted_prof_end]
            
            num_bases_missing = target_prof_width * 2 - subset_of_prof.shape[-1]
            padded_array = np.full((subset_of_prof.shape[0], num_bases_missing), np.nan)
            
            if prof_start < 0:
                # the seqlet was too far left, so adjust the profile by shifting to the right
                prof_at_seqlet = np.concatenate((padded_array, subset_of_prof), axis=1)
            else:
                # the seqlet was too far right, so adjust the profile by shifting to the left
                prof_at_seqlet = np.concatenate((subset_of_prof, padded_array), axis=1)
                
        else:
            prof_at_seqlet = profiles[peak_index, :, prof_start:prof_end]
            
            if prof_at_seqlet.shape[-1] == 0:
                print(peak_index, prof_start, prof_end)
            
        
        if is_seqlet_rc:
            # we want to align profiles with respect to motif orientation.
            # so if the motif is flipped, flip the profile too (and swap strands)
            prof_at_seqlet = prof_at_seqlet[::-1, ::-1]
        
        seqlet_profs.append(prof_at_seqlet)  
    
    seqlet_profs = np.stack(seqlet_profs)
    
    assert seqlet_profs.shape[-1] == target_prof_width * 2, seqlet_profs.shape
    return seqlet_profs


def load_profiles_at_motifs_from_modisco_results(profs, modisco_results):
    profiles_at_motifs = []
    for pattern_group_name in ['pos_patterns', 'neg_patterns']:
        if pattern_group_name not in modisco_results.keys():
            continue

        pattern_group = modisco_results[pattern_group_name]
        for pattern_i in range(len(pattern_group.keys())):
            pattern_name = "pattern_" + str(pattern_i)
            pattern = pattern_group[pattern_name]
            seqlets = pattern["seqlets"]

            profs_at_seqlets = extract_profs_at_seqlets(seqlets, profs)

            profiles_at_motifs.append(np.nanmean(profs_at_seqlets, axis=0))
    
    return np.array(profiles_at_motifs)


def plot_avg_profile(profiles, ax, prof_width = 200, bottom_ticks=False,
                     color = None, tick_len=2.5, tick_fontsize=6):
    assert len(profiles.shape) == 2

    # Plot the average predictions
    prof_center = profiles.shape[-1] // 2
    ax.plot(profiles[0, prof_center - prof_width:prof_center + prof_width],
            color="#001DAC", linewidth=0.7)
    ax.plot(-profiles[1, prof_center - prof_width:prof_center + prof_width],
            color="#001DAC", alpha = 0.5, linewidth=0.7)
    
    # Set axes
    max_mean_val = np.max(profiles)
    
    mean_ylim = math.ceil(max_mean_val * 1.02)  # Make 5% higher
        
    ax.set_ylim(-mean_ylim, mean_ylim)
    ax.set_xlim(- int(prof_width / 50), 2 * prof_width)

    ax.set_yticks([0, mean_ylim], [0, mean_ylim],
                  fontsize=tick_fontsize)
    ax.tick_params("y", length=tick_len, pad=2)
    
    if not bottom_ticks:
        ax.set_xticks([])
        for side in ["top", "right", "bottom"]:
            ax.spines[side].set_visible(False)
        
    else:
        ax.set_xticks([prof_width, 2 * prof_width], [0, prof_width],
                      fontsize=tick_fontsize)
        ax.tick_params("x", length=tick_len, pad=2, labelsize=tick_fontsize)
        
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        
        x_rect = patches.Rectangle((- int(prof_width / 50), - mean_ylim * 1.1),
                                   prof_width, mean_ylim / 7,
                                   clip_on=False, zorder=50, color="white")
        ax.add_patch(x_rect)
        
        ax.spines['bottom'].set_position(('data', - mean_ylim))
        ax.spines["bottom"].set_color("#333333")
        
    y_rect = patches.Rectangle((- 2 * int(prof_width / 50), - mean_ylim),
                               5, mean_ylim * 0.93,
                               clip_on=False, zorder=50, color="white")
    ax.add_patch(y_rect)


def draw_modisco_results_on_axes(axes, motif_labels, motifs, profs_at_motifs):
    assert len(motif_labels) == len(motifs), (len(motif_labels), len(motifs))
    column_labels = ["Pattern", "Description", "PWM", "CWM", None, "Avg. PRO-cap Signal"]

    for row_i in range(len(motifs) + 1):
        if row_i == 0:
            for col_i in range(len(column_labels)):
                axes[row_i, col_i].text(0.5, 0.5, column_labels[col_i],
                                        ha="center", va="center", color="k", fontsize=8)
                
                axes[row_i, col_i].set_axis_off()

        else:
            axes[row_i, 0].text(0.5, 0.5, row_i,
                                ha="center", va="center", color="k", fontsize=7)
            
            if motif_labels[row_i - 1] is not None:
                motif_name = motif_labels[row_i - 1]
                
                if motif_name.replace("_neg", "") in motif_names_to_bold:
                    axes[row_i, 1].text(0.5, 0.5, motif_name,
                                        ha="center", va="center", color="k",
                                        fontsize=7, weight="bold")
                else:
                    axes[row_i, 1].text(0.5, 0.5, motif_name,
                                        ha="center", va="center", color="k",
                                        fontsize=7)

            _, pwm, cwm = motifs[row_i - 1]

            plot_pattern_on_ax(axes[row_i, 2], pwm)
            plot_pattern_on_ax(axes[row_i, 3], cwm)
            
            plot_avg_profile(profs_at_motifs[row_i - 1], axes[row_i, 5],
                             bottom_ticks = row_i == len(motifs))

            for col_i in [0,1,4]:
                axes[row_i, col_i].set_axis_off()
                axes[row_i, col_i].patch.set_alpha(0.)
                
            axes[row_i, 2].patch.set_alpha(0.)
            axes[row_i, 3].patch.set_alpha(0.)
                

def make_modisco_patterns_report(modisco_motifs, profs_at_motifs, pattern_labels,
                                 save_path = None):

    two_pattern_groups = len(modisco_motifs.keys()) == 2

    if two_pattern_groups:
        # positive and negative patterns
        num_motifs_per_group = [len(modisco_motifs["pos_patterns"]), len(modisco_motifs["neg_patterns"])]
        num_motifs_total = sum(num_motifs_per_group)
        num_rows = num_motifs_total + 4
    else:
        # only positive patterns...probably
        # if you're reading this. hello. nice to meet you.
        # if you came here because of this assert failing,
        # please tell me. I want to see the dataset with
        # only negative motifs. thanks. have a good day.
        assert "pos_patterns" in modisco_motifs.keys(), modisco_motifs.keys()
        num_motifs_per_group = [len(modisco_motifs["pos_patterns"])]
        num_motifs_total = sum(num_motifs_per_group)
        num_rows = num_motifs_total + 2
        
    assert len(pattern_labels) == num_motifs_total, (len(pattern_labels), num_motifs_total)

    fig, axes = plt.subplots(num_rows, 6,
                             figsize=(8, num_rows * 0.5), dpi=300,
                             gridspec_kw={"width_ratios" : [0.2, 0.5, 1, 1, 0.02, 0.6]})

    make_title_row(axes, fig, 0, "Positive Patterns")

    draw_modisco_results_on_axes(axes[1 : len(modisco_motifs["pos_patterns"]) + 2],
                                 pattern_labels[:len(modisco_motifs["pos_patterns"])],
                                 modisco_motifs["pos_patterns"],
                                 profs_at_motifs[:len(modisco_motifs["pos_patterns"])])

    if 'neg_patterns' in modisco_motifs.keys():
        neg_patterns_start_row = len(modisco_motifs["pos_patterns"]) + 2

        make_title_row(axes, fig, neg_patterns_start_row, "Negative Patterns")

        draw_modisco_results_on_axes(axes[neg_patterns_start_row + 1 :],
                                     pattern_labels[- len(modisco_motifs["neg_patterns"]) :],
                                     modisco_motifs["neg_patterns"],
                                     profs_at_motifs[- len(modisco_motifs["neg_patterns"]) :])

    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.5, dpi = 300)

    plt.show()
    
    
modisco_motifs = load_ppms_cwms(modisco_results)

profiles_at_motifs = load_profiles_at_motifs_from_modisco_results(true_profs, modisco_results)
    
make_modisco_patterns_report(modisco_motifs, profiles_at_motifs, pattern_labels,
                             save_path = pdf_to_write_path)

print("Done making report for " + cell_type + ", " + task)



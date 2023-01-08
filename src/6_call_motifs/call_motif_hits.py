import sys
sys.path.append("../2_train_models")
sys.path.append("../5_modisco")

import os
import numpy as np
import h5py
import pandas as pd
import gzip

import modisco
from modisco.hit_scoring import densityadapted_hitscoring

from utils import load_json
from modiscolite_utils import load_sequences, load_scores
from old_modisco_format_utils import import_tfmodisco_results, get_patterns_from_modisco_results

# Load info from config file of modisco run

assert len(sys.argv) == 3, len(sys.argv)  # expecting config path, list of motifs to keep (comma-separated indexes)
config_path = sys.argv[1]
patterns_to_keep =  [int(num) for num in sys.argv[2].split(",")]  # SPECIFY FOR EACH EXPERIMENT

config = load_json(config_path)

data_type = config["data_type"]
cell_type = config["cell_type"]
model_type = config["model_type"]
timestamp = config["timestamp"]
task = config["task"]

in_window = config["in_window"]
out_window = config["out_window"]

genome_path = config["genome_path"]
chrom_sizes = config["chrom_sizes"]

peak_path = config["train_val_peak_path"]

stranded = config["stranded_model"]

slice_len = config["slice"]

scores_path = config["scores_path"]

modisco_out_path = config["results_save_path"].replace("modisco_results.h", "old_fmt_modisco_results.h")
assert os.path.exists(modisco_out_path), modisco_out_path  # run conversion script before running this!
print("Using old-format modisco file at " + modisco_out_path + ".")

# Load data, modisco results

onehot_seqs = load_sequences(genome_path, chrom_sizes, peak_path,
                                 slice_len, in_window=in_window)

scores = load_scores(scores_path, slice_len, in_window=in_window)
act_scores = scores * onehot_seqs

modisco_results = import_tfmodisco_results(modisco_out_path, scores, onehot_seqs)
patterns = get_patterns_from_modisco_results(modisco_results)

filtered_patterns_list = [patterns[i] for i in patterns_to_keep] 
print("Will keep patterns: ", patterns_to_keep)


# Instantiate the hit scorer

hit_scorer = densityadapted_hitscoring.MakeHitScorer(
    patterns=filtered_patterns_list,
    target_seqlet_size=25,
    bg_freq=np.mean(onehot_seqs, axis=(0, 1)),
    task_names_and_signs=[("task0", 1)],
    n_cores=15
)

# Set seqlet identification method

hit_scorer.set_coordproducer(
    contrib_scores={"task0": act_scores},
    core_sliding_window_size=7,
    target_fdr=0.2,
    min_passing_windows_frac=0.03,
    max_passing_windows_frac=0.2,
    separate_pos_neg_thresholds=False,                             
    max_seqlets_total=np.inf
)


print("Starting hit scoring...")

# this takes forever (possibly up to 8-24 hours)

batch_size = 1024
num_batches = int(np.ceil(len(act_scores) / batch_size))
rows = []

for i in range(num_batches):
    print("\tScoring batch %d/%d" % (i + 1, num_batches))
    batch_slice = slice(i * batch_size, (i + 1) * batch_size)
    example_to_matches, pattern_to_matches = hit_scorer(
        contrib_scores={"task0": act_scores[batch_slice]},
        hypothetical_contribs={"task0": scores[batch_slice]},
        one_hot=onehot_seqs[batch_slice],
        hits_to_return_per_seqlet=1
    )

    offset = i * batch_size
    for example_index, match_list in example_to_matches.items():
        for match in match_list:
            rows.append([
                match.exampleidx + offset, match.patternidx, match.start,
                match.end, match.is_revcomp, match.total_importance,
                match.aggregate_sim, match.mod_delta, match.mod_precision,
                match.mod_percentile, match.fann_perclasssum_perc,
                match.fann_perclassavg_perc
            ])
            
            
# Load in peak info (to combine with hits)

def load_coords(peak_bed, peak_half_len = in_window):
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
        window_start = mid - peak_half_len // 2
        window_end = mid + peak_half_len // 2
        coords.append((chrom, window_start, window_end))
    return coords
    
def make_peak_table(peak_path):
    coords = load_coords(peak_path)

    peak_table = pd.DataFrame(coords, columns = ["peak_chrom", "peak_start", "peak_end"])
    peak_table["summit_offset"] = in_window // 2
    peak_table = peak_table.reset_index()
    
    return peak_table


peak_table = make_peak_table(peak_path)


print("Cleaning up matches...")

# Collate the matches together into a big table
colnames = [
    "example_index", "pattern_index", "start", "end", "revcomp",
    "imp_total_score", "agg_sim", "mod_delta", "mod_precision",
    "mod_percentile", "fann_perclasssum_perc", "fann_perclassavg_perc"
]
match_table = pd.DataFrame(rows, columns=colnames)

# Compute importance fraction of each hit, using just the matched actual scores
total_track_imp = np.sum(np.abs(act_scores), axis=(1, 2))
match_table["imp_frac_score"] = match_table["imp_total_score"] / \
    total_track_imp[match_table["example_index"]]

# Convert example index to peak index
match_table["peak_index"] = match_table["example_index"]

# Convert pattern index to motif key
match_table["key"] = match_table["pattern_index"]

# Convert revcomp to strand
# Note we are assuming that the input scores were all positive strand
match_table["strand"] = match_table["revcomp"].map({True: "-", False: "+"})

# Convert start/end of motif hit to genomic coordinate
# `peak_starts[i] == j` is such that if `i` is a peak index, `j` is the peak
# start in genomic coordinate space
peak_starts = np.empty(np.max(peak_table["index"]) + 1, dtype=int)
peak_starts[peak_table["index"]] = peak_table["peak_start"]
# Now reduce `peak_starts` to match `match_table` exactly
peak_starts = peak_starts[match_table["peak_index"]]
offset = (in_window - slice_len) // 2

match_table["chrom"] = peak_table["peak_chrom"].loc[
    match_table["peak_index"]
].reset_index(drop=True)
# Note: "peak_chrom" was an index column so we need to drop that before
# setting it as a value
match_table["start"] = match_table["start"] + offset + peak_starts
match_table["end"] = match_table["end"] + offset + peak_starts

# Re-order columns (and drop a few) before saving the result
match_table = match_table[[
    "chrom", "start", "end", "key", "strand", "peak_index",
    "imp_total_score", "imp_frac_score", "agg_sim", "mod_delta",
    "mod_precision", "mod_percentile", "fann_perclasssum_perc",
    "fann_perclassavg_perc"
]]


match_table.to_csv(os.path.join(os.path.dirname(modisco_out_path), "motif_hits.bed"),
                   sep="\t", header=False, index=False)

print("Done calling motif hits.")

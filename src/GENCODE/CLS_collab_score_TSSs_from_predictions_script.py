import numpy as np
from collections import defaultdict
import gzip
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
import pyBigWig
import os

import sys
sys.path.append("../utils")
from misc import load_chrom_sizes

primary_transcripts = True

if primary_transcripts:
    print("Running on primary transcripts.")
    
    regions_out_dir = "primary_transcripts_regions_to_predict/"
    preds_dir = "primary_transcripts_predictions/"
else:
    regions_out_dir = "regions_to_predict/"
    preds_dir = "predictions/"


chrom_sizes = "genome/hg38.gencode_naming.chrom.sizes"
chrom_sizes_dict = {k : v for (k,v) in load_chrom_sizes(chrom_sizes)}

TSSs_filepath = regions_out_dir + "all_TSSs.bed.gz"

assert len(sys.argv) == 2, len(sys.argv)
cell_type = sys.argv[1]

preds_out_dir = preds_dir + cell_type + "/"

preds_bws = {"+" : preds_out_dir + "preds." + cell_type + ".pos.bigWig",
             "-" : preds_out_dir + "preds." + cell_type + ".neg.bigWig"}

assert os.path.exists(preds_bws["+"]) and os.path.exists(preds_bws["-"]), preds_bws

scored_TSSs_filepath = preds_out_dir + "all_TSSs_scored.bed.gz"

extend_bys = [0,1,2,3,5,10,20,25,50,100,500]



def extract_observed_profiles(plus_bw_path, minus_bw_path, peak_path,
                              extend_by = 0, verbose=True):
    signals = []

    names = ['chrom', 'start', 'end', 'strand']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2, 3), 
        header=None, index_col=False, names=names)

    assert os.path.exists(plus_bw_path), plus_bw_path
    assert os.path.exists(minus_bw_path), minus_bw_path
    plus_bw = pyBigWig.open(plus_bw_path, "r")
    minus_bw = pyBigWig.open(minus_bw_path, "r")

    desc = "Loading Profiles"
    d = not verbose
    for _, (chrom, start, end, strand) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        if strand == "+":
            try:
                signal = plus_bw.values(chrom, start - extend_by, end + extend_by, numpy=True)
            except:
                signal = plus_bw.values(chrom, max(0, start - extend_by),
                                        min(chrom_sizes_dict[chrom], end + extend_by), numpy=True)
        else:
            try:
                signal = minus_bw.values(chrom, start - extend_by, end + extend_by, numpy=True)
            except:
                signal = minus_bw.values(chrom, max(0, start - extend_by),
                                         min(chrom_sizes_dict[chrom], end + extend_by), numpy=True)
            
        signal = np.nan_to_num(signal)

        # Append signal to growing signal list
        assert extend_by == 500 or len(signal) == end - start + 2 * extend_by, (len(signal), start, end, extend_by)
        
        # how to collapse small window around TSS? for now, take max:
        signals.append(np.max(signal))

    assert len(signals) == len(peaks), (len(signals), len(peaks))
    return np.array(signals)


TSS_scores = dict()
for extend_by in extend_bys:
    print(extend_by)
    TSS_scores[extend_by] = extract_observed_profiles(preds_bws["+"], preds_bws["-"],
                                                      TSSs_filepath, extend_by=extend_by)
    
    
TSS_scores_norm = {extend_by : TSS_scores[extend_by] / TSS_scores[500] for extend_by in TSS_scores.keys()}

TSS_scores_log_norm = {extend_by : np.log1p(TSS_scores[extend_by]) / np.log1p(TSS_scores[500]) for extend_by in TSS_scores.keys()}


def make_megatable(original_TSS_bed, scores, scores_norm, scores_log_norm, extend_bys=extend_bys):
    arr = []
    
    with gzip.open(original_TSS_bed) as original_bed:
        for line_i, line in enumerate(original_bed):
            new_line = line.decode().rstrip().split()
            
            scores_line = [scores[extend_by][line_i] for extend_by in extend_bys]
            new_line.extend(scores_line)

            # don't include the scores that just get normalized to be 1
            scores_norm_line = [scores_norm[extend_by][line_i] for extend_by in extend_bys[:-1]]
            new_line.extend(scores_norm_line)

            scores_log_norm_line = [scores_log_norm[extend_by][line_i] for extend_by in extend_bys[:-1]]
            new_line.extend(scores_log_norm_line)
            
            arr.append(new_line)
            
    arr = np.array(arr)
    return arr

all_scores = make_megatable(TSSs_filepath, TSS_scores, TSS_scores_norm, TSS_scores_log_norm)


def write_TSS_scores_to_bed(out_TSS_bed, megatable, extend_bys = extend_bys,
                            primary_transcripts=primary_transcripts):
    
    if primary_transcripts:
        colnames = ["chrom", "start", "end", "strand", "gene_type", "transcript_type"]
    else:
        colnames = ["chrom", "start", "end", "strand", "support", "seq_tech", "capture"]
    
    for extend_by in extend_bys:
        colnames.append("score_window_extended_" + str(extend_by) + "bp")
        
    for extend_by in extend_bys[:-1]:
        colnames.append("score_norm_window_extended_" + str(extend_by) + "bp")
        
    for extend_by in extend_bys[:-1]:
        colnames.append("score_log_norm_window_extended_" + str(extend_by) + "bp")
        
    colnames_line = "\t".join(colnames) + "\n"
        
    with gzip.open(out_TSS_bed, "w") as out_bed:
        out_bed.write(colnames_line.encode())
        for row in megatable:
            new_line = "\t".join([str(thing) for thing in row]) + "\n"
            out_bed.write(new_line.encode())
                

write_TSS_scores_to_bed(scored_TSSs_filepath, all_scores)

print("Done writing scores to ", scored_TSSs_filepath)
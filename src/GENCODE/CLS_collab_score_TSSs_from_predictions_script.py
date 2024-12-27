import numpy as np
from collections import defaultdict
import gzip
import pandas as pd
from tqdm import tqdm
import sys
import pyBigWig
import os

import sys
sys.path.append("../utils")
from misc import load_chrom_sizes


assert len(sys.argv) == 2, len(sys.argv)
cell_type = sys.argv[1]



random_relocate = False

if random_relocate: # the negative control regions Tamara sent
    print("Running on decoy models!")
    
    regions_out_dir = "random_relocate_regions_to_predict/"
    preds_dir = "random_relocate_predictions/"
else:
    regions_out_dir = "regions_to_predict/"
    preds_dir = "predictions/"


chrom_sizes = "genome/hg38.gencode_naming.chrom.sizes"
chrom_sizes_dict = {k : v for (k,v) in load_chrom_sizes(chrom_sizes)}

# all of the candidate TSSs generated from the CLS expts scored
# made by CLS_collab_make_regions.ipynb
TSSs_filepath = regions_out_dir + "all_TSSs.bed.gz"

# where predictions were saved to: made by CLS_collab_make_predictions_script.py
preds_out_dir = preds_dir + cell_type + "/"

preds_bws = {"+" : preds_out_dir + "preds." + cell_type + ".pos.bigWig",
             "-" : preds_out_dir + "preds." + cell_type + ".neg.bigWig"}

assert os.path.exists(preds_bws["+"]) and os.path.exists(preds_bws["-"]), preds_bws

# where we will save predictions converted to scores 
scored_TSSs_filepath = preds_out_dir + "all_TSSs_scored.bed.gz"

# all window widths we aggregated predictions over
# (ex: "5" means TSS score is predicted PRO-cap within +/- 5 bp of TSS)
extend_bys = [0,1,2,3,5,10,20,25,50,100,500]





### Load in the predictions from their bigwigs

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

    

def make_megatable_and_save(original_TSS_bed, scores, out_TSS_bed,
                            extend_bys=extend_bys,
                            random_relocate=random_relocate):
    
    if random_relocate:
        colnames = ["chrom", "start", "end", "strand", "gene_type"]
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
    
        with gzip.open(original_TSS_bed) as original_bed:
            for line_i, line in enumerate(original_bed):
                new_line = line.decode().rstrip().split()

                # do it in the order below to keep consistent with earlier file format

                for extend_by in extend_bys:
                    new_line.append(scores[extend_by][line_i])

                # don't include the scores that just get normalized to be 1
                for extend_by in extend_bys[:-1]:
                    new_line.append(scores[extend_by][line_i] / scores[extend_bys[-1]][line_i])

                for extend_by in extend_bys[:-1]:
                    new_line.append(np.log1p(scores[extend_by][line_i]) / np.log1p(scores[extend_bys[-1]][line_i]))

                new_line = "\t".join([str(thing) for thing in new_line]) + "\n"
                out_bed.write(new_line.encode())


                
                
TSS_scores = dict()
for extend_by in extend_bys:
    TSS_scores[extend_by] = extract_observed_profiles(preds_bws["+"], preds_bws["-"],
                                                      TSSs_filepath, extend_by=extend_by)

    
make_megatable_and_save(TSSs_filepath, TSS_scores, scored_TSSs_filepath)
            


print("Done writing scores to ", scored_TSSs_filepath)

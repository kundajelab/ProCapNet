import numpy as np
from collections import defaultdict
import gzip
import pandas as pd
from tqdm import tqdm
import sys
import pyBigWig
import os


random_relocate = False

if random_relocate:  # the negative control regions Tamara sent
    print("Running on decody models!")
    
    regions_out_dir = "random_relocate_regions_to_predict/"
    preds_dir = "random_relocate_predictions/"
else:
    regions_out_dir = "regions_to_predict/"
    preds_dir = "predictions/"


# all cell types we have ProCapNet models for
cell_types = ["K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"]

# all window widths we aggregated predictions over
# (ex: "5" means TSS score is predicted PRO-cap within +/- 5 bp of TSS)
extend_bys = [0,1,2,3,5,10,20,25,50,100,500]


# all of the candidate TSSs generated from the CLS expts scored
# made by CLS_collab_make_regions.ipynb
TSSs_filepath = regions_out_dir + "all_TSSs.bed.gz"

# where we will save the files of scores merged across cell types
out_dir = preds_dir + "merged_across_cells/"
os.makedirs(out_dir, exist_ok=True)


# ways scores will be summarized across the cell types
# ("top3" = mean of 3 highest scores)
mean_or_max_options = ["mean", "max", "top3"]



# functions holding filepaths

def get_preds_bw_path(cell_type, pos_or_neg):
    assert pos_or_neg in ["pos", "neg"], pos_or_neg
    
    preds_out_dir = preds_dir + cell_type + "/"
    return preds_out_dir + "preds." + cell_type + "." + pos_or_neg + ".bigWig"


def get_scores_path(cell_type):
    return preds_dir + cell_type + "/all_TSSs_scored.bed.gz"


def load_scores(cell_type):
    scores_path = get_scores_path(cell_type)
    scores_df = pd.read_csv(scores_path, sep="\t")
    return scores_df





def merge_scores_across_cells(all_scores, mean_or_max, mean_or_max_options = mean_or_max_options):
    assert mean_or_max in mean_or_max_options
    cell_types = list(all_scores.keys())
    
    score_cols = [col for col in all_scores[cell_types[0]].columns if col.startswith("score")]
    nonscore_cols = [col for col in all_scores[cell_types[0]].columns if not col.startswith("score")]
    
    # check that chrom, start, end, capture_tech etc. are all the same across cell types
    for i in range(1, len(cell_types)):
        assert all_scores[cell_types[0]][nonscore_cols].equals(all_scores[cell_types[i]][nonscore_cols]), (cell_types[0], cell_types[i], nonscore_cols, all_scores[cell_types[0]][nonscore_cols], all_scores[cell_types[i]][nonscore_cols])
    
    # columns that are the same (like chromosome) don't need to be "averaged"
    df_cols_to_save = all_scores[cell_types[0]][nonscore_cols]
    
    # these are the columns to actually average
    df_cols_to_merge = [all_scores[cell_type][score_cols] for cell_type in cell_types]
        
    
    if mean_or_max == "mean":
        merged_cols = pd.concat(df_cols_to_merge).groupby(level=0).mean()
    elif mean_or_max == "max":
        merged_cols = pd.concat(df_cols_to_merge).groupby(level=0).max()
    else:
        # the other option is "top3": take average of 3 highest scores
        merged_cols = dict()
        for score_col in score_cols:
            score_series = [all_scores[cell_type][score_col] for cell_type in cell_types]
            # get 3 highest, then take mean
            mean_of_top3 = np.sort(score_series, axis=0)[-3:].mean(axis=0)
            merged_cols[score_col] = mean_of_top3

        merged_cols = pd.DataFrame.from_dict(merged_cols)
        
    full_merged_df = pd.concat([df_cols_to_save, merged_cols], axis=1)
    return full_merged_df
    
    
def write_TSS_scores_to_bed(out_TSS_bed, megatable, extend_bys = extend_bys,
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
        for _, row in megatable.iterrows():
            line = "\t".join([str(thing) for thing in row]) + "\n"
            out_bed.write(line.encode())
                


all_scores = {cell_type : load_scores(cell_type) for cell_type in cell_types}

for mean_or_max in mean_or_max_options:
    merged_scores = merge_scores_across_cells(all_scores, "mean")
    write_TSS_scores_to_bed(out_dir + "all_TSSs_scored." + mean_or_max + "_all_cells.bed.gz", merged_scores)


print("done.")

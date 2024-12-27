import numpy as np
import pandas as pd


### Settings + Constants + Filepath Choices

random_relocate = False

if random_relocate:  # the negative control regions Tamara sent
    preds_dir = "random_relocate_predictions"
    regions_out_dir = "random_relocate_regions_to_predict"
else:
    preds_dir = "predictions"
    regions_out_dir = "regions_to_predict"
    
# all of the candidate TSSs generated from the CLS expts scored
TSSs_filepath = regions_out_dir + "/all_TSSs.bed.gz"

# all cell types we have ProCapNet models for
cell_types = ["K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"]

# all window widths we aggregated predictions over
# (ex: "5" means TSS score is predicted PRO-cap within +/- 5 bp of TSS)
extend_bys = [0,1,2,3,5,10,20,25,50,100,500]

# ways scores were summarized across the cell types
# ("top3" = mean of 3 highest scores)
mean_or_max_options = ["mean", "max", "top3"]




### Filepaths

def get_preds_bw_path(cell_type, pos_or_neg):
    assert pos_or_neg in ["pos", "neg"], pos_or_neg
    
    preds_out_dir = preds_dir + "/" + cell_type + "/"
    return preds_out_dir + "preds." + cell_type + "." + pos_or_neg + ".bigWig"


def get_cell_type_scores_path(cell_type):
    return preds_dir + "/" + cell_type + "/all_TSSs_scored.bed.gz"


def get_merged_scores_path(mean_or_max):
    assert mean_or_max in mean_or_max_options, mean_or_max
    
    if mean_or_max == "mean":
        return preds_dir + "/merged_across_cells/all_TSSs_scored.mean_all_cells.bed.gz"
    elif mean_or_max == "top3":
        return preds_dir + "/merged_across_cells/all_TSSs_scored.top3_all_cells.bed.gz"
    else:
        return preds_dir + "/merged_across_cells/all_TSSs_scored.max_all_cells.bed.gz"
    
    
def get_merged_scores_with_labels_path(mean_or_max):
    assert mean_or_max in mean_or_max_options, mean_or_max
    
    if mean_or_max == "mean":
        return preds_dir + "/merged_across_cells/all_TSSs_scored.mean_all_cells.procapnet_support_labels.bed.gz"
    elif mean_or_max == "top3":
        return preds_dir + "/merged_across_cells/all_TSSs_scored.top3_all_cells.procapnet_support_labels.bed.gz"
    else:
        return preds_dir + "/merged_across_cells/all_TSSs_scored.max_all_cells.procapnet_support_labels.bed.gz"
    
    
    
    
### Load scores for all TSSs, both per-cell-type and aggregated

def load_scores_for_cell_type(cell_type):
    scores_path = get_cell_type_scores_path(cell_type)
    scores_df = pd.read_csv(scores_path, sep="\t")
    return scores_df

def load_scores_merged_cell_types(mean_or_max):
    scores_path = get_merged_scores_path(mean_or_max)
    scores_df = pd.read_csv(scores_path, sep="\t")
    return scores_df

cell_type_scores = {cell_type : load_scores_for_cell_type(cell_type) for cell_type in cell_types}
merged_scores = {mean_or_max : load_scores_merged_cell_types(mean_or_max) for mean_or_max in mean_or_max_options}




### Determine "support labels" for each TSS by thresholding the scores

def get_indexes_passing_thresh(scores_df, extend_by, raw_thresh, norm_thresh=0):
    raw_scores = scores_df["score_window_extended_" + str(extend_by) + "bp"]
    passing = raw_scores >= raw_thresh
    
    if norm_thresh > 0:
        norm_scores = scores_df["score_log_norm_window_extended_" + str(extend_by) + "bp"]
        passing *= norm_scores >= norm_thresh
        
    return np.where(passing)[0]


def label_TSSs_by_support(scores_df):
    # this function decides what label to give each TSS
    
    support_labels = np.array(["no_support"] * len(scores_df), dtype="<U25")
    
    def label(label_str, where):
        for index in where:
            if support_labels[index] == "no_support":
                support_labels[index] = label_str
    
    # ordered from highest-prioritized label to lowest
    # (label() above makes sure higher-priority labels are not overriden by lower-priority labels)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=500)
    label("highest_precise_support", where_support)
            
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=500)
    label("highest_local_support", where_support)
    
    # -----
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=100)
    label("high_precise_support", where_support)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=100)
    label("high_local_support", where_support)

    # -----
    
    # norm score of 0.75 roughly equates to 1/3 of the max pred. PRO-cap for the window
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=20, norm_thresh=0.75)
    label("medium_precise_support", where_support)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=20, norm_thresh=0.75)
    label("medium_local_support", where_support)

    where_support = get_indexes_passing_thresh(scores_df, extend_by=100, raw_thresh=20)
    label("medium_region_support", where_support)
    
    # -----
    
    # 0.9 corresponds to about 60% of the max pred. PRO-cap in the window
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=5, norm_thresh=0.9)
    label("low_precise_support", where_support)

    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=5, norm_thresh=0.9)
    label("low_local_support", where_support)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=100, raw_thresh=5)
    label("low_region_support", where_support)

    # -----
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=0, norm_thresh=0.9)
    label("positioning_only_support", where_support)
    
    return support_labels


support_labels = {mean_or_max : label_TSSs_by_support(merged_scores[mean_or_max]) for mean_or_max in mean_or_max_options}




# same as above, but with +/- 50 bp as widest window, since that seems desirable too
# (I think this is what the CLS team went with in the end)

def label_TSSs_by_support_v2(scores_df):
    # this function decides what label to give each TSS
    
    support_labels = np.array(["no_support"] * len(scores_df), dtype="<U25")
    
    def label(label_str, where):
        for index in where:
            if support_labels[index] == "no_support":
                support_labels[index] = label_str
    
    # ordered from highest-prioritized label to lowest
    # (label() above makes sure higher-priority labels are not overriden by lower-priority labels)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=500)
    label("highest_precise_support", where_support)
            
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=500)
    label("highest_local_support", where_support)
    
    # -----
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=100)
    label("high_precise_support", where_support)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=100)
    label("high_local_support", where_support)

    # -----
    
    # norm score of 0.75 roughly equates to 1/3 of the max pred. PRO-cap for the window
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=20, norm_thresh=0.75)
    label("medium_precise_support", where_support)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=20, norm_thresh=0.75)
    label("medium_local_support", where_support)

    where_support = get_indexes_passing_thresh(scores_df, extend_by=50, raw_thresh=20)
    label("medium_region_support", where_support)
    
    # -----
    
    # 0.9 corresponds to about 60% of the max pred. PRO-cap in the window
    where_support = get_indexes_passing_thresh(scores_df, extend_by=5, raw_thresh=5, norm_thresh=0.9)
    label("low_precise_support", where_support)

    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=5, norm_thresh=0.9)
    label("low_local_support", where_support)
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=50, raw_thresh=5)
    label("low_region_support", where_support)

    # -----
    
    where_support = get_indexes_passing_thresh(scores_df, extend_by=25, raw_thresh=0, norm_thresh=0.9)
    label("positioning_only_support", where_support)
    
    return support_labels


support_labels_v2 = {mean_or_max : label_TSSs_by_support_v2(merged_scores[mean_or_max]) for mean_or_max in mean_or_max_options}




# we make support labels for each of the possible ways of aggregating scores across cell types

for mean_or_max in mean_or_max_options:
    merged_scores[mean_or_max]["ProCapNet_support"] = support_labels[mean_or_max]
    merged_scores[mean_or_max]["ProCapNet_support_v2"] = support_labels_v2[mean_or_max]
    
    
# write to file

for mean_or_max in mean_or_max_options:
    dest_path = get_merged_scores_with_labels_path(mean_or_max)
    merged_scores[mean_or_max].to_csv(dest_path, sep="\t", index=False)
    
print("done.")

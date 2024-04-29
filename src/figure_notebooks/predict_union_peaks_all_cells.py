import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import numpy as np
import torch

sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig, MergedFilesConfig
from data_loading import extract_observed_profiles

sys.path.append("../3_eval_models")
from eval_utils import model_predict_with_rc
from data_loading import extract_sequences

from common_functions import load_coords



def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    model = model.cuda()
    return model


def predict_on_union_peaks(cell_types, timestamps, union_peaks_path,
                           genome_path, chrom_sizes,
                           model_type="strand_merged_umap", data_type="procap", in_window=2114):
    
    
    onehot_seqs = extract_sequences(genome_path, chrom_sizes, union_peaks_path,
                                    in_window=in_window, verbose=True)
    
    pred_logcounts = dict()
    pred_profiles = dict()
    
    for cell_type in cell_types:
        print("Predicting across union peak set for cell type: " + cell_type)
        
        fold_pred_profs = []
        fold_pred_logcounts = []
        for fold_i, timestamp in enumerate(timestamps[cell_type]):
            fold = str(fold_i + 1)
            config = FoldFilesConfig(cell_type, model_type, fold, timestamp, data_type)
        
            model = load_model(config.model_save_path)
            
            pred_profs, pred_logcts = model_predict_with_rc(model, onehot_seqs)
            
            # exp because log-softmax profiles are returned
            fold_pred_profs.append(np.exp(pred_profs))
            fold_pred_logcounts.append(pred_logcts)
    
        # merge by taking mean (after softmax for profiles)
        pred_profiles[cell_type] = np.array(fold_pred_profs).mean(axis=0)
        pred_logcounts[cell_type] = np.array(fold_pred_logcounts).mean(axis=0)
        
    return pred_logcounts, pred_profiles
                                                                   

def save_preds(union_pred_logcounts, union_pred_profiles, save_dir):
    for cell_type in union_pred_logcounts.keys():
        np.save(save_dir + cell_type + "_pred_logcounts.npy", union_pred_logcounts[cell_type])
        np.save(save_dir + cell_type + "_pred_profiles.npy", union_pred_profiles[cell_type])
        
        
if __name__ == "__main__":
    # all sets of models we trained, based on available ENCODE data
    cell_types = ["K562"]#, "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"]

    # the unique IDs for each of the folds / models in each cell type
    timestamps = {"K562" : ["2023-05-29_15-51-40", "2023-05-29_15-58-41", "2023-05-29_15-59-09",
                            "2023-05-30_01-40-06", "2023-05-29_23-21-23", "2023-05-29_23-23-45",
                            "2023-05-29_23-24-11"],
                  "A673" : ["2023-06-11_20-11-32","2023-06-11_23-42-00", "2023-06-12_03-29-06",
                            "2023-06-12_07-17-43", "2023-06-12_11-10-59", "2023-06-12_14-36-40",
                            "2023-06-12_17-26-09"],
                  "CACO2" : ["2023-06-12_21-46-40", "2023-06-13_01-28-24", "2023-06-13_05-06-53",
                             "2023-06-13_08-52-39", "2023-06-13_13-12-09", "2023-06-13_16-40-41",
                             "2023-06-13_20-08-39"],
                  "CALU3" : ["2023-06-14_00-43-44", "2023-06-14_04-26-48", "2023-06-14_09-34-26",
                             "2023-06-14_13-03-59", "2023-06-14_17-22-28", "2023-06-14_21-03-11",
                             "2023-06-14_23-58-36"],
                  "HUVEC" : ["2023-06-16_21-59-35", "2023-06-17_00-20-34", "2023-06-17_02-17-07",
                             "2023-06-17_04-27-08", "2023-06-17_06-42-19", "2023-06-17_09-16-24",
                             "2023-06-17_11-09-38"],
                  "MCF10A" : ["2023-06-15_06-07-40", "2023-06-15_10-37-03", "2023-06-15_16-23-56",
                              "2023-06-15_21-44-32", "2023-06-16_03-47-46", "2023-06-16_09-41-26",
                              "2023-06-16_15-07-01"]}
    
    model_type = "strand_merged_umap"
    data_type = "procap"
    
    in_window = 2114

    any_config = MergedFilesConfig(cell_types[0], model_type, data_type = data_type)
    proj_dir = any_config.proj_dir

    save_dir = proj_dir + "src/figure_notebooks/fig_cell_types_union_preds/"
    os.makedirs(save_dir, exist_ok=True)
    
    union_peaks_path = proj_dir + "data/" + data_type + "/processed/union_peaks.bed.gz"
    
    union_pred_logcounts, union_pred_profiles = predict_on_union_peaks(cell_types,
                                                                       timestamps,
                                                                       union_peaks_path,
                                                                       genome_path=any_config.genome_path,
                                                                       chrom_sizes=any_config.chrom_sizes,
                                                                       model_type=model_type,
                                                                       data_type=data_type,
                                                                       in_window=in_window)
    
    #save_preds(union_pred_logcounts, union_pred_profiles, save_dir)
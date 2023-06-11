import numpy as np
import torch
import pandas as pd
import gzip
import random
import sys

sys.path.append("../2_train_models")
from data_loading import extract_peaks
from performance_metrics import compute_performance_metrics

sys.path.append("../utils")
from write_bigwigs import write_tracks_to_bigwigs
from misc import ensure_parent_dir_exists



def model_predict_with_rc(model, onehot_seqs):
    with torch.no_grad():
        onehot_seqs = torch.tensor(onehot_seqs, dtype=torch.float32).cuda()
        pred_profiles, pred_logcounts = model.predict(onehot_seqs)
        rc_pred_profiles, rc_pred_logcounts = model.predict(torch.flip(onehot_seqs, [-1, -2]))
    
    rc_pred_profiles = rc_pred_profiles[:, ::-1, ::-1]
    
    # take the average prediction across the fwd and RC sequences
    merged_pred_profiles = np.log(np.array([np.exp(pred_profiles), np.exp(rc_pred_profiles)]).mean(axis=0))
    merged_pred_logcounts = np.array([pred_logcounts, rc_pred_logcounts]).mean(axis=0)
    
    return merged_pred_profiles, merged_pred_logcounts


def run_eval(sequence_path, chrom_sizes, plus_bw_path, minus_bw_path, peak_path,
             model_path, pred_profiles_path, pred_logcounts_path,
             metrics_save_path, log_save_path, in_window=2114, out_window=1000,
             stranded=False, save=True):

    to_print = "=== Running Model Eval ==="
    to_print += "\nBigwigs:\n   - " + plus_bw_path + "\n   - " + minus_bw_path
    to_print += "\nPeaks: " + peak_path
    to_print += "\nModel: " + model_path
    to_print += "\nSequence length: " + str(in_window)
    to_print += "\nProfile length: " + str(out_window)
    to_print += "\nStranded model: " + str(stranded)
    
    to_print += "\nPred. profiles (out): " + pred_profiles_path
    to_print += "\nPred. counts (out): " + pred_logcounts_path
    to_print += "\nEval metrics (out): " + metrics_save_path
    to_print += "\nEval log (out): " + log_save_path
    
    if save:
        for filepath in [pred_profiles_path, pred_logcounts_path, metrics_save_path, log_save_path]:
            ensure_parent_dir_exists(filepath)
    
    # load model
    model = torch.load(model_path)
    model.eval()
    model = model.cuda()

    # load data for peak set
    onehot_seqs, true_profiles = extract_peaks(sequence_path, chrom_sizes,
        plus_bw_path, minus_bw_path, peak_path,
        in_window=in_window, out_window=out_window,
        max_jitter=0, verbose=True)
    
    # make predictions
    pred_profiles, pred_logcounts = model_predict_with_rc(model, onehot_seqs)
    
    # save predictions to files
    if save:
        np.save(pred_profiles_path, pred_profiles)
        np.save(pred_logcounts_path, pred_logcounts)
    
        scaled_pred_profiles = np.exp(pred_profiles) * np.exp(pred_logcounts)[..., None]
        write_tracks_to_bigwigs(scaled_pred_profiles, peak_path,
                                pred_profiles_path, chrom_sizes)

    # re-format arrays for performance metrics code
    if stranded:
        true_counts = np.expand_dims(true_profiles.sum(axis=2), 1) 
        true_profiles = np.expand_dims(np.swapaxes(true_profiles, 1, 2), 1)
        
        pred_profiles = np.expand_dims(np.swapaxes(pred_profiles, 1, 2), 1)
        pred_logcounts = np.expand_dims(pred_logcounts, 1)
    else:
        true_profiles = true_profiles.reshape(true_profiles.shape[0], -1)
        true_profiles = np.expand_dims(true_profiles, (1, 3))
        true_counts = true_profiles.sum(axis=2)
        
        pred_profiles = pred_profiles.reshape(pred_profiles.shape[0], -1)
        pred_profiles = np.expand_dims(pred_profiles, (1, 3))
        pred_logcounts = np.expand_dims(pred_logcounts, 1)

    # compute metrics
    metrics = compute_performance_metrics(true_profiles, pred_profiles, 
        true_counts, pred_logcounts, smooth_true_profs=False, smooth_pred_profs=False)

    # save metrics and log results
    df_dict = {metric : list(metrics[metric].squeeze()) for metric in ["nll", "jsd", "profile_pearson"]}
    metrics_df = pd.DataFrame(df_dict)
    if save:
        metrics_df.to_csv(metrics_save_path, sep="\t", index=False)

    metrics_to_report = ["nll", "jsd", "profile_pearson", "count_pearson", "count_mse"]
    metrics_summary = [str(metrics[metric].mean()) for metric in metrics_to_report] 

    to_log = ["Peaks: " + peak_path]
    to_log += ["Model: " + model_path]
    to_log += ["Pred_profiles: " + pred_profiles_path]
    to_log += ["Pred_counts: " + pred_logcounts_path]
    to_log += ["Val_metrics: " + metrics_save_path]
    for metric, val in zip(metrics_to_report, metrics_summary):
        to_log += ["Mean " + metric + ": " + val]
        print("Mean " + metric + ": " + val)

    if save:
        with open(log_save_path, "w") as logf:
            logf.write("\n".join(to_log) + "\n")
            
        
    
    
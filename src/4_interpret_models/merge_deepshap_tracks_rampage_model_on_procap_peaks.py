import sys
import numpy as np

sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig as FilesConfig
from file_configs import MergedFilesConfig

sys.path.append("../utils")
from misc import ensure_parent_dir_exists
from write_bigwigs import write_scores_to_bigwigs


def main(cell_type, model_type, timestamps):
    # load in all predictions made by individual models
    prof_scores = []
    prof_scores_onehot = []
    counts_scores = []
    counts_scores_onehot = []
    
    for timestamp in timestamps:
        # (fold doesn't matter here, files named just by timestamp)
        config = FilesConfig(cell_type, model_type, "1", timestamp = timestamp, data_type = "rampage")
        
        # exp here, take log after merge
        prof_scores.append(np.load(config.profile_scores_path.replace("all_", "all_procap_peaks_")))
        prof_scores_onehot.append(np.load(config.profile_onehot_scores_path.replace("all_", "all_procap_peaks_")))
        counts_scores.append(np.load(config.counts_scores_path.replace("all_", "all_procap_peaks_")))
        counts_scores_onehot.append(np.load(config.counts_onehot_scores_path.replace("all_", "all_procap_peaks_")))
        
    merged_config = MergedFilesConfig(cell_type, model_type, data_type = "rampage")
    merged_config_procap = MergedFilesConfig(cell_type, model_type, data_type = "procap")
    
    # merge by taking mean (after softmax for profiles)
    merged_prof_scores = np.array(prof_scores).mean(axis=0)
    merged_prof_scores_onehot = np.array(prof_scores_onehot).mean(axis=0)
    merged_counts_scores = np.array(counts_scores).mean(axis=0)
    merged_counts_scores_onehot = np.array(counts_scores_onehot).mean(axis=0)
    
    # save
    ensure_parent_dir_exists(merged_config.profile_scores_path)
    np.save(merged_config.profile_scores_path.replace("all_", "all_procap_peaks_"), merged_prof_scores)
    np.save(merged_config.profile_onehot_scores_path.replace("all_", "all_procap_peaks_"), merged_prof_scores_onehot)
    np.save(merged_config.counts_scores_path.replace("all_", "all_procap_peaks_"), merged_counts_scores)
    np.save(merged_config.counts_onehot_scores_path.replace("all_", "all_procap_peaks_"), merged_counts_scores_onehot)
    
    write_scores_to_bigwigs(np.sum(merged_prof_scores_onehot, axis = 1),
                            merged_config_procap.all_peak_path,
                            merged_config.profile_scores_path.replace("all_", "all_procap_peaks_"),
                            merged_config.chrom_sizes)
    
    write_scores_to_bigwigs(np.sum(merged_counts_scores_onehot, axis = 1),
                            merged_config_procap.all_peak_path,
                            merged_config.counts_scores_path.replace("all_", "all_procap_peaks_"),
                            merged_config.chrom_sizes)
    
    
if __name__ == "__main__":
    assert len(sys.argv) == 4, len(sys.argv)

    cell_type, model_type = sys.argv[1:3]
    timestamps = sys.argv[3].split()
    
    main(cell_type, model_type, timestamps)
    

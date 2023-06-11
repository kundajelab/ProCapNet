import sys
import numpy as np

sys.path.append("../2_train_models")
from file_configs import FoldFilesConfig as FilesConfig
from file_configs import MergedFilesConfig

sys.path.append("../utils")
from misc import ensure_parent_dir_exists
from write_bigwigs import write_tracks_to_bigwigs


def main(cell_type, model_type, timestamps, data_type):
    # load in all predictions made by individual models
    pred_profiles = []
    pred_logcounts = []
    
    for timestamp in timestamps:
        # (fold doesn't matter here, files named just by timestamp)
        config = FilesConfig(cell_type, model_type, "1", timestamp = timestamp, data_type = data_type)
        
        # exp here, take log after merge
        pred_profiles.append(np.exp(np.load(config.pred_profiles_all_path)))
        pred_logcounts.append(np.load(config.pred_logcounts_all_path))
        
    merged_config = MergedFilesConfig(cell_type, model_type, data_type = data_type)
    
    # merge by taking mean (after softmax for profiles)
    merged_pred_profiles = np.array(pred_profiles).mean(axis=0)
    merged_pred_logcounts = np.array(pred_logcounts).mean(axis=0)
    
    # save
    ensure_parent_dir_exists(merged_config.pred_profiles_all_path)
    np.save(merged_config.pred_profiles_all_path, np.log(merged_pred_profiles))
    np.save(merged_config.pred_logcounts_all_path, merged_pred_logcounts)
    
    scaled_pred_profiles = merged_pred_profiles * np.exp(merged_pred_logcounts)[..., None]
    write_tracks_to_bigwigs(scaled_pred_profiles, merged_config.all_peak_path,
                            merged_config.pred_profiles_all_path, merged_config.chrom_sizes)
    
    
if __name__ == "__main__":
    assert len(sys.argv) == 4, len(sys.argv)

    cell_type, model_type = sys.argv[1:3]
    timestamps = sys.argv[3].split()
    data_type = "procap"
    
    main(cell_type, model_type, timestamps, data_type)
    
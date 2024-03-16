import os
from datetime import datetime
import json
import shutil

# this file exists to consolidate all hardcoded filepaths into one place


# what model types are implemented (this must track with options allowed in train.py)
MODEL_TYPES = ["strand_merged_umap", "promoters_only_strand_merged_umap", "strand_merged_umap_replicate"]
    
    
class GeneralFilesConfig():
    def __init__(self, cell_type, model_type, data_type = "procap"):
        
        ## Parse inputs, figure out root directory
        
        assert model_type in MODEL_TYPES, model_type
        
        self.cell_type = cell_type
        self.model_type = model_type
        self.data_type = data_type
        
        self.stranded_model = "stranded" in self.model_type
        self.umap = "umap" in self.model_type
        
        # proj_dir is the root of the git repository, one level above src/
        # (don't move this file or this will break)
        self.proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/"     
        
        ## Store filepaths to everything
        
        # Genome files and annotations
        
        self.genome_path = self.proj_dir + "genomes/hg38.withrDNA.fasta"
        self.chrom_sizes = self.proj_dir + "genomes/hg38.withrDNA.chrom.sizes"
        
        for filepath in [self.genome_path, self.chrom_sizes]:
            assert os.path.exists(filepath), filepath
        
        if self.umap:
            self.mask_bw_path = self.proj_dir + "/annotations/hg38.k36.multiread.umap.bigWig"
            assert os.path.exists(self.mask_bw_path), self.mask_bw_path
        else:
            self.mask_bw_path = None
            
        
        # Data files (peaks, bigWigs)
        
        self.data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed", self.cell_type)) + "/"
        
        self.all_peak_path = self.data_dir + "peaks.bed.gz"
        self.plus_bw_path = self.data_dir + "5prime.pos.bigWig"
        self.minus_bw_path = self.data_dir + "5prime.neg.bigWig"

        for filepath in [self.all_peak_path, self.plus_bw_path, self.minus_bw_path]:
            
            assert os.path.exists(filepath), filepath

            
        # Random Modisco param
        
        self.slice = 1000
        

    def save_config(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
            
        os.makedirs(self.configs_dir, exist_ok=True)
        
        with open(self.config_path, "w") as json_file:
            json.dump(self.__dict__, json_file)


    def load_model_params(self):
        with open(self.params_path) as f:
            model_params = json.load(f)
        return model_params["in_window"], model_params["out_window"] 

    
    
class FoldFilesConfig(GeneralFilesConfig):
    def __init__(self, cell_type, model_type, fold, timestamp = None, data_type = "procap"):
        
        super().__init__(cell_type, model_type, data_type)
        
        self.fold = fold
        
        # timestamp should be None when training a new model, otherwise should use existing
        # serves as unique identifier for a model and all downstream analysis
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.timestamp = timestamp
        print("Timestamp: " + self.timestamp)
        
        
        ## Store filepaths to everything
            
        
        # Data files (peaks, bigWigs)
        
        self.train_peak_path = self.data_dir + "peaks_fold" + fold + "_train.bed.gz"
        self.val_peak_path = self.data_dir + "peaks_fold" + fold + "_val.bed.gz"
        self.test_peak_path = self.data_dir + "peaks_fold" + fold + "_test.bed.gz"
        self.train_val_peak_path = self.data_dir + "peaks_fold" + fold + "_train_and_val.bed.gz"
        
        self.dnase_train_path = self.data_dir + "dnase_peaks_no_" + data_type + "_overlap_fold" + fold + "_train.bed.gz"

        for filepath in [self.train_peak_path,
                         self.val_peak_path,
                         self.test_peak_path,
                         self.train_val_peak_path,
                         self.dnase_train_path]:
            
            assert os.path.exists(filepath), filepath
            
        
        # Model save files
        
        self.model_dir = self.proj_dir + "/".join(("models", self.data_type, self.cell_type, self.model_type)) + "/"
        
        self.model_save_path = self.model_dir + self.timestamp + ".model"
        
        self.params_path = self.model_dir + self.timestamp + "_params.json"
        self.arch_path = self.model_dir + self.timestamp + "_model_arch.txt"
        
        self.configs_dir = self.proj_dir + "/".join(("configs", self.data_type, self.cell_type, self.model_type)) + "/"
        
        self.config_path = self.configs_dir + self.timestamp + ".json"

        
        # Model evaluation files

        val_dir = self.proj_dir + "/".join(("model_out", self.data_type, self.cell_type, self.model_type, self.timestamp)) + "/"

        self.pred_profiles_all_path = val_dir + "all_pred_profiles.npy"
        self.pred_logcounts_all_path = val_dir + "all_pred_logcounts.npy"
        self.pred_profiles_test_path = val_dir + "test_pred_profiles.npy"
        self.pred_logcounts_test_path = val_dir + "test_pred_logcounts.npy"
        
        self.metrics_all_path = val_dir + "all_metrics.txt"
        self.metrics_test_path = val_dir + "test_metrics.txt"
        
        self.log_all_path = val_dir + "all_run_log.txt"
        self.log_test_path = val_dir + "test_run_log.txt"
        
        # DeepSHAP files
        
        shap_dir = self.proj_dir + "/".join(("deepshap_out", self.data_type, self.cell_type, self.model_type, self.timestamp)) + "/"
        
        self.profile_scores_path = shap_dir + "all_profile_deepshap.npy"
        self.profile_onehot_scores_path = shap_dir + "all_profile_deepshap_onehot.npy"
        
        self.counts_scores_path = shap_dir + "all_counts_deepshap.npy"
        self.counts_onehot_scores_path = shap_dir + "all_counts_deepshap_onehot.npy"
        
        # Modisco files, params
        
        # You won't actually run modisco on models from individual folds
        # but I'll leave this here anyways
        
        modisco_dir = self.proj_dir + "/".join(("modisco_out", self.data_type, self.cell_type, self.model_type, self.timestamp)) + "/"
        
        # NEW: use profile_scores_path or counts_scores_path instead of "scores_path"
        # NEW: no more results_save_path, use these instead
        self.modisco_profile_results_path = modisco_dir + "profile_modisco_results.hd5"
        self.modisco_counts_results_path = modisco_dir + "counts_modisco_results.hd5"
        
        # Motif calling files
        
        # NEW: use profile_scores_path or counts_scores_path instead of "scores_path"
        # NEW: no more results_save_path, use these instead
        self.refmt_modisco_profile_results_path = modisco_dir + "old_fmt_profile_modisco_results.hd5"
        self.refmt_modisco_counts_results_path = modisco_dir + "old_fmt_counts_modisco_results.hd5"

        motifs_dir = self.proj_dir + "/".join(("motif_calls_out", self.data_type, self.cell_type, self.model_type, self.timestamp)) + "/"
        
        self.profile_hits_path = motifs_dir + "profile_motif_hits.bed"
        self.counts_hits_path = motifs_dir + "counts_motif_hits.bed"

        
        
class MergedFilesConfig(GeneralFilesConfig):
    def __init__(self, cell_type, model_type, data_type = "procap"):
        
        super().__init__(cell_type, model_type, data_type)
        
        ## Store filepaths to everything
        
        # Model evaluation files

        val_dir = self.proj_dir + "/".join(("model_out", self.data_type, self.cell_type, self.model_type, "merged")) + "/"

        self.pred_profiles_all_path = val_dir + "all_pred_profiles.npy"
        self.pred_logcounts_all_path = val_dir + "all_pred_logcounts.npy"
        
        # DeepSHAP files
        
        shap_dir = self.proj_dir + "/".join(("deepshap_out", self.data_type, self.cell_type, self.model_type, "merged")) + "/"
        
        self.profile_scores_path = shap_dir + "all_profile_deepshap.npy"
        self.profile_onehot_scores_path = shap_dir + "all_profile_deepshap_onehot.npy"
        
        self.counts_scores_path = shap_dir + "all_counts_deepshap.npy"
        self.counts_onehot_scores_path = shap_dir + "all_counts_deepshap_onehot.npy"
        
        # Modisco files, params
        
        modisco_dir = self.proj_dir + "/".join(("modisco_out", self.data_type, self.cell_type, self.model_type, "merged")) + "/"
        
        self.modisco_profile_results_path = modisco_dir + "profile_modisco_results.hd5"
        self.modisco_counts_results_path = modisco_dir + "counts_modisco_results.hd5"
        
        # Motif calling files
        
        self.refmt_modisco_profile_results_path = modisco_dir + "old_fmt_profile_modisco_results.hd5"
        self.refmt_modisco_counts_results_path = modisco_dir + "old_fmt_counts_modisco_results.hd5"

        motifs_dir = self.proj_dir + "/".join(("motifs_out", self.data_type, self.cell_type, self.model_type, "merged")) + "/"
        
        self.profile_hits_path = motifs_dir + "profile_hits.bed"
        self.counts_hits_path = motifs_dir + "counts_hits.bed"

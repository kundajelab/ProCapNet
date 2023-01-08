from utils import get_proj_dir
import os
from datetime import datetime
import json
import shutil


# what model types are implemented (this must track with options allowed in train.py)
MODEL_TYPES = ["strand_merged_umap", "stranded_umap", "strand_merged", "stranded", "EMD_strand_merged", "EMD_and_multinomial_strand_merged"]


class FilesConfig():
    def __init__(self, cell_type, model_type, data_type = "procap"):
        
        assert model_type in MODEL_TYPES, model_type
        
        self.cell_type = cell_type
        self.model_type = model_type
        self.data_type = data_type
        
        self.stranded_model = "stranded" in self.model_type
        self.umap = "umap" in self.model_type
        
        self.proj_dir = get_proj_dir()
        
        # Genome files and annotations
        
        self.genome_path = self.proj_dir + "genomes/hg38.withrDNA.fasta"
        self.chrom_sizes = self.proj_dir + "genomes/hg38.withrDNA.chrom.sizes"
        
        assert os.path.exists(self.genome_path), self.genome_path
        assert os.path.exists(self.chrom_sizes), self.chrom_sizes

        
    def copy_input_files(self):
        if os.path.exists(self.inputs_dir):
            shutil.rmtree(self.inputs_dir)
            
        os.makedirs(self.inputs_dir, exist_ok=True)
        
        for original_filepath in self.input_files_to_copy:
            filename = os.path.basename(original_filepath)
            dest_filepath = os.path.join(self.inputs_dir, filename)
            shutil.copy2(original_filepath, dest_filepath)


    def save_config(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
            
        os.makedirs(self.inputs_dir, exist_ok=True)
        
        with open(self.config_path, "w") as json_file:
            json.dump(self.__dict__, json_file)


    def load_model_params(self, model_params_path):
        with open(model_params_path) as f:
            model_params = json.load(f)
        return model_params["in_window"], model_params["out_window"]

    

class TrainFilesConfig(FilesConfig):
    def __init__(self, cell_type, model_type, timestamp = None, data_type = "procap"):
        
        super().__init__(cell_type, model_type, data_type)
        
        if self.umap:
            self.mask_bw_path = self.proj_dir + "/annotations/hg38.k36.multiread.umap.bigWig"
            assert os.path.exists(self.mask_bw_path), self.mask_bw_path
        else:
            self.mask_bw_path = None
            
        # Data files (peaks, bigWigs)
        
        data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed", self.cell_type)) + "/"
        self.train_peak_path = data_dir + "peaks_uni_and_bi_train.bed.gz"
        self.val_peak_path = data_dir + "peaks_uni_and_bi_val.bed.gz"
        self.plus_bw_path = data_dir + "5prime.pos.bigWig"
        self.minus_bw_path = data_dir + "5prime.neg.bigWig"
        
        assert os.path.exists(self.train_peak_path), self.train_peak_path
        assert os.path.exists(self.val_peak_path), self.val_peak_path
        assert os.path.exists(self.plus_bw_path), self.plus_bw_path
        assert os.path.exists(self.minus_bw_path), self.minus_bw_path
        
        # Filepaths for locations to save model and various log files

        save_dir = self.proj_dir + "/".join(("models", self.data_type, self.cell_type, self.model_type)) + "/"

        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.timestamp = timestamp
        print("Timestamp: " + self.timestamp)
        
        # Filepaths for saving inputs
        
        self.inputs_dir = save_dir + self.timestamp + "_in/"
        
        self.config_path = self.inputs_dir + "config.json"
        self.params_path = self.inputs_dir + "params.json"
        self.arch_path = self.inputs_dir + "model_arch.txt"
        
        self.input_files_to_copy = [self.chrom_sizes,
                                    self.train_peak_path,
                                    self.val_peak_path,
                                    self.plus_bw_path,
                                    self.minus_bw_path]
        
        if self.umap:
            self.input_files_to_copy += [self.mask_bw_path]
        
        # Filepaths for saving outputs
        
        self.model_save_path = save_dir + self.timestamp + ".model"
        

class TrainWithCpGMatchedNegsFilesConfig(TrainFilesConfig):
    def __init__(self, cell_type, model_type, timestamp = None, data_type = "procap"):
        
        super().__init__(cell_type, model_type, timestamp = timestamp, data_type = data_type)

        data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed", self.cell_type)) + "/"
        self.train_peak_path = data_dir + "peaks_uni_and_bi_train_and_val_cpg_matched_train_peaks_and_matched.bed.gz"

        
class TrainWithUnionPeaksFilesConfig(TrainFilesConfig):
    def __init__(self, cell_type, model_type, timestamp = None, data_type = "procap"):
        
        super().__init__(cell_type, model_type, timestamp = timestamp, data_type = data_type)

        # not cell-type-specific
        data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed")) + "/"
        self.train_peak_path = data_dir + "union_train_peaks_across_cell_types.bed.gz"


class TrainWithDNasePeaksFilesConfig(TrainFilesConfig):
    def __init__(self, cell_type, model_type, timestamp = None, data_type = "procap"):
        
        super().__init__(cell_type, model_type, timestamp = timestamp, data_type = data_type)

        # not cell-type-specific
        data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed", self.cell_type)) + "/"
        self.train_peak_paths = [data_dir + "peaks_uni_and_bi_train.bed.gz",
                                data_dir + "dnase_peaks_no_train_peaks_train.bed.gz"]
        self.source_fracs = [0.5, 0.5]
        
        
class ValFilesConfig(FilesConfig):
    def __init__(self, cell_type, model_type, timestamp, data_type = "procap"):
        
        super().__init__(cell_type, model_type, data_type)
        
        self.timestamp = timestamp
        print("Timestamp: " + self.timestamp)

        # Data files (peaks, bigWigs)
        
        model_dir = self.proj_dir + "/".join(("models", self.data_type, self.cell_type, self.model_type)) + "/"
        model_inputs_dir = model_dir + self.timestamp + "_in/"

        self.val_peak_path = model_inputs_dir + "peaks_uni_and_bi_val.bed.gz"
        self.plus_bw_path = model_inputs_dir + "5prime.pos.bigWig"
        self.minus_bw_path = model_inputs_dir + "5prime.neg.bigWig"
        
        assert os.path.exists(self.val_peak_path), self.val_peak_path
        assert os.path.exists(self.plus_bw_path), self.plus_bw_path
        assert os.path.exists(self.minus_bw_path), self.minus_bw_path
        
        self.in_window, self.out_window = self.load_model_params(model_inputs_dir + "params.json")
        
        data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed", self.cell_type)) + "/"
        self.train_val_peak_path = data_dir + "peaks_uni_and_bi_train_and_val.bed.gz"
        
        assert os.path.exists(self.train_val_peak_path), self.train_val_peak_path
        
        # Saved model file
        
        self.model_save_path = model_dir + self.timestamp + ".model"
        
        assert os.path.exists(self.model_save_path), self.model_save_path
        
        # Directory to save everything into

        save_dir = self.proj_dir + "/".join(("model_out", self.data_type, self.cell_type, self.model_type)) + "/"
        
        # Filepaths for saving input files
        
        self.inputs_dir = save_dir + self.timestamp + "_in/"
        
        self.config_path = save_dir + self.timestamp + "_in/config.json"

        self.input_files_to_copy = [self.chrom_sizes,
                                    self.train_val_peak_path,
                                    self.val_peak_path,
                                    self.plus_bw_path,
                                    self.minus_bw_path,
                                    self.model_save_path]
        
        # Filepaths for saving outputs and various log files

        self.outputs_dir = save_dir + self.timestamp + "_out/"
        
        #TODO: .npz?
        self.pred_profiles_train_val_path = self.outputs_dir + "train_and_val_pred_profiles.npy"
        self.pred_logcounts_train_val_path = self.outputs_dir + "train_and_val_pred_logcounts.npy"
        self.pred_profiles_val_path = self.outputs_dir + "val_pred_profiles.npy"
        self.pred_logcounts_val_path = self.outputs_dir + "val_pred_logcounts.npy"
        
        self.metrics_train_val_path = self.outputs_dir + "train_and_val_metrics.txt"
        self.metrics_val_path = self.outputs_dir + "val_metrics.txt"
        
        self.log_train_val_path = self.outputs_dir + "train_and_val_run_log.txt"
        self.log_val_path = self.outputs_dir + "val_run_log.txt"
    
    
    
    
class DeepshapFilesConfig(FilesConfig):
    def __init__(self, cell_type, model_type, timestamp, data_type = "procap"):
        
        super().__init__(cell_type, model_type, data_type)
        
        self.timestamp = timestamp
        print("Timestamp: " + self.timestamp)

        # Data files (peaks, bigWigs)
        
        model_dir = self.proj_dir + "/".join(("models", self.data_type, self.cell_type, self.model_type)) + "/"
        model_inputs_dir = model_dir + self.timestamp + "_in/"

        self.plus_bw_path = model_inputs_dir + "5prime.pos.bigWig"
        self.minus_bw_path = model_inputs_dir + "5prime.neg.bigWig"
        
        assert os.path.exists(self.plus_bw_path), self.plus_bw_path
        assert os.path.exists(self.minus_bw_path), self.minus_bw_path
        
        self.in_window, self.out_window = self.load_model_params(model_inputs_dir + "params.json")
        
        data_dir = self.proj_dir + "/".join(("data", self.data_type, "processed", self.cell_type)) + "/"
        self.train_val_peak_path = data_dir + "peaks_uni_and_bi_train_and_val.bed.gz"
        
        assert os.path.exists(self.train_val_peak_path), self.train_val_peak_path
        
        # Saved model file
        
        self.model_save_path = model_dir + self.timestamp + ".model"
        
        assert os.path.exists(self.model_save_path), self.model_save_path
        
        # Directory to save everything into

        save_dir = self.proj_dir + "/".join(("deepshap_out", self.data_type, self.cell_type, self.model_type)) + "/"
        
        # Filepaths for saving input files
        
        self.inputs_dir = save_dir + self.timestamp + "_in/"
        
        self.config_path = save_dir + self.timestamp + "_in/config.json"

        self.input_files_to_copy = [self.chrom_sizes,
                                    self.train_val_peak_path,
                                    self.model_save_path]
        
        # Filepaths for saving outputs and various log files

        self.outputs_dir = save_dir + self.timestamp + "_out/"
        
        #TODO: .npz?
        self.profile_scores_path = self.outputs_dir + "train_and_val_profile_deepshap.npy"
        self.profile_onehot_scores_path = self.outputs_dir + "train_and_val_profile_deepshap_onehot.npy"
        
        self.counts_scores_path = self.outputs_dir + "train_and_val_counts_deepshap.npy"
        self.counts_onehot_scores_path = self.outputs_dir + "train_and_val_counts_deepshap_onehot.npy"
    
    
    
class ModiscoFilesConfig(FilesConfig):
    def __init__(self, cell_type, model_type, timestamp, task, data_type = "procap"):
        
        super().__init__(cell_type, model_type, data_type)
        
        self.timestamp = timestamp
        print("Timestamp: " + self.timestamp)
        
        assert task in ["profile", "counts"], task
        self.task = task
        
        self.slice = 1000

        # Data files (peaks, hypothetical scores)
        
        model_dir = self.proj_dir + "/".join(("models", self.data_type, self.cell_type, self.model_type)) + "/"
        model_inputs_dir = model_dir + self.timestamp + "_in/"
        
        self.in_window, self.out_window = self.load_model_params(model_inputs_dir + "params.json")
        
        deepshap_dir = self.proj_dir + "/".join(("deepshap_out", self.data_type, self.cell_type, self.model_type)) + "/"
        deepshap_inputs_dir = deepshap_dir + self.timestamp + "_in/"
        
        self.train_val_peak_path = deepshap_inputs_dir + "peaks_uni_and_bi_train_and_val.bed.gz"
        
        assert os.path.exists(self.train_val_peak_path), self.train_val_peak_path
        
        deepshap_outputs_dir = deepshap_dir + self.timestamp + "_out/"
        
        if self.task == "profile":
            self.scores_path = deepshap_outputs_dir + "train_and_val_profile_deepshap.npy"
        else:
            self.scores_path = deepshap_outputs_dir + "train_and_val_counts_deepshap.npy"
        
        # Directory to save everything into

        save_dir = self.proj_dir + "/".join(("modisco_out", self.data_type, self.cell_type, self.model_type)) + "/"
        
        # Filepaths for saving input files
        
        self.inputs_dir = save_dir + self.timestamp + "_" + task + "_in/"
        
        self.config_path = save_dir + self.timestamp + "_" + task + "_in/config.json"

        self.input_files_to_copy = [self.chrom_sizes,
                                    self.train_val_peak_path,
                                    self.scores_path]
        
        # Filepaths for saving outputs and various log files

        self.outputs_dir = save_dir + self.timestamp + "_" + task + "_out/"
        
        self.results_save_path = self.outputs_dir + "modisco_results.hd5"
        
        
    
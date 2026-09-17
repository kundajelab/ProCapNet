import os
from datetime import datetime
import json
import shutil
from file_configs import FoldFilesConfig, MergedFilesConfig

# this file exists to consolidate all hardcoded filepaths into one place


# what model types are implemented (this must track with options allowed in train.py)
MODEL_TYPES = ["strand_merged_umap", "promoters_only_strand_merged_umap", "strand_merged_umap_replicate",
               "cpg_matched_negs_strand_merged_umap", "cell_union_negs_strand_merged_umap"]
    
    
    
class CpGMatchNegsFoldFilesConfig(FoldFilesConfig):
    def __init__(self, cell_type, model_type, fold, timestamp = None, data_type = "procap"):
        
        if "cpg_matched_negs" not in model_type:
            model_type = "cpg_matched_negs_" + model_type
        
        super().__init__(cell_type, model_type, fold, timestamp = timestamp, data_type = data_type)
        
        # ok. is this actually "dnase"? no. but if I keep the variable name the same, life is easier.
        self.dnase_train_path = self.data_dir + "peaks_cpg_matched_fold" + fold + "_train.bed.gz"

        assert os.path.exists(self.dnase_train_path), filepath
        
        
class CpGMatchNegsMergedFilesConfig(MergedFilesConfig):
    def __init__(self, cell_type, model_type, data_type = "procap"):
        
        if "cpg_matched_negs" not in model_type:
            model_type = "cpg_matched_negs_" + model_type
        
        super().__init__(cell_type, model_type, data_type)
        

class CellUnionNegsFoldFilesConfig(FoldFilesConfig):
    def __init__(self, cell_type, model_type, fold, timestamp = None, data_type = "procap"):
        
        if "cell_union_negs" not in model_type:
            model_type = "cell_union_negs_" + model_type
        
        super().__init__(cell_type, model_type, fold, timestamp = timestamp, data_type = data_type)
        
        # ok. is this actually "dnase"? no. but if I keep the variable name the same, life is easier.
        self.dnase_train_path = self.data_dir + "../union_other_peaks_fold" + fold + "_train.bed.gz"

        assert os.path.exists(self.dnase_train_path), filepath
        
        
class CellUnionNegsMergedFilesConfig(MergedFilesConfig):
    def __init__(self, cell_type, model_type, data_type = "procap"):
        
        if "cell_union_negs" not in model_type:
            model_type = "cell_union_negs_" + model_type
        
        super().__init__(cell_type, model_type, data_type)
from file_configs import FoldFilesConfig, MergedFilesConfig
import os
    
    
class PromotersOnlyFoldFilesConfig(FoldFilesConfig):
    def __init__(self, cell_type, model_type, fold, timestamp = None, data_type = "procap"):
        
        if "promoters_only" not in model_type:
            model_type = "promoters_only_" + model_type
        
        super().__init__(cell_type, model_type, fold, timestamp = timestamp, data_type = data_type)
        
        self.all_peak_path = self.data_dir + "peaks_promoters_only.bed.gz"
        assert os.path.exists(self.all_peak_path), self.all_peak_path
        
        self.train_peak_path = self.data_dir + "peaks_promoters_only_fold" + fold + "_train.bed.gz"
        self.val_peak_path = self.data_dir + "peaks_promoters_only_fold" + fold + "_val.bed.gz"
        self.test_peak_path = self.data_dir + "peaks_promoters_only_fold" + fold + "_test.bed.gz"
        self.train_val_peak_path = self.data_dir + "peaks_promoters_only_fold" + fold + "_train_and_val.bed.gz"
        
        for filepath in [self.train_peak_path,
                         self.val_peak_path,
                         self.test_peak_path,
                         self.train_val_peak_path]:
            
            assert os.path.exists(filepath), filepath

        
class PromotersOnlyMergedFilesConfig(MergedFilesConfig):
    def __init__(self, cell_type, model_type, data_type = "procap"):
        
        if "promoters_only" not in model_type:
            model_type = "promoters_only_" + model_type
        
        super().__init__(cell_type, model_type, data_type = data_type)
        
        self.all_peak_path = self.data_dir + "peaks_promoters_only.bed.gz"
        assert os.path.exists(self.all_peak_path), self.all_peak_path

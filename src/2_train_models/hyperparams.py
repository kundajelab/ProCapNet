import json


class DefaultParams():
    def __init__(self):
        self.n_filters = 512
        self.n_layers = 8
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.counts_weight = 100
        self.max_epochs = 500
        self.val_iters = 100
        self.early_stop_epochs = 10

        self.in_window = 2114
        self.out_window = 1000
        self.trimming = (self.in_window - self.out_window) // 2
        self.max_jitter = 200
        
        self.source_fracs = [0.875, 0.125]
    
    
    def save_config(self, filepath):
        with open(filepath, "w") as json_file:
            json.dump(self.__dict__, json_file) 


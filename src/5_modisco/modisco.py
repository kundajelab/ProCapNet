import os, sys
assert len(sys.argv) == 5, len(sys.argv)  # expecting celltype, model_type, timestamp, task
cell_type, model_type, timestamp, task = sys.argv[1:]


sys.path.append("../2_train_models")
from file_configs import ModiscoFilesConfig
config = ModiscoFilesConfig(cell_type, model_type, timestamp, task)

config.copy_input_files()
config.save_config()


from modiscolite_utils import modisco


print("Running modisco...")

modisco(config.genome_path,
        config.chrom_sizes,
        config.train_val_peak_path,
        config.scores_path,
        config.slice,
        config.in_window,
        config.results_save_path)


print("Done running modisco.")





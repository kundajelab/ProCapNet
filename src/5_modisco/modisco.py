import os, sys
assert len(sys.argv) == 4, len(sys.argv)  # expecting celltype, model_type, task
cell_type, model_type, task = sys.argv[1:]

data_type = "procap"
in_window = 2114
print("Using in_window size of " + str(in_window) + ".")

assert task in ["profile", "counts"]

sys.path.append("../2_train_models")
from file_configs import MergedFilesConfig
config = MergedFilesConfig(cell_type, model_type, data_type)


from modiscolite_utils import modisco


print("Running modisco (" + task + " task)...")

if task == "profile":
    modisco(config.genome_path,
            config.chrom_sizes,
            config.all_peak_path,
            config.profile_scores_path,
            config.slice,
            in_window,
            config.modisco_profile_results_path)
else:
    modisco(config.genome_path,
            config.chrom_sizes,
            config.all_peak_path,
            config.counts_scores_path,
            config.slice,
            in_window,
            config.modisco_counts_results_path)


print("Done running modisco.")





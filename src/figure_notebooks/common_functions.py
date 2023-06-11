import numpy as np
import gzip

def get_orientation_indexes(profiles):
    return np.max(profiles.sum(axis=-1), axis=-1) / np.sum(profiles, axis=(-1,-2))


def get_shannon_entropies(profiles):
    assert len(profiles.shape) >= 2, profiles.shape
    profiles_strand_sum = profiles.sum(axis=-2)
    
    profiles_norm = profiles_strand_sum / profiles_strand_sum.sum(axis=-1, keepdims=True)
    profiles_norm_log = np.log(profiles_norm + 1e-20)
    
    entropies = - (profiles_norm * profiles_norm_log).sum(axis=-1)
    return entropies


def get_norm_shannon_entropies(profiles, counts):
    shannon_entropies = get_shannon_entropies(profiles)
    
    if counts.shape[-1] == 2:  # if stranded counts
        counts = counts.sum(axis=-1)
    
    norm_shannon_entropies = shannon_entropies / np.log(np.log(counts))
    return norm_shannon_entropies


def load_coords(peak_bed, in_window):
    if peak_bed.endswith(".gz"):
        with gzip.open(peak_bed) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(peak_bed) as f:
            lines = [line.split() for line in f]

    coords = []
    for line in lines:
        chrom, peak_start, peak_end = line[0], int(line[1]), int(line[2])
        mid = (peak_start + peak_end) // 2
        window_start = mid - in_window // 2
        window_end = mid + in_window // 2
        coords.append((chrom, window_start, window_end))
    return coords
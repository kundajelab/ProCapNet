import os
import numpy as np
import pyBigWig
from collections import defaultdict
import sys

assert len(sys.argv) == 3, sys.argv  # expecting cell type, chromosome name
cell_type = sys.argv[1]
which_chromosome = sys.argv[2]

print("Cell type:", cell_type)
print("Chromsome:", which_chromosome)



# This is now redundant with v1_process_preds_whole_chromosome.py (run that instead);
# but I am keeping it just in case for historical record


# the input sequence length and output prediction window length
in_window = 2114
out_window = 1000

which_genome = "t2t"

chrom_sizes_filepath = which_genome + "/" + which_genome + ".chrom.sizes"


# loead chromosome info

def load_chrom_sizes(chrom_sizes_filepath):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
    return chrom_sizes


chrom_sizes = load_chrom_sizes(chrom_sizes_filepath)



def get_merged_preds_path(genome = which_genome,
                          chrom = which_chromosome,
                          cell_type = cell_type):
    
    # where we saved all of the predictions generated
                                      
    merged_preds_dir = "/".join(["raw_preds", genome, cell_type, chrom, "merged"])
    os.makedirs(merged_preds_dir, exist_ok=True)
    return merged_preds_dir + "/preds.npy"
        
        
def make_track_values_dict(values):
                                      
    # simplified for one-chrom, one-array case
    track_values = defaultdict(lambda : [])

    for position, value in enumerate(values):
        track_values[position] = track_values[position] + [value]
    
    # take the mean at each position, so that if there was ovelap, the average value is used
    track_values = { key : sum(vals) / len(vals) for key, vals in track_values.items() }
    return track_values
    
    
def get_bigwigs_save_dir(which_genome = which_genome,
                         which_chromosome = which_chromosome,
                         cell_type = cell_type):
    
    bw_dir = "/".join(["bigwigs", which_genome, cell_type, which_chromosome]) + "/"
    os.makedirs(bw_dir, exist_ok=True)
    return bw_dir


def write_preds_to_bigwigs(chrom_sizes = chrom_sizes,
                           which_chromosome = which_chromosome,
                           cell_type = cell_type):
    
    print("Writing predicted profiles to bigwigs.")
    
    bw_save_dir = get_bigwigs_save_dir()
    
    for strand_idx, strand in enumerate(["pos", "neg"]):
        # write separate bigwigs for svalues on the forward vs. reverse strands (in case of overlap)
        filename = ".".join([which_chromosome, cell_type, strand, "bigWig"])
        bw_path = bw_save_dir + filename
        print("Save path: " + bw_path)

            
        bw = pyBigWig.open(bw_path, 'w')
        # bigwigs need headers before they can be written to
        # the header is just the info you'd find in a chrom.sizes file
        bw.addHeader(chrom_sizes)
        
        merged_preds_path = get_merged_preds_path(genome = which_genome,
                                                  chrom = which_chromosome,
                                                  cell_type = cell_type)
        assert os.path.exists(merged_preds_path), merged_preds_path

        merged_preds = np.load(merged_preds_path).squeeze()

        strand_preds = merged_preds[0] if strand == "pos" else merged_preds[1]

        # convert arrays of scores for each peak into dict of base position : score
        # this will average together scores at the same position from different called peaks
        track_values_dict = make_track_values_dict(strand_preds)
        num_entries = len(track_values_dict)

        starts = sorted(list(track_values_dict.keys()))
        ends = [position + 1 for position in starts]
        values_to_write = [track_values_dict[key] for key in starts]

        assert len(values_to_write) == len(starts) and len(values_to_write) == len(ends)

        bw.addEntries([which_chromosome for _ in range(num_entries)], 
                       starts, ends = ends, values = values_to_write)

        bw.close()
        
    print("Done writing bigwigs.")

    
    
if __name__ == "__main__":
    write_preds_to_bigwigs()
    
    


import os
import numpy as np
import pyBigWig
from collections import defaultdict
import sys

assert len(sys.argv) == 2, sys.argv  # expecting cell type
cell_type = sys.argv[1]

print("Cell type:", cell_type)


# the input sequence length and output prediction window length
in_window = 2114
out_window = 1000

which_genome = "t2t"

chrom_sizes_filepath = which_genome + "/" + which_genome + ".chrom.sizes"


# loead chromosome info

def load_chrom_sizes(chrom_sizes_filepath, exclude = ["chrM"]):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines if line[0] not in exclude]
    return chrom_sizes


chrom_sizes = load_chrom_sizes(chrom_sizes_filepath)



def get_merged_preds_path(chrom, genome = which_genome,
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
                         cell_type = cell_type):
    
    bw_dir = "/".join(["bigwigs", which_genome, cell_type, "genomewide"]) + "/"
    os.makedirs(bw_dir, exist_ok=True)
    return bw_dir


def write_preds_to_bigwigs(chrom_sizes = chrom_sizes,
                           cell_type = cell_type):
    
    print("Writing predicted profiles to bigwigs.")
    
    bw_save_dir = get_bigwigs_save_dir()
    
    for strand_idx, strand in enumerate(["pos", "neg"]):
        # write separate bigwigs for svalues on the forward vs. reverse strands (in case of overlap)
        filename = ".".join([cell_type, strand, "bigWig"])
        bw_path = bw_save_dir + filename
        print("Save path: " + bw_path)

            
        bw = pyBigWig.open(bw_path, 'w')
        # bigwigs need headers before they can be written to
        # the header is just the info you'd find in a chrom.sizes file
        bw.addHeader(chrom_sizes)
        
        for chrom, chrom_size in chrom_sizes:
            print("Chromosome:", chrom)
            
            merged_preds_path = get_merged_preds_path(chrom = chrom,
                                                      genome = which_genome,
                                                      cell_type = cell_type)
            assert os.path.exists(merged_preds_path), merged_preds_path

            merged_preds = np.load(merged_preds_path).squeeze()
            assert merged_preds.shape[0] == 2, merged_preds.shape  # assume first axis is strand
            
            strand_preds = merged_preds[strand_idx]
            assert len(strand_preds) == chrom_size, (len(strand_preds), chrom_size)

            # updated to take advantage of pybigwig supporting numpy (new?)
            starts = np.arange(len(strand_preds))
            ends = starts + 1
            
            chrom_repeated = np.empty((len(strand_preds),), dtype="<U" + str(len(chrom)))
            chrom_repeated[:] = chrom

            bw.addEntries(chrom_repeated, starts, ends = ends, values = strand_preds)

        bw.close()
        
    print("Done writing bigwigs.")

    
    
if __name__ == "__main__":
    write_preds_to_bigwigs()
    
    


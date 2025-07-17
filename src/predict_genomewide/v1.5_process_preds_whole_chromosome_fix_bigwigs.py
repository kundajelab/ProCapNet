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


# the input sequence length and output prediction window length
in_window = 2114
out_window = 1000

# how many folds we trained the models across (so, how many models per cell type)
num_folds = 7

which_genome = "hg38"

chrom_sizes_filepath = which_genome + "/" + which_genome + ".chrom.sizes"


# loead chromosome info

def load_chrom_sizes(chrom_sizes_filepath):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
    return chrom_sizes


chrom_sizes = load_chrom_sizes(chrom_sizes_filepath)



### where we saved all of the predictions generated

def get_merged_preds_save_dir(cell_type = cell_type,
                           which_genome = which_genome, which_chromosome = which_chromosome):
    
    return "/".join(["raw_preds", which_genome, cell_type, which_chromosome, "merged"])


def get_merged_preds_path(first_chunk_start, last_chunk_start,
                          which_genome = which_genome, which_chromosome = which_chromosome,
                          cell_type = cell_type):
                                      
    merged_preds_dir = get_merged_preds_save_dir()
    filename = "chunk_" + str(first_chunk_start) + "_" + str(last_chunk_start) + "_preds.npy"
    return merged_preds_dir + "/" + filename


### Load predictions in


def infer_chunks_from_preds():
    # for an arbitrary fold, infer from filenames what the chunks were for this chrom
    merged_pred_filename = sorted(os.listdir(get_merged_preds_save_dir()))
    
    chunks = []
    for pred_filename in merged_pred_filename:
        chunk_start = int(pred_filename.split("_")[1])
        chunk_end = int(pred_filename.split("_")[2])
        chunks.append((chunk_start, chunk_end))
    
    chunks = sorted(chunks, key = lambda chunk : chunk[0])
    return chunks

        
        
def make_track_values_dict(values, start):
                                      
    # simplified for one-chrom, one-array case
    track_values = defaultdict(lambda : [])

    for position, value in enumerate(values):
        position_offset = position + start
        track_values[position_offset] = track_values[position_offset] + [value]
    
    # take the mean at each position, so that if there was ovelap, the average value is used
    track_values = { key : sum(vals) / len(vals) for key, vals in track_values.items() }
    return track_values


def chunk_start_coords_to_bw_offset(first_start, in_window = in_window, out_window = out_window):
    return first_start + (in_window - out_window) // 2
    
    
def get_bigwigs_save_dir(which_genome = which_genome,
                         which_chromosome = which_chromosome,
                         cell_type = cell_type):
    
    bw_dir = "/".join(["bigwigs", which_genome, cell_type, which_chromosome]) + "/"
    os.makedirs(bw_dir, exist_ok=True)
    return bw_dir


def write_preds_to_bigwigs(chunk_start_coords, chrom_sizes,
                           which_chromosome = which_chromosome, cell_type = cell_type):
    
    print("Writing predicted profiles to bigwigs.")
    
    bw_save_dir = get_bigwigs_save_dir()
    
    for strand_idx, strand in enumerate(["pos", "neg"]):
        # write separate bigwigs for svalues on the forward vs. reverse strands (in case of overlap)
        bw_filename = bw_save_dir + ".".join([which_chromosome, cell_type, strand, "bigWig"])
        
        print("Save path: " + bw_filename)

            
        bw = pyBigWig.open(bw_filename, 'w')
        # bigwigs need headers before they can be written to
        # the header is just the info you'd find in a chrom.sizes file
        bw.addHeader(chrom_sizes)
        
        for first_chunk_start, last_chunk_start in chunk_start_coords:
            merged_preds_path = get_merged_preds_path(first_chunk_start, last_chunk_start)
            if not os.path.exists(merged_preds_path):
                print("Can't find " + merged_preds_path)
                continue
            merged_preds = np.load(merged_preds_path).squeeze()

            strand_preds = merged_preds[0] if strand == "pos" else merged_preds[1]

            # convert arrays of scores for each peak into dict of base position : score
            # this will average together scores at the same position from different called peaks
            bw_offset = chunk_start_coords_to_bw_offset(first_chunk_start)
            track_values_dict = make_track_values_dict(strand_preds, bw_offset)
            num_entries = len(track_values_dict)
            
            starts = sorted(list(track_values_dict.keys()))
            ends = [position + 1 for position in starts]
            values_to_write = [track_values_dict[key] for key in starts]
            
            assert len(values_to_write) == len(starts) and len(values_to_write) == len(ends)
            
            bw.addEntries([which_chromosome for _ in range(num_entries)], 
                           starts, ends = ends, values = values_to_write)
    
        bw.close()
        
    print("Done writing bigwigs.")

    
    
def main():
    chunk_start_coords = infer_chunks_from_preds()
    print("Num chunks:", len(chunk_start_coords))

    write_preds_to_bigwigs(chunk_start_coords, chrom_sizes)
    
    
main()

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

def get_raw_preds_save_dir(model_fold, cell_type = cell_type,
                           which_genome = which_genome, which_chromosome = which_chromosome):
    
    dir_prefix = "/".join(["raw_preds", which_genome, cell_type, which_chromosome])
    save_dir = dir_prefix + "/fold_" + str(model_fold) + "/"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_preds_save_prefix(model_fold, first_chunk_start, last_chunk_start,
                          cell_type = cell_type, which_genome = which_genome,
                          which_chromosome = which_chromosome):
    
    preds_out_dir = get_raw_preds_save_dir(model_fold, cell_type = cell_type,
                           which_genome = which_genome, which_chromosome = which_chromosome)
    
    file_prefix = "chunk_" + str(first_chunk_start) + "_" + str(last_chunk_start) + "_pred_"
    return preds_out_dir + file_prefix





### Load predictions in


def infer_chunks_from_raw_preds():
    # for an arbitrary fold, infer from filenames what the chunks were for this chrom
    fold0_raw_pred_filenames = sorted(os.listdir(get_raw_preds_save_dir(0)))
    
    chunks = []
    for raw_pred_filename in fold0_raw_pred_filenames:
        # you'll do everything twice if you don't ignore the counts files
        # (there should always be pairs of files for each chunk: profiles and counts)
        if "profiles" in raw_pred_filename:
            chunk_start = int(raw_pred_filename.split("_")[1])
            chunk_end = int(raw_pred_filename.split("_")[2])
            chunks.append((chunk_start, chunk_end))
    
    chunks = sorted(chunks, key = lambda chunk : chunk[0])
    return chunks


def _load_raw_preds_from_chunk(model_fold, first_chunk_start, last_chunk_start):
    save_prefix = get_preds_save_prefix(model_fold, first_chunk_start, last_chunk_start)
    pred_profiles = np.load(save_prefix + "profiles.npy")
    pred_logcounts = np.load(save_prefix + "logcounts.npy")
    return pred_profiles, pred_logcounts


def merge_preds_across_folds(first_chunk_start, last_chunk_start):
    # load model predictions across all folds, then average across the folds
    
    pred_profs_across_folds = []
    pred_logcounts_across_folds = []
    for model_fold in range(num_folds):
        _pred_profiles, _pred_logcounts = _load_raw_preds_from_chunk(model_fold,
                                                                     first_chunk_start,
                                                                     last_chunk_start)
        pred_profs_across_folds.append(_pred_profiles)
        pred_logcounts_across_folds.append(_pred_logcounts)

    # exponentiating profiles here
    pred_profs_across_folds = np.exp(np.array(pred_profs_across_folds))
    pred_logcounts_across_folds = np.array(pred_logcounts_across_folds)

    # average across folds
    pred_profs = pred_profs_across_folds.mean(axis=0)
    pred_logcounts = pred_logcounts_across_folds.mean(axis=0)
    return pred_profs, pred_logcounts


def merge_preds_across_chunk(first_chunk_start, last_chunk_start,
                             stride = 250, chunk_size = in_window, out_window = out_window):
    
    # load and merge preds across model folds for this chunk
    pred_profs, pred_logcounts = merge_preds_across_folds(first_chunk_start,
                                                          last_chunk_start)

    # combine counts and profile predictions
    pred_profs_scaled = pred_profs * np.exp(pred_logcounts)[..., None]
    

    # then, get average of scaled profiles tiled across the whole chunk 
    
    # make list of inner chunk starts, so you can tell when a base was part of an inner chunk
    
    num_tile_bounds = int(np.ceil((last_chunk_start - first_chunk_start) / stride + 1))
    tile_bounds = list(stride * np.arange(0, num_tile_bounds - 1))
    if tile_bounds[-1] != last_chunk_start - first_chunk_start:
        tile_bounds.append(last_chunk_start - first_chunk_start)
    
    
    # then, for each base, get avg pred across all tiles that overlapped it
    
    # we will take average by taking sum, then dividing by the # of sums
    # (most bases will have the same # of sums, but the edge cases won't)
    
    
    # first: just sum all the tiled predictions into one long vector,
    # keeping track of relative positioning of tiles
    
    chunk_len = last_chunk_start - first_chunk_start + out_window
    sum_preds = np.zeros((2, chunk_len))
    
    for tile_i, tile_bound in enumerate(tile_bounds):
        sum_preds[:, tile_bound : tile_bound + out_window] += pred_profs_scaled[tile_i]
    
    
    # second: count the number of sums that will happen at each base
    
    # HEY!! THIS ONLY WORKS IF [out_window] IS A MULTIPLE OF [stride]!!
    num_overlaps_default = out_window // stride
    assert out_window / stride == float(num_overlaps_default), (out_window / stride, num_overlaps_default)

    # set default number of tiles overlapping a base to be [out_window // stride]
    num_sums = np.ones((1, chunk_len,)) * num_overlaps_default
    
    # then just check the edge cases, where the default won't be true, and adjust them

    for base in range(chunk_len):
        # if not on the far left edge or far right edge of this chunk
        # (where there are fewer tiles covering the bases than usual)
        if not (base <= out_window or base >= chunk_len - out_window - 2):
            continue

        num_sums_this_base = 0
        for tile_i, tile_bound in enumerate(tile_bounds):
            # if not on the far left or far right edge tiles
            if not (tile_i <= num_overlaps_default or tile_i >= - num_overlaps_default - 1):
                continue 
                
            # check if this tile actually overlaps this base
            relative_position_of_base = base - tile_bound
            if relative_position_of_base >= 0 and relative_position_of_base < out_window:
                num_sums_this_base += 1
                
        num_sums[:, base] = num_sums_this_base
    
    # finally: turn sum into mean by dividing
    avg_preds = sum_preds / num_sums
    return avg_preds




def get_merged_preds_path(first_chunk_start, last_chunk_start,
                          which_genome = which_genome, which_chromosome = which_chromosome,
                          cell_type = cell_type):
                                      
    merged_preds_dir = "/".join(["raw_preds", which_genome, cell_type, which_chromosome, "merged"]) + "/"
    os.makedirs(merged_preds_dir, exist_ok=True)
    filename = "chunk_" + str(first_chunk_start) + "_" + str(last_chunk_start) + "_preds.npy"
    return merged_preds_dir + filename
        
        
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
    chunk_start_coords = infer_chunks_from_raw_preds()
    print("Num chunks:", len(chunk_start_coords))
    
    for first_chunk_start, last_chunk_start in chunk_start_coords:
        print("Chunk:", first_chunk_start, last_chunk_start)
        save_path = get_merged_preds_path(first_chunk_start, last_chunk_start)

        if not os.path.exists(save_path):
            print("Saving to ", save_path)
            preds_merged = merge_preds_across_chunk(first_chunk_start, last_chunk_start)
            np.save(save_path, preds_merged)

    write_preds_to_bigwigs(chunk_start_coords, chrom_sizes)
    
    
main()

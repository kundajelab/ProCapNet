'''
Code here is based very loosely off of Surag's script:
https://github.com/kundajelab/surag-scripts/blob/master/bpnet-pipeline/importance/importance_hdf5_to_bigwig.py

'''

import numpy as np
import pyBigWig
from collections import defaultdict
import gzip
from utils import load_chrom_names


def make_track_values_dict(all_values, coords, chrom):
    track_values = defaultdict(lambda : [])

    for values, (coord_chrom, start, end) in zip(all_values, coords):
        # subset to only peaks/values for the chromosome we're looking at now
        if coord_chrom == chrom:
            assert values.shape[0] == end - start, (values.shape, start, end, end - start)
            
            for position, value in enumerate(values):
                position_offset = position + start
                track_values[position_offset] = track_values[position_offset] + [value]
    
    # take the mean at each position, so that if there was ovelap, the average value is used
    track_values = { key : sum(vals) / len(vals) for key, vals in track_values.items() }
    return track_values


def get_header_for_bigwig(chrom_sizes_filepath):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
    return chrom_sizes


def load_coords(peaks_file, window_len):
    lines = []
    if peaks_file.endswith(".gz"):
        with gzip.open(peaks_file) as f:
            lines = [line.decode().split()[:3] for line in f]
    else:
        with open(peaks_file) as f:
            lines = [line.split()[:3] for line in f]
            
    coords = []
    for line in lines:
        chrom, start, end = line[0], int(line[1]), int(line[2])
        mid = start + (end - start) // 2
        coord = (chrom, mid - window_len // 2, mid + window_len // 2)
        coords.append(coord)
    return coords


def write_tracks_to_bigwigs(values,
                            peaks_file,
                            save_filepath,
                            chrom_sizes_filepath):
    
    values = values.astype("float64")  # pybigwig will throw error if you don't do this
    assert len(values.shape) == 3, values.shape  # going with 2-stranded data here
    window_len = values.shape[-1]  # assuming last axis is profile len axis
    
    print("Writing predicted profiles to bigwigs.")
    print("Peaks: " + peaks_file)
    print("Save path: " + save_filepath.replace(".npy", ""))
    print("Chrom sizes file: " + chrom_sizes_filepath)
    print("Window length: " + str(window_len))
    
    coords = load_coords(peaks_file, window_len)
    assert len(coords) == values.shape[0]
    
    for strand_idx, strand in enumerate(["+", "-"]):
        # write separate bigwigs for svalues on the forward vs. reverse strands (in case of overlap)
        if strand == "+":
            bw_filename = save_filepath.replace(".npy", "") + ".pos.bigWig"
        else:
            bw_filename = save_filepath.replace(".npy", "") + ".neg.bigWig"
        
        values_for_strand = values[:, strand_idx, :]
            
        # bigwigs need to be written in order -- so we have to go chromosome by chromosome,
        # and the chromosomes need to be numerically sorted (i.e. chr9 then chr10)
        chrom_names = load_chrom_names(chrom_sizes_filepath)
        chrom_order = {chrom : i for i, chrom in enumerate(chrom_names)}

        chromosomes = sorted(list({coord[0] for coord in coords}),
                             key = lambda chrom_str : chrom_order[chrom_str])
            
        bw = pyBigWig.open(bw_filename, 'w')
        # bigwigs need headers before they can be written to
        # the header is just the info you'd find in a chrom.sizes file
        bw.addHeader(get_header_for_bigwig(chrom_sizes_filepath))
        
        for chrom in chromosomes:
            # convert arrays of scores for each peak into dict of base position : score
            # this function will average together scores at the same position from different called peaks
            track_values_dict = make_track_values_dict(values_for_strand, coords, chrom)
            num_entries = len(track_values_dict)
            
            starts = sorted(list(track_values_dict.keys()))
            ends = [position + 1 for position in starts]
            values_to_write = [track_values_dict[key] for key in starts]
            
            assert len(values_to_write) == len(starts) and len(values_to_write) == len(ends)
            
            bw.addEntries([chrom for _ in range(num_entries)], 
                           starts, ends = ends, values = values_to_write)
    
        bw.close()
        
        
def write_scores_to_bigwigs(scores,
                            peaks_file,
                            save_filepath,
                            chrom_sizes_filepath):
    
    scores = scores.astype("float64")  # pybigwig will throw error if you don't do this
    assert len(scores.shape) == 2, scores.shape  # should be one-hot and flattened
    window_len = scores.shape[-1]  # assuming last axis is profile len axis
    
    coords = load_coords(peaks_file, window_len)
    assert len(coords) == scores.shape[0]
    
    save_filepath = save_filepath.replace(".npy", "")
    if not (save_filepath.endswith(".bigWig") or save_filepath.endswith(".bw")):
        bw_filename = save_filepath + ".bigWig"
    else:
        bw_filename = save_filepath
          
    print("Writing attributions to bigwig.")
    print("Peaks: " + peaks_file)
    print("Save path: " + bw_filename)
    print("Chrom sizes file: " + chrom_sizes_filepath)
    print("Window length: " + str(window_len))

    # bigwigs need to be written in order -- so we have to go chromosome by chromosome,
    # and the chromosomes need to be numerically sorted (i.e. chr9 then chr10)
    chrom_names = load_chrom_names(chrom_sizes_filepath)
    chrom_order = {chrom : i for i, chrom in enumerate(chrom_names)}
    chromosomes = sorted(list({coord[0] for coord in coords}),
                         key = lambda chrom_str : chrom_order[chrom_str])

    bw = pyBigWig.open(bw_filename, 'w')
    # bigwigs need headers before they can be written to
    # the header is just the info you'd find in a chrom.sizes file
    bw.addHeader(get_header_for_bigwig(chrom_sizes_filepath))

    for chrom in chromosomes:
        # convert arrays of scores for each peak into dict of base position : score
        # this function will average together scores at the same position from different called peaks
        track_values_dict = make_track_values_dict(scores, coords, chrom)
        num_entries = len(track_values_dict)

        starts = sorted(list(track_values_dict.keys()))
        ends = [position + 1 for position in starts]
        scores_to_write = [track_values_dict[key] for key in starts]

        assert len(scores_to_write) == len(starts) and len(scores_to_write) == len(ends)

        bw.addEntries([chrom for _ in range(num_entries)], 
                       starts, ends = ends, values = scores_to_write)

    bw.close()

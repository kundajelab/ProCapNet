# Code modified from an earlier version of Anusri's ChromBPNet's GC-matching:
# https://github.com/kundajelab/chrombpnet/tree/be0fe06cf0f2dec3147c87cf08e81ec831881fec/chrombpnet/helpers/make_gc_matched_negatives



import argparse
import pandas as pd
from tqdm import tqdm
import random
import csv 
import gzip
import pyBigWig
import numpy
from collections import defaultdict
import sys

def parse_args():
    parser=argparse.ArgumentParser(description="generate a bed file of non-peak regions that are CpG-matched with foreground")
    parser.add_argument("-c","--candidate_negatives",help="candidate negatives bed file with CpG content in 4th column rounded to 2 decimals")
    parser.add_argument("-f","--peaks_cpg_bed", help="regions with their corresponding CpG fractions for matching, 4th column has CpG content value rounded to 2 decimals")
    parser.add_argument("-o","--out", help="CpG-matched non-peaks output file name")
    parser.add_argument("-b", "--bw", help="filepath prefix for bigwig files (used for calculating counts in windows)")
    parser.add_argument("-s", "--chrom_sizes", help="filepath for chromosome sizes file")
    return parser.parse_args()


def load_coords(coords_file, load_cpg = True):
    df = pd.read_csv(coords_file, header=None, sep="\t", compression="gzip")
    chroms, starts, ends, cpgs = df[0], df[1], df[2], df[len(df.columns) - 1]

    coords = list(zip(chroms, starts, ends))
    if load_cpg:
        return { coord : cpg for coord, cpg in zip(coords, cpgs) }
    else:
        return coords


def load_chrom_sizes(chrom_sizes_file, filter_out = ["_", "M", "Un", "EBV"], filter_in = ["chr"]):
    with open(chrom_sizes_file) as f:
        lines = [(line.split()[0], int(line.split()[1])) for line in f]
        
    # filtering out the scaffolds and weird stuff
    chroms = { chrom : length for chrom, length in lines if all([filt not in chrom for filt in filter_out]) }
    chroms_filt = { chrom : length for chrom, length in chroms.items() if all([filt in chrom for filt in filter_in]) }
    return chroms_filt


def get_counts_in_peaks(coords, bw_prefix, window_len = 1057):
    plus_bw = pyBigWig.open(bw_prefix + ".pos.bigWig", "r")
    minus_bw = pyBigWig.open(bw_prefix + ".neg.bigWig", "r")

    counts = []
    for chrom, peak_start, peak_end in tqdm(coords):
        mid = peak_start + (peak_end - peak_start) // 2
        start = mid - window_len
        end = mid + window_len

        signal_pos = plus_bw.values(chrom, start, end, numpy=True)
        signal_pos = numpy.nan_to_num(signal_pos)

        signal_neg = minus_bw.values(chrom, start, end, numpy=True)
        signal_neg = numpy.nan_to_num(signal_neg)

        counts.append(numpy.sum(signal_pos) + numpy.sum(signal_neg))

    return counts


def get_counts_in_windows(coords, bw_prefix, chrom_sizes, upper_thresh = None, lower_thresh=1, window_len = 500):
    # window_len here should be the length of the profile, not the sequence
    
    plus_bw = pyBigWig.open(bw_prefix + ".pos.bigWig", "r")
    minus_bw = pyBigWig.open(bw_prefix + ".neg.bigWig", "r")
    
    # read in the per-base counts values for all the chromosomes
    print("Loading chromosomes...")
    chrom_sizes_dict = load_chrom_sizes(chrom_sizes)
    chrom_counts_dict = {}
    for chrom, chrom_len in tqdm(chrom_sizes_dict.items()):
        counts_along_chrom_pos = plus_bw.values(chrom, 0, chrom_sizes_dict[chrom], numpy=True)
        counts_along_chrom_pos = numpy.nan_to_num(counts_along_chrom_pos)

        counts_along_chrom_neg = minus_bw.values(chrom, 0, chrom_sizes_dict[chrom], numpy=True)
        counts_along_chrom_neg = numpy.nan_to_num(counts_along_chrom_neg)
        chrom_counts_dict[chrom] = counts_along_chrom_pos + counts_along_chrom_neg

    counts = []
    coords_to_keep = []
    for chrom, peak_start, peak_end in tqdm(coords):
        if chrom not in chrom_sizes_dict:
            continue
            
        mid = peak_start + (peak_end - peak_start) // 2
        prof_start = mid - window_len
        prof_end = mid + window_len

        if prof_end > chrom_sizes_dict[chrom]:
            continue
            
        count = numpy.sum(chrom_counts_dict[chrom][prof_start:prof_end])
        if count >= lower_thresh:
            if upper_thresh is None or count <= upper_thresh:
                counts.append(count)
                coords_to_keep.append((chrom, peak_start, peak_end))
    
    return dict(zip(coords_to_keep, counts)) 


def get_count_upper_threshold(peak_coords, bw_prefix, count_thresh_frac):
    peak_counts = get_counts_in_peaks(peak_coords, bw_prefix)
    min_peak_count = min(peak_counts)
    upper_thresh = int(min_peak_count * count_thresh_frac)
    print("Using " + str(upper_thresh) + " as the upper threshold for counts.")
    return upper_thresh


def format_cpg_dict(candidate_negative_coords_with_cpg, candidate_negative_coords_with_counts):
    """
    Imports the candidate negatives into a dictionary structure.
    The `key` is the cpg content fraction, and the `values` are a list 
    containing the (chrom,start,end) of a region with the corresponding 
    cpg content fraction.
    """
    cpg_dict=defaultdict(lambda : defaultdict( lambda : []))
    counts_dict=defaultdict(lambda : defaultdict( lambda : []))
    for coord, cpg in tqdm(candidate_negative_coords_with_cpg.items()):
        chrom = coord[0]
        cpg_dict[chrom][cpg].append(coord)
        counts_dict[chrom][candidate_negative_coords_with_counts[coord]].append(coord)
    return cpg_dict, counts_dict

def scale_cpg(cur_cpg):
    """
    Randomly increase/decrease the cpg-fraction value by 0.01
    """
    if random.random()>0.5:
        cur_cpg+=0.01
    else:
        cur_cpg-=0.01
    cur_cpg=round(cur_cpg,2)
    if cur_cpg<=0:
        cur_cpg+=0.01
    if cur_cpg>=1:
        cur_cpg-=0.01
    assert cur_cpg >=0
    assert cur_cpg <=1
    return cur_cpg 

def adjust_cpg(chrom, cur_cpg, negatives, used_negatives):
    """
    Function that checks if (1) the given cpg fraction value is available
    in the negative candidates or (2) if the given cpg fraction value has 
    candidates not already sampled. If eitheir of the condition fails we  
    sample the neighbouring cpg_fraction value by randomly scaling with 0.01.
    """
    if chrom  not in used_negatives:
        used_negatives[chrom]={}

    if cur_cpg not in used_negatives[chrom]:
        used_negatives[chrom][cur_cpg]=[]

    while (cur_cpg not in negatives[chrom]) or (len(used_negatives[chrom][cur_cpg])>=len(negatives[chrom][cur_cpg])):
        cur_cpg=scale_cpg(cur_cpg)
        if cur_cpg not in used_negatives[chrom]:
            used_negatives[chrom][cur_cpg]=[]
    return cur_cpg,used_negatives 

        
    
if __name__=="__main__":

    args=parse_args()

    print("Loading current peak set...")
    cur_peaks_with_cpg = load_coords(args.peaks_cpg_bed)

    #print("Calculating upper threshold for counts...")
    #count_upper_threshold = get_count_upper_threshold(cur_peaks_with_cpg, args.bw, count_thresh_frac = 1) ###### was 0.85
    print("Not applying an upper threshold for counts!!!        #############")
    count_upper_threshold = None

    print("Loading candidate negative windows...")
    candidate_negatives_with_cpg = load_coords(args.candidate_negatives)

    # load count values for candidate negative windows, filter by > 0 and < count_upper_threshold
    print("Loading count values for negative windows...")
    candidate_negatives_with_counts = get_counts_in_windows(candidate_negatives_with_cpg, args.bw, args.chrom_sizes, count_upper_threshold)

    print("Filtering candidate negatives...")
    # apply filter to this dict also
    tmp = {key:candidate_negatives_with_cpg[key] for key in candidate_negatives_with_counts}
    candidate_negatives_with_cpg = tmp

    print("Re-formatting candidate negatives by chromosome, cpg bin...")
    negatives_with_cpg, negatives_with_counts = format_cpg_dict(candidate_negatives_with_cpg, candidate_negatives_with_counts)

    print("Matching peaks to candidate negatives by CpG content...")
    used_negatives = dict()
    negatives_bed = []
    for coord, cpg_value in tqdm(cur_peaks_with_cpg.items()):
        chrom = coord[0]
        cur_cpg, used_negatives = adjust_cpg(chrom, cpg_value, negatives_with_cpg, used_negatives)
        num_candidates = len(negatives_with_cpg[chrom][cur_cpg])
        
        rand_neg_index = random.randint(0, num_candidates - 1)
        while rand_neg_index in used_negatives[chrom][cur_cpg]:
            cur_cpg,used_negatives = adjust_cpg(chrom, cur_cpg, negatives_with_cpg, used_negatives)
            num_candidates = len(negatives_with_cpg[chrom][cur_cpg])
            rand_neg_index = random.randint(0, num_candidates - 1)

        used_negatives[chrom][cur_cpg].append(rand_neg_index)
        neg_tuple = negatives_with_cpg[chrom][cur_cpg][rand_neg_index]
        neg_chrom, neg_start, neg_end = neg_tuple[:3]
        negatives_bed.append([neg_chrom, int(neg_start), int(neg_end), cur_cpg])        
       
    print("Saving results...")
    negatives_bed = pd.DataFrame(negatives_bed)
    negatives_bed.to_csv(args.out, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE, compression="gzip")

    print("Done!")


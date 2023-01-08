from collections import defaultdict

import sys
sys.path.append("../2_train_models")
from utils import get_proj_dir

import gzip
import numpy as np


def get_ccre_bed(cell_type):
    proj_dir = get_proj_dir()
    return proj_dir + "/annotations/" + cell_type + "/cCREs.bed.gz"


def load_coords_with_summits(peak_bed, in_window):
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
        
        if line[-2] == ".":
            summit_pos = None
        else:
            summit_pos = int(line[-2])

        if line[-1] == ".":
            summit_neg = None
        else:
            summit_neg = int(line[-1])
        
        coords.append((chrom, window_start, window_end, summit_pos, summit_neg))
    return coords


def load_annotations(annot_bed):
    if annot_bed.endswith(".gz"):
        with gzip.open(annot_bed) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(annot_bed) as f:
            lines = [line.split() for line in f]

    coords = []
    for line in lines:
        chrom, start, end, annot = line[0], int(line[1]), int(line[2]), line[9]
        coords.append((chrom, start, end, annot))
    return coords


def get_overlap_annots(list_a, list_b):
    # This function is similar to bedtools intersect,
    # but returns the bed file's labels, if windows overlap, for each
    # window in list_a.
    
    # Assumes everything's on the same chromosome
    # Assumes list_a is list of (start, stop), list_b is (start, stop, label)
    
    # output is list with len == len(list_a)
    matches = []
    
    for a_item in list_a:
        a_labels = []
        a_start, a_end = a_item
        b_index = 0
        while b_index < len(list_b):
            b_start, b_end, label = list_b[b_index]
            
            # b is after a
            if b_start >= a_end:
                break
            # b is before a
            if b_end <= a_start:
                b_index += 1
                continue
            # only other case left: b overlaps a
            a_labels.append(label)
            b_index += 1

        matches.append(a_labels)
    assert len(matches) == len(list_a), matches
    return matches


def format_annot_list(annot_list):
    # if input is ["A", "B,C", "D"], returns ["A", "B", "C", "D"]
    fixed_list = []
    for item in annot_list:
        fixed_list.extend(item.split(","))
    return fixed_list


def get_annotations_for_peaks(coords, annotation_file, in_window, out_window):
    annots = load_annotations(annotation_file)

    # get set of chromosomes included in peak set
    chroms = sorted(list(set(coord[0] for coord in coords)))

    # make dict of chromosome --> sorted list of annotated regions + labels
    annots_by_chrom = {chrom : sorted([a[1:] for a in annots if a[0] == chrom]) for chrom in chroms}
    
    # adjust the starts and ends of peak coordinates so they only cover +/- 500 bp
    # (otherwise we'd probably get a lot of FP annotation overlaps)
    adjust_by = (in_window - out_window) // 2
    coords_adjust = [(c[0], c[1] + adjust_by, c[2] - adjust_by) for c in coords]
    
    # get list of annotations overlapping peak, for each peak (takes a few min)
    overlap_annots_raw = [get_overlap_annots((coord[1:],), annots_by_chrom[coord[0]])[0] for coord in coords_adjust]
    
    # process raw string annotations into list of unique hits
    overlap_annots = [sorted(list(set(format_annot_list(annot_list)))) for annot_list in overlap_annots_raw]
    
    # get set of unique annotation labels
    all_annot_labels = set([annot for annot_list in overlap_annots for annot in annot_list])
    
    # make dict of annotation label --> list of len(num_peaks), where each element is True if overlap
    overlap_annots_bools = {annot_label : np.array([annot_label in annot_list for annot_list in overlap_annots]) for annot_label in all_annot_labels}
    return overlap_annots_bools


def load_annotations_no_label(bed_filepath):
    if bed_filepath.endswith(".gz"):
        with gzip.open(bed_filepath) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(hk_bed) as f:
            lines = [line.split() for line in f]

    coords = []
    for line in lines:
        chrom, start, end = line[0], int(line[1]), int(line[2])
        coords.append((chrom, start, end))
    return coords


def get_overlap_hks(coord_a, list_b):
    # This function is similar to bedtools intersect,
    # but returns True/False based on if any windows overlap
    # with coord_a.
    
    # Assumes everything's on the same chromosome
    # Assumes list_b is list of format (start, stop)
    
    a_start, a_end = coord_a[:2]
    b_index = 0
    a_found_in_b = False
    while b_index < len(list_b):
        b_start, b_end = list_b[b_index][:2]

        # b is after a (since b is sorted, there will be no overlap)
        if b_start >= a_end:
            return False
        # b is before a (since list_b is sorted, look at next element)
        elif b_end <= a_start:
            b_index += 1
        else:
            # only other case left: b overlaps a
            return True

    # never found overlap
    return False


def get_annotations_for_hk_genes(coords, hk_bed, in_window=2114, out_window=1000):
    annots = load_annotations_no_label(hk_bed)

    # get set of chromosomes included in peak set
    chroms = sorted(list(set(coord[0] for coord in coords)))

    # make dict of chromosome --> sorted list of annotation regions
    annots_by_chrom = {chrom : sorted([a[1:] for a in annots if a[0] == chrom]) for chrom in chroms}
    
    # adjust the starts and ends of peak coordinates so they only cover +/- 500 bp
    # (otherwise we'd probably get a lot of FP annotation overlaps)
    adjust_by = (in_window - out_window) // 2
    coords_adjust = [(c[0], c[1] + adjust_by, c[2] - adjust_by) for c in coords]
    
    # get bool for peak overlap, for each peak
    overlap_annots = np.array([get_overlap_hks(coord[1:], annots_by_chrom[coord[0]]) for coord in coords_adjust])
    return overlap_annots
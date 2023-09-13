from collections import defaultdict
import gzip
import numpy as np


def get_ccre_bed(cell_type, proj_dir):
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


def load_annotations(bed_filepath, label=True, label_col=9):
    if bed_filepath.endswith(".gz"):
        with gzip.open(bed_filepath) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(bed_filepath) as f:
            lines = [line.split() for line in f]

    coords = []
    for line in lines:
        if label:
            chrom, start, end, label = line[0], int(line[1]), int(line[2]), line[label_col]
            coords.append((chrom, start, end, label))
        else:
            chrom, start, end = line[0], int(line[1]), int(line[2])
            coords.append((chrom, start, end))
    return coords


def get_labels_of_what_a_overlaps_in_b(coord_a, list_b):
    # This function is similar to bedtools intersect,
    # but returns the bed file's labels, for any windows in b
    # that overlap with region a.
    
    # Assumes coord_a is (start, stop)
    # Assumes list_b is list of format (chrom, start, stop, label)
    assert check_list_is_sorted(list_b), list_b
    
    # check that list_b is regions with all the same chromosome
    assert len(set([coord[0] for coord in list_b])) == 1, set([coord[0] for coord in list_b])
    
    # if the chromosomes don't match, definitely nothing overlaps
    if set([coord[0] for coord in list_b]) != set([coord_a[0]]):
        return []
    
    overlap_labels = []
    a_start, a_end = coord_a[1:3]
    assert isinstance(a_start, int) and isinstance(a_end, int)
    b_index = 0
    while b_index < len(list_b):
        b_start, b_end, label = list_b[b_index][1:4]
        assert isinstance(b_start, int) and isinstance(b_end, int)
        
        # b is after a (since b is sorted, there will be no overlap)
        if b_start >= a_end:
            break
        # b is before a (since list_b is sorted, continue to next element)
        if b_end <= a_start:
            b_index += 1
            continue
        # only other case left: b overlaps a
        overlap_labels.append(label)
        b_index += 1

    return overlap_labels


def format_label_list(label_list):
    # if input is ["B,C", "A", "D", "A,B"], returns ["A", "B", "C", "D"]
    fixed_list = []
    for item in label_list:
        fixed_list.extend(item.split(","))
    return sorted(list(set(fixed_list)))


def find_peak_overlap_labels(coords, bed_file_with_labels, in_window=2114, out_window=1000):
    annots = load_annotations(bed_file_with_labels, label=True)

    # get set of chromosomes included in peak set
    chroms = sorted(list(set(coord[0] for coord in coords)))

    # make dict of chromosome --> sorted list of regions + labels
    # (later code will expect sorted input)
    annots_by_chrom = {chrom : sorted([a for a in annots if a[0] == chrom]) for chrom in chroms}
    
    # adjust the starts and ends of peak coordinates so they only cover +/- 500 bp
    adjust_by = (in_window - out_window) // 2
    
    overlaps = []
    for coord in coords:
        chrom, start, end = coord[:3]
        coord_adjust = (chrom, start + adjust_by, end - adjust_by)
    
        # get list of labels for regions overlapping peak
        overlap_annots_raw = get_labels_of_what_a_overlaps_in_b(coord_adjust, annots_by_chrom[chrom])
    
        # process raw strings into list of unique hits
        overlap_annots = format_label_list(overlap_annots_raw)
        overlaps.append(overlap_annots)
    
    # get set of all unique annotation labels that ever overlapped
    all_labels = set([label for label_list in overlaps for label in label_list])
    
    # make dict of label --> list of len(num_peaks),
    # where element i is True if peak i overlapped with that label
    overlaps_dict = dict()
    for label in all_labels:
        overlap_bools = [label in overlap_label_list for overlap_label_list in overlaps]
        overlaps_dict[label] = np.array(overlap_bools)

    return overlaps_dict


def check_list_is_sorted(a_list):
    list_len = len(a_list)
    for i in range(list_len - 1):
        if not a_list[i] <= a_list[i+1]:
            return False
    return True

def does_a_overlap_anything_in_b(coord_a, list_b):
    # This function is similar to bedtools intersect,
    # but returns True/False based on if any windows overlap
    # with coord_a.
    
    # Assumes list_b is list of format (chrom, start, stop)
    assert check_list_is_sorted(list_b), list_b
    
    # check that list_b is regions with all the same chromosome
    assert len(set([coord[0] for coord in list_b])) <= 1, set([coord[0] for coord in list_b])
    
    # if the chromosomes don't match, definitely no overlap
    if set([coord[0] for coord in list_b]) != set([coord_a[0]]):
        return False
    
    a_start, a_end = coord_a[1:3]
    assert isinstance(a_start, int) and isinstance(a_end, int)
    b_index = 0
    while b_index < len(list_b):
        b_start, b_end = list_b[b_index][1:3]
        assert isinstance(b_start, int) and isinstance(b_end, int)

        # b is after a (since b is sorted, there will be no overlap)
        if b_start >= a_end:
            return False
        # b is before a (since list_b is sorted, continue to next element)
        elif b_end <= a_start:
            b_index += 1
        else:
            # only other case left: b overlaps a
            return True

    # never found overlap
    return False


def find_peak_overlap(coords, bed_filepath, in_window=2114, out_window=1000):
    annots = load_annotations(bed_filepath, label=False)

    # get set of chromosomes included in peak set
    chroms = sorted(list(set(coord[0] for coord in coords)))

    # make dict of chromosome --> sorted list of annotation regions
    annots_by_chrom = {chrom : sorted([a for a in annots if a[0] == chrom]) for chrom in chroms}

    # adjust the starts and ends of peak coordinates so they only cover +/- 500 bp
    # (otherwise we'd probably get a lot of FP annotation overlaps)
    adjust_by = (in_window - out_window) // 2
    
    # for each peak, get boolean for whether or not it overlaps with anything
    overlaps = []
    for coord in coords:
        chrom, start, end = coord[:3]
        coord_adjust = (chrom, start + adjust_by, end - adjust_by)
        overlap = does_a_overlap_anything_in_b(coord_adjust, annots_by_chrom[chrom])
        overlaps.append(overlap)
    return np.array(overlaps)


def clean_coord_summits(coord):
    assert len(coord) >= 5, coord
    chrom, start, end, summit_pos, summit_neg = coord[:5]
    if summit_pos is None:
        summit_pos = summit_neg
    if summit_neg is None:
        summit_neg = summit_pos
    return (chrom, start, end, summit_pos, summit_neg)


def get_gene_region_overlap(coords, gene_regions_files, in_window=2114, out_window=1000):
    overlaps = dict()
    
    for region_name, region_filepath in gene_regions_files.items():
        regions = load_annotations(region_filepath, label=False)

        # get set of chromosomes included in peak set
        chroms = sorted(list(set(coord[0] for coord in coords)))

        # make dict of chromosome --> sorted list of regions
        regions_by_chrom = {chrom : sorted([c for c in regions if c[0] == chrom]) for chrom in chroms}
        
        overlap_bools = []
        for coord in coords:
            chrom, start, end, summit_pos, summit_neg = clean_coord_summits(coord)
            # adjust the starts and ends of peak coordinates so they only cover summits +1 bp
            coord_adjust = (chrom, min(summit_pos, summit_neg) - 1, max(summit_pos, summit_neg))

            overlap_bool = does_a_overlap_anything_in_b(coord_adjust, regions_by_chrom[chrom])
            overlap_bools.append(overlap_bool)
        
        overlaps[region_name] = np.array(overlap_bools)
        
    return overlaps


def get_dist_to_TSS(coords, TSSs_bed, in_window=2114, out_window=1000):
    TSSs = load_annotations(TSSs_bed, label=False)

    # get set of chromosomes included in peak set
    chroms = sorted(list(set(coord[0] for coord in coords)))

    # make dict of chromosome --> sorted list of annotation regions
    TSSs_by_chrom = {chrom : np.array(sorted([t[1] for t in TSSs if t[0] == chrom])) for chrom in chroms}

    TSS_dists = []
    for coord in coords:
        chrom, _, _, summit_pos, summit_neg = clean_coord_summits(coord)
        summit_midpoint = (summit_pos + summit_neg) // 2

        dist_to_nearest_TSS = np.min(np.abs(TSSs_by_chrom[chrom] - summit_midpoint))
        TSS_dists.append(dist_to_nearest_TSS)

    return np.array(TSS_dists)
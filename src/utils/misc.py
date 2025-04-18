import os
import json


def load_chrom_names(chrom_sizes, filter_out = ["_", "M", "Un", "EBV"], filter_in = ["chr"]):
    with open(chrom_sizes) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    chroms = [line[0] for line in lines]
    
    if filter_out and len(filter_out) > 0:
        chroms = [c for c in chroms if all([filt not in c for filt in filter_out])]
    if filter_in and len(filter_in) > 0:
        chroms = [c for c in chroms if all([filt in c for filt in filter_in])]
    return chroms

def load_chrom_sizes(chrom_sizes_filepath):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
        chrom_sizes = [(line[0], int(line[1])) for line in chrom_sizes_lines]
    return chrom_sizes


def ensure_parent_dir_exists(filepath):
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)
    
    
def load_json(json_path):
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
    return json_dict
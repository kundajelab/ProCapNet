import gzip
import os
import subprocess
from collections import defaultdict

import sys
sys.path.append("../2_train_models")
from write_bigwigs import get_header_for_bigwig
from utils import get_proj_dir


def load_gtf(gtf_filepath):
    regions = defaultdict(lambda : [])
    
    if gtf_filepath.endswith(".gz"):
        with gzip.open(gtf_filepath) as f:
            for line in f:
                line = line.decode()
                if line.startswith("#"):
                    continue  # skip header
                chrom, _, label, start, end = line.split()[:5]
                strand = line.split()[6]
                if label != "transcript":
                    regions[label].append((chrom, start, end, strand))
    else:
        with open(gtf_filepath) as f:
            for line in f:
                if line.startswith("#"):
                    continue  # skip header
                chrom, _, label, start, end = line.split()[:5]
                strand = line.split()[6]
                if label != "transcript":
                    regions[label].append((chrom, start, end, strand))
            
    return regions

def load_chrom_sizes(chrom_sizes_filepath, filter_out=["chrUn", "chrM", "chrEBV", "_"]):
    chrom_sizes = get_header_for_bigwig(chrom_sizes_filepath)
    
    filter_chrom_sizes = []
    for chrom, size in chrom_sizes:
        if not any(filt in chrom for filt in filter_out):
            filter_chrom_sizes.append((chrom, 0, size))

    return filter_chrom_sizes


def load_bed_file(filepath):
    if filepath.endswith(".gz"):
        with gzip.open(filepath) as f:
            regions = [line.decode().strip().split() for line in f.readlines()]
    else:
        with open(filepath) as f:
            regions = [line.strip().split() for line in f.readlines()]
    return regions

def write_regions_to_bed_file(regions, filepath):
    regions = sorted(regions, key = lambda region : (region[0], int(region[1])))
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "w") as f:
            for region_info in regions:
                line = "\t".join([str(thing) for thing in region_info]) + "\n"
                f.write(line.encode())
    else:
        with open(filepath, "w") as f:
            for region_info in regions:
                line = "\t".join([str(thing) for thing in region_info]) + "\n"
                f.write(line)


def run_bedtools_subtract(filepath_a, filepath_b, dest_filepath, other_args=[]):
    cmd = ["bedtools", "subtract"]
    cmd += ["-a", filepath_a]
    cmd += ["-b", filepath_b]
    for arg in other_args:
        cmd += [arg]
        
    with open(dest_filepath, "w") as outf:
        subprocess.call(cmd, stdout=outf)

def run_bedtools_intersect(filepath_a, filepath_b, dest_filepath, other_args=[]):
    cmd = ["bedtools", "intersect"]
    cmd += ["-a", filepath_a]
    cmd += ["-b", filepath_b]
    for arg in other_args:
        cmd += [arg]
        
    with open(dest_filepath, "w") as outf:
        subprocess.call(cmd, stdout=outf)
        
def run_bedtools_merge(filepath_i, dest_filepath, other_args=[]):
    cmd = ["bedtools", "merge"]
    cmd += ["-i", filepath_i]
    for arg in other_args:
        cmd += [arg]
        
    with open(dest_filepath, "w") as outf:
        subprocess.call(cmd, stdout=outf)

def sort_bed_file(in_bed, out_bed):
    regions = load_bed_file(in_bed)
    regions = sorted(regions, key = lambda region : (region[0], int(region[1])))
    write_regions_to_bed_file(regions, out_bed)
    
def filter_file_by_chrom(in_bed, out_bed, filter_out=["chrUn", "chrM", "chrEBV", "_"]):
    in_regions = load_bed_file(in_bed)
    
    if out_bed.endswith(".gz"):
        with gzip.open(out_bed, "w") as f:
            for region_info in in_regions:
                chrom = region_info[0]
                if not any(filt in chrom for filt in filter_out):
                    line = "\t".join([str(thing) for thing in region_info]) + "\n"
                    f.write(line.encode())
    else:
        with open(out_bed, "w") as f:
            for region_info in in_regions:
                chrom = region_info[0]
                if not any(filt in chrom for filt in filter_out):
                    line = "\t".join([str(thing) for thing in region_info]) + "\n"
                    f.write(line)


def main():
    proj_dir = get_proj_dir()
    gtf_filepath = proj_dir + "annotations/gencode.v41.annotation.gtf.gz"
    chrom_sizes_filepath = proj_dir + "genomes/hg38.chrom.sizes"
    gtf_regions = load_gtf(gtf_filepath)
    
    genes_bed = proj_dir + "annotations/gene_regions.bed"
    tmp_a = "tmpA.bed"
    tmp_b = "tmpB.bed"
    intergenic_bed = proj_dir + "annotations/intergenic_regions.bed"
    
    write_regions_to_bed_file(gtf_regions["gene"], genes_bed)
    write_regions_to_bed_file(load_chrom_sizes(chrom_sizes_filepath), tmp_a)
    run_bedtools_subtract(tmp_a, genes_bed, tmp_b)
    run_bedtools_merge(tmp_b, intergenic_bed)
    
    exon_bed = proj_dir + "annotations/exons.bed"
    intron_bed = proj_dir + "annotations/introns.bed"
    write_regions_to_bed_file(gtf_regions["exon"], tmp_a)
    run_bedtools_merge(tmp_a, exon_bed)
    run_bedtools_subtract(genes_bed, exon_bed, tmp_a)
    sort_bed_file(tmp_a, tmp_b)
    run_bedtools_merge(tmp_b, intron_bed)

    utr_bed = proj_dir + "annotations/utrs.bed"
    write_regions_to_bed_file(gtf_regions["UTR"], tmp_a)
    run_bedtools_merge(tmp_a, utr_bed)

    os.remove(tmp_a)
    os.remove(tmp_b)
    
    
if __name__ == "__main__":
    main()

import gzip
import os
import subprocess
from collections import defaultdict

import sys
sys.path.append("../2_train_models")
from write_bigwigs import get_header_for_bigwig
from utils import get_proj_dir


def load_gtf(gtf_filepath, region_types_to_load = ["transcript", "gene", "exon", "UTR", "CDS"]):
    regions = defaultdict(lambda : [])
    
    if gtf_filepath.endswith(".gz"):
        f = gzip.open(gtf_filepath)
    else:
        f = open(gtf_filepath)
        
    for line in f:
        if gtf_filepath.endswith(".gz"):
            line = line.decode()
        
        if line.startswith("#"):
            continue  # skip header
        
        chrom, _, label, start, end, _, strand = line.split()[:7]

        if label in region_types_to_load:
            if label == "gene":
                gene_name = line.split()[13][1:-2]
            else:
                gene_name = line.split()[15][1:-2]  # second indexing is to remove "...";

            regions[label].append((chrom, start, end, strand, gene_name))
            
    f.close()
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

def sort_regions_list(regions):
    return sorted(regions, key = lambda region : (region[0], int(region[1])))
        
def sort_bed_file(in_bed, out_bed):
    regions = sort_regions_list(load_bed_file(in_bed))
    write_regions_to_bed_file(regions, out_bed)

    
def make_promoter_regions(transcript_regions, extend_upstream=300, extend_downstream=200):
    promoters = []
    for region in transcript_regions:
        chrom, start, end, strand, _ = region
        if strand == "+":
            TSS = int(start)
            promoter_start = TSS - extend_upstream
            promoter_end = TSS + extend_downstream
        else:
            TSS = int(end)
            promoter_start = TSS - extend_downstream
            promoter_end = TSS + extend_upstream
        
        promoters.append((chrom, str(promoter_start), str(promoter_end), strand))
        
    return promoters

def main():
    proj_dir = get_proj_dir()
    gtf_filepath = proj_dir + "annotations/gencode.v41.annotation.gtf.gz"
    chrom_sizes_filepath = proj_dir + "genomes/hg38.chrom.sizes"

    # output files
    genes_bed = proj_dir + "annotations/gene_regions.bed"
    intergenic_bed = proj_dir + "annotations/intergenic_regions.bed"
    exon_bed = proj_dir + "annotations/exons.bed"
    intron_bed = proj_dir + "annotations/introns.bed"
    utr_bed = proj_dir + "annotations/utrs.bed"
    promoters_bed = proj_dir + "annotations/promoters.bed"
    TSSs_bed = proj_dir + "annotations/TSSs.bed"
    
    # tmp files (will be deleted)
    tmp_a = "tmpA.bed"
    tmp_b = "tmpB.bed"
    
    gtf_regions = load_gtf(gtf_filepath)
    
    write_regions_to_bed_file(gtf_regions["gene"], genes_bed)
    write_regions_to_bed_file(load_chrom_sizes(chrom_sizes_filepath), tmp_a)
    run_bedtools_subtract(tmp_a, genes_bed, tmp_b)
    run_bedtools_merge(tmp_b, intergenic_bed)
    
    write_regions_to_bed_file(gtf_regions["exon"], tmp_a)
    run_bedtools_merge(tmp_a, exon_bed)
    run_bedtools_subtract(genes_bed, exon_bed, tmp_a)
    sort_bed_file(tmp_a, tmp_b)
    run_bedtools_merge(tmp_b, intron_bed)

    write_regions_to_bed_file(gtf_regions["UTR"], tmp_a)
    run_bedtools_merge(tmp_a, utr_bed)

    promoters = make_promoter_regions(gtf_regions["transcript"])
    promoters = sort_regions_list(promoters)
    write_regions_to_bed_file(promoters, promoters_bed)
    
    TSSs = make_promoter_regions(gtf_regions["transcript"],
                                 extend_upstream=1, extend_downstream=0)
    TSSs = sort_regions_list(TSSs)
    write_regions_to_bed_file(TSSs, TSSs_bed)
    
    os.remove(tmp_a)
    os.remove(tmp_b)
    
    
if __name__ == "__main__":
    main()

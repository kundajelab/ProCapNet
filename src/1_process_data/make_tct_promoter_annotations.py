import gzip
import os
import subprocess
from collections import defaultdict

import sys
sys.path.append("../2_train_models")
from data_loading import extract_sequences
from utils import get_proj_dir
from make_gene_region_annotations import load_gtf, load_bed_file, write_regions_to_bed_file
from make_gene_region_annotations import run_bedtools_merge, make_promoter_regions 

        
def run_bedtools_getfasta(filepath_bed, filepath_fasta, dest_filepath, other_args=[]):
    cmd = ["bedtools", "getfasta"]
    cmd += ["-bed", filepath_bed]
    cmd += ["-fi", filepath_fasta]
    for arg in other_args:
        cmd += [arg]
        
    with open(dest_filepath, "w") as outf:
        subprocess.call(cmd, stdout=outf)
    
def load_seqs(getfasta_out):
    with open(getfasta_out) as f:
        seqs = [line.strip().upper() for line in f if not line.startswith(">")]
    return seqs

    
def filter_for_RP_genes(gtf_regions):
    rp_regions = []
    for region in gtf_regions:
        gene_name = region[-1]
        if not (gene_name.startswith("RPS") or gene_name.startswith("RPL")):
            continue
        # this discards read-throughs and kinases
        if "-" in gene_name or "K" in gene_name:
            continue
        # this discards pseudogenes
        if "P" in gene_name[2:]:
            continue
        rp_regions.append(region)
        
    return rp_regions


def TCT_motif_in_seq(seq):
    consensus_motif = "TTCTTT"
    one_off_motifs = ["CTCTTT", "TCCTTT", "TTCCTT", "TTCTCT", "TTCTTC"]
    two_off_motif = "TCTT"
    if consensus_motif in seq:
        return True
    elif any([option in seq for option in one_off_motifs]):
        return True
    elif two_off_motif in seq:
        return True
    return False
    

def main():
    proj_dir = get_proj_dir()
    gtf_filepath = proj_dir + "annotations/gencode.v41.annotation.gtf.gz"
    genome_filepath = proj_dir + "genomes/hg38.withrDNA.fasta"
    gtf_regions = load_gtf(gtf_filepath,
                           region_types_to_load = ["gene"])["gene"]
    assert len(gtf_regions) > 0, len(gtf_regions)
    
    # output files
    rp_promoters_bed = proj_dir + "annotations/rp_promoters.bed"
    tct_promoters_bed = proj_dir + "annotations/tct_promoters.bed"
    
    # filter to only RP transcripts by gene name
    rp_genes = filter_for_RP_genes(gtf_regions)
    assert len(set([r[-1] for r in rp_genes])) == 84, len(set([r[-1] for r in rp_genes]))  # how many we expect in human
    
    # make promoter windows around TSS (keeping strand in mind)
    rp_promoters = make_promoter_regions(rp_genes)
    write_regions_to_bed_file(rp_promoters, rp_promoters_bed)
    
    # get sequence within promoter windows
    tmp_a = proj_dir + "annotations/tmpA.txt"
    run_bedtools_getfasta(rp_promoters_bed, genome_filepath, tmp_a, other_args=["-s"])
    rp_promoter_seqs = load_seqs(tmp_a)
    
    # filter for TCT motif
    has_TCTs = [TCT_motif_in_seq(seq) for seq in rp_promoter_seqs]
    tct_promoters = [p for p, has_TCT in zip(rp_promoters, has_TCTs) if has_TCT]
    
    tmp_b = proj_dir + "annotations/tmpB.bed"
    write_regions_to_bed_file(tct_promoters, tmp_b)
    # merge any overlapping promoters (since we used all transcripts above)
    run_bedtools_merge(tmp_b, tct_promoters_bed)
    
    os.remove(tmp_a)
    os.remove(tmp_b)
    print("Done.")
    
    
if __name__ == "__main__":
    main()

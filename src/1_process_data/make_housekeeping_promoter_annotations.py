import gzip
import os
import subprocess
from collections import defaultdict
import pandas as pd

import sys
sys.path.append("../2_train_models")
from utils import get_proj_dir
from make_gene_region_annotations import load_gtf, load_bed_file, write_regions_to_bed_file
from make_gene_region_annotations import run_bedtools_merge, make_promoter_regions 




def load_housekeeping_gene_names(housekeeping_csv):
    df = pd.read_csv(housekeeping_csv)
    
    gene_names = df["GeneSymbol"][df["JesseSpecificityClass"] == "UbiquitousUniform"]
    gene_names = [name.replace(u'\xa0', u'').replace(".00", "") for name in gene_names]
    return set(gene_names)

    
def filter_for_target_genes(gtf_regions, gene_names):
    target_regions = []
    for region in gtf_regions:
        gene_name = region[-1]
        if not (gene_name in gene_names):
            continue
        
        target_regions.append(region)
        
    return target_regions


def fix_gene_names(name_fix_csv, gene_names):
    df = pd.read_csv(name_fix_csv, header=0)
    df = df[["Input", "Match type", "Approved symbol"]]
    df = df[df["Match type"] != "Unmatched"]
    old_gene_names = df["Input"].values
    new_gene_names = df["Approved symbol"].values
    converter = {old_name : new_name for old_name, new_name in zip(old_gene_names, new_gene_names)}
    
    fixed_gene_names = set()
    for old_name in gene_names:
        if old_name in converter.keys():
            fixed_gene_names.add(converter[old_name])
        else:
            fixed_gene_names.add(old_name)
    
    return fixed_gene_names
    

def main():
    proj_dir = get_proj_dir()
    gtf_filepath = proj_dir + "annotations/gencode.v41.annotation.gtf.gz"
    genome_filepath = proj_dir + "genomes/hg38.withrDNA.fasta"
    gtf_regions = load_gtf(gtf_filepath,
                           region_types_to_load = ["gene", "transcript"])
    assert len(gtf_regions) > 0, len(gtf_regions)
    
    housekeeping_csv = proj_dir + "annotations/JEngreitz_housekeeping_genes.csv"
    hk_gene_names = load_housekeeping_gene_names(housekeeping_csv)
    print(len(hk_gene_names))
    
    # I downloaded this from https://www.genenames.org/tools/multi-symbol-checker/
    # after I ran this code once, and input every gene name that I hadn't found a match for
    name_fix_csv = proj_dir + "annotations/gene_name_fixes.csv"
    fixed_hk_gene_names = fix_gene_names(name_fix_csv, hk_gene_names)
    
    # output file
    hk_promoters_bed = proj_dir + "annotations/hk_promoters.bed"
    hk_promoters_tx_bed = proj_dir + "annotations/hk_promoters_by_transcripts.bed"
    
    # filter to only HK transcripts by gene name
    hk_genes = filter_for_target_genes(gtf_regions["gene"], fixed_hk_gene_names)
    hk_transcripts = filter_for_target_genes(gtf_regions["transcript"], fixed_hk_gene_names)
    print(len(hk_genes), len(hk_transcripts))
    
    #found_hk_gene_names = set([tup[-1] for tup in hk_genes])
    #print(sorted(list(fixed_hk_gene_names - found_hk_gene_names)))
    
    # make promoter windows around TSS (keeping strand in mind)
    hk_promoters = make_promoter_regions(hk_genes)
    write_regions_to_bed_file(hk_promoters, hk_promoters_bed)
    
    hk_promoters_tx = make_promoter_regions(hk_genes)
    tmp_filepath = proj_dir + "annotations/tmp.bed"
    write_regions_to_bed_file(hk_promoters_tx, tmp_filepath)
    run_bedtools_merge(tmp_filepath, hk_promoters_tx_bed)
    
    os.remove(tmp_filepath)

    print("Done.")
    
    
if __name__ == "__main__":
    main()

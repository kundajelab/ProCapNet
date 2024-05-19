import os
import gzip
from collections import defaultdict

from make_gene_region_annotations import write_regions_to_bed_file
from make_gene_region_annotations import run_bedtools_merge, make_promoter_regions 


def load_gtf_with_gene_type_tag(gtf_filepath, region_types_to_load = ["gene"]):
    # not tested with other region_types for correct tag loading
    
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
            gene_type = line.split("\t")[-1].split()[3][1:-2] # last indexing is to remove "..."; 

            regions[label].append((chrom, start, end, strand, gene_type))
            
    f.close()
    return regions

    
def filter_for_gene_function(gtf_regions, target_gene_type):
    target_regions = []
    for region in gtf_regions:
        gene_type = region[-1]
        if not (target_gene_type in gene_type):
            continue
        
        target_regions.append(region)
        
    return target_regions

    

def main():
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/"
    
    gtf_filepath = proj_dir + "annotations/gencode.v41.annotation.gtf.gz"
    genome_filepath = proj_dir + "genomes/hg38.withrDNA.fasta"
    gtf_regions = load_gtf_with_gene_type_tag(gtf_filepath,
                           region_types_to_load = ["gene"])
    assert len(gtf_regions) > 0, len(gtf_regions)
    
    # output file
    protein_coding_promoters_bed = proj_dir + "annotations/promoters_protein_coding.bed"
    lncRNA_promoters_bed = proj_dir + "annotations/promoters_lncRNA.bed"
    
    # filter to only HK transcripts by gene name
    protein_coding_genes = filter_for_gene_function(gtf_regions["gene"], "protein_coding")
    lncRNA_genes = filter_for_gene_function(gtf_regions["gene"], "lncRNA")
    
    print("Num. protein_coding_genes:", len(protein_coding_genes))
    print("Num. lncRNA_genes:", len(lncRNA_genes))
    
    # make promoter windows around TSS (keeping strand in mind)
    protein_coding_promoters = make_promoter_regions(protein_coding_genes)
    write_regions_to_bed_file(protein_coding_promoters, protein_coding_promoters_bed)

    lncRNA_promoters = make_promoter_regions(lncRNA_genes)
    write_regions_to_bed_file(lncRNA_promoters, lncRNA_promoters_bed)
    
    print("Done making gene function annotations.")
    
    
if __name__ == "__main__":
    main()

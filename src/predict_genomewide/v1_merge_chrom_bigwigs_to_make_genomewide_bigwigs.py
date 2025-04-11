import os
import numpy as np
import pyBigWig
import sys

assert len(sys.argv) == 2, sys.argv
cell_type = sys.argv[1]

which_genome = "hg38"

chroms_to_use = ["chr" + str(i+1) for i in range(22)] + ["chrX", "chrY"]

chrom_sizes_filepath = which_genome + "/" + which_genome + ".chrom.sizes"



def load_chrom_sizes(chrom_sizes_filepath, chroms_to_use):
    with open(chrom_sizes_filepath) as f:
        chrom_sizes_lines = [line.strip().split('\t') for line in f]
    chrom_sizes = {line[0] : int(line[1]) for line in chrom_sizes_lines if line[0] in chroms_to_use}
    return chrom_sizes

chrom_sizes = load_chrom_sizes(chrom_sizes_filepath, chroms_to_use)



def get_chrom_bw_path(cell_type, chrom, pos_or_neg, which_genome = which_genome):
    assert pos_or_neg in ["pos", "neg"], pos_or_neg
    bws_dir = "/".join(["bigwigs", which_genome, cell_type, chrom]) + "/"
    bw_path = bws_dir + ".".join([chrom, cell_type, pos_or_neg, "bigWig"])
    assert os.path.exists(bw_path), bw_path
    return bw_path


def get_wg_bw_path(cell_type, pos_or_neg, which_genome = which_genome):
    assert pos_or_neg in ["pos", "neg"], pos_or_neg
    bws_dir = "/".join(["bigwigs", which_genome, cell_type, "genomewide"]) + "/"
    os.makedirs(bws_dir, exist_ok=True)
    assert os.path.exists(bws_dir), bws_dir
    bw_path = bws_dir + ".".join([cell_type, pos_or_neg, "bigWig"])
    return bw_path


def load_chromosome_wide_preds(chrom, cell_type, strand, chrom_sizes=chrom_sizes):
    bw_path = get_chrom_bw_path(cell_type, chrom, strand)
    bw = pyBigWig.open(bw_path, "r")

    preds = bw.values(chrom, 0, chrom_sizes[chrom], numpy=True)

    bw.close()
    return preds



def merge_chrom_bigwigs_whole_genome(cell_type, pos_or_neg,
                                     chrom_sizes = chrom_sizes,
                                     which_genome = which_genome):
    
    wg_bw_path = get_wg_bw_path(cell_type, pos_or_neg, which_genome = which_genome)
    
    print("Writing predicted profiles to bigwigs.")
    print("Save path: " + wg_bw_path)
    
    # bigwigs need to be written in order -- so we have to go chromosome by chromosome,
    # and the chromosomes need to be numerically sorted (i.e. chr9 then chr10)
    
    chroms = list(chrom_sizes.keys())

    # do I need to sort this list here? ^
    
    wg_bw = pyBigWig.open(wg_bw_path, 'w')
    
    # bigwigs need headers before they can be written to
    # the header is just the info you'd find in a chrom.sizes file
    wg_bw.addHeader(list(chrom_sizes.items()))
    
    for chrom in chroms:
        print(chrom)
        chrom_preds = load_chromosome_wide_preds(chrom, cell_type, pos_or_neg,
                                                 chrom_sizes=chrom_sizes)
    
        # pybigwig will throw error if you don't do this
        chrom_preds = chrom_preds.astype("float64").squeeze()  
        assert len(chrom_preds.shape) == 1, chrom_preds.shape
        assert len(chrom_preds) == chrom_sizes[chrom], (chrom, len(chrom_preds), chrom_sizes[chrom])

        starts = np.arange(len(chrom_preds))
        ends = starts + 1
        
        wg_bw.addEntries([chrom for _ in range(len(starts))], 
                       starts, ends = ends, values = chrom_preds)
    
    wg_bw.close()
        
        
for strand in ["pos", "neg"]:
    print("Cell type:", cell_type)
    print("Strand:", strand)

    merge_chrom_bigwigs_whole_genome(cell_type, strand)


print("Done!")


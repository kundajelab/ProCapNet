# Code modified from an earlier version of Anusri's ChromBPNet's GC-matching:
# https://github.com/kundajelab/chrombpnet/tree/be0fe06cf0f2dec3147c87cf08e81ec831881fec/chrombpnet/helpers/make_gc_matched_negatives


from pyfaidx import Fasta
from tqdm import tqdm 
import pandas as pd
import os, sys



genome_fasta = sys.argv[1]
input_bed = sys.argv[2]
output_bed = sys.argv[3]
FLANK_SIZE = int(sys.argv[4]) // 2
assert os.path.exists(input_bed), input_bed


def main():
    print("Calculating the CpG content of the peaks...")
    ref=Fasta(genome_fasta)
    
    if input_bed.endswith(".gz"):
        data=pd.read_csv(input_bed,header=None,sep='\t',compression="gzip")
    else:
        data=pd.read_csv(input_bed,header=None,sep='\t')

    num_rows=data.shape[0]
    print("Number of peaks:", num_rows) 

    cpg_fracs = []
    for index,row in tqdm(data.iterrows()):
        chrom=row[0]
        start=row[1]
        end=row[2] 

        summit=(start+end)//2
        start=summit - FLANK_SIZE
        end=summit + FLANK_SIZE

        # calculate CpG content when centered at summit
        seq=ref[chrom][start:end].seq.upper()
        cpg=seq.count('CG') * 2
        cpg_fract=round(cpg/len(seq),2)                
        cpg_fracs.append(cpg_fract)
    ref.close()
    
    assert len(cpg_fracs) == num_rows, (len(cpg_fracs), num_rows)

    data["CpG"] = cpg_fracs
    if output_bed.endswith(".gz"):
        data.to_csv(output_bed, sep='\t',compression="gzip",index=False,header=False)
    else:
        data.to_csv(output_bed, sep='\t',index=False,header=False)
    print("Done.")

        
if __name__=="__main__":
    main()

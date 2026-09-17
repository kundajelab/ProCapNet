# Code modified from an earlier version of Anusri's ChromBPNet's GC-matching:
# https://github.com/kundajelab/chrombpnet/tree/be0fe06cf0f2dec3147c87cf08e81ec831881fec/chrombpnet/helpers/make_gc_matched_negatives


import pyfaidx
import argparse
import gzip

def parse_args():
    parser=argparse.ArgumentParser(description="get CpG content after binning the entire genome into bins")
    parser.add_argument("-g","--genome", required=True, help="reference genome file")
    parser.add_argument("-o","--output_bed", required=True, help="output BED file to store the CpG content of binned genome. If this path contains a directory make sure it exists.")
    parser.add_argument("-f","--inputlen", type=int,default=2114, help="inputlen to use to find CpG content")
    parser.add_argument("-s","--stride", type=int,default=1000, help="stride to use for shifting the bins")
    return parser.parse_args()

def get_genomewide_cpg(genome_fa, outfname, width, stride):
    """
    Get CpG fraction in bins of width "width" strided by "stride".

    Main speedups come from:
    - loading chromosome string using pyfaidx
    - using the str.count function for substrings
    - avoiding redundant counting

    Redundant counting is avoided by counting in bins of size "stride"
    at a time and caching the most recent values in cache. As an example:

    For width 2114 and stride 1000, when considering the 3000-4000 bin, 
    with already cached counts in 1000-2000 and 2000-3000, count in
    3000-3114 and write 1000-3114. Then count in 3114-4000, now cache 
    3000-4000 and delete 1000-2000. And repeat.
    """

    f = pyfaidx.Fasta(genome_fa, as_raw=True)

    if outfname.endswith(".gz"):
        outf = gzip.open(outfname, 'wb')
    else:
        outf = open(outfname, 'w')

    div = width//stride
    rem = width%stride
    stride_x_div = div * stride

    # cache will store the CpG counts in the most recent
    # div bins of length stride each
    cache = [0]*div

    for chrm in f.keys():
        s = f[chrm][:].upper()

        runsum = 0
        # fill first div values
        for i in range(0, stride_x_div, stride):
            # the * 2 is because we will divide by # of bases later, and CG is two bases  - Kelly
            c = s.count("CG", i, i+stride) * 2
            cache[i//stride % div] = c
            runsum += c
        
        for i in range(div*stride, len(s)-rem, stride):
            # invariant: runsum = sum(cache)
            left_ct=0
            if rem!=0:
                left_ct = s.count("CG", i, i+rem) * 2
                runsum += left_ct

            #print("In for loop in _get_genomewide_bins...")
            to_write = "{}\t{}\t{}\t{}\n".format(chrm, i - stride_x_div, i - stride_x_div + width, round(runsum/width,2))
            #print("To write:", to_write)
            #print("Again but encoded", to_write.encode())
            if outfname.endswith(".gz"):
                outf.write(to_write.encode())
            else:
                outf.write(to_write)

            if div == 0: # stride > width, do no more
                runsum = 0
            else:
                runsum -= cache[i//stride % div]

                right_ct = s.count("CG", i+rem, i+stride) * 2
                runsum += right_ct
                cache[i//stride % div] = left_ct+right_ct 

    f.close()
    outf.close()

if __name__=="__main__":
    args = parse_args()
    get_genomewide_cpg(args.genome, args.output_bed, args.inputlen, args.stride)


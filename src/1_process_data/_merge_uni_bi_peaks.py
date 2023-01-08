import gzip
import sys
import os


# File format we're aiming for: columns are chrom, start, end, strand, confidence, uni-or-bi-directional, "summit(s)" on + strand, "summit(s)" on - strand
# Since only bidirectional peaks have confidence labels, that column is empty for unidirectional peaks
# Since only unidirectional peaks have strand labels, the strand is set to "Both" for bidirectional peaks


def merge_uni_bi_peaks(uni_peaks_filename, bi_peaks_filename):
    # First, read in all unidirectional peaks and re-format lines
    uni_peaks = []
    # expecting gzipped input for both files
    with gzip.open(uni_peaks_filename) as uni_peaks_file:
        for line in uni_peaks_file:
            line = line.decode().split()
            chrom, start, end = line[:3]
            strand = line[5]
            summit = line[-2]
            if strand == "+":
                uni_peaks.append((chrom, start, end, strand, ".", "Unidirectional", summit, "."))
            else:
                uni_peaks.append((chrom, start, end, strand, ".", "Unidirectional", ".", summit))

    # Then, read in all bidirectional peaks and possibly re-format lines
    bi_peaks = []
    with gzip.open(bi_peaks_filename) as bi_peaks_file:
        for line in bi_peaks_file:
            line = line.decode().split()
            chrom, start, end, confidence, summits_pos, summits_neg = line[:6]
            bi_peaks.append((chrom, start, end, "Both", confidence, "Bidirectional", summits_pos, summits_neg))

    # Sort all peaks accoridng to same rule as `sort -k1,1 -k2,2n`
    all_peaks = sorted(uni_peaks + bi_peaks, key = lambda tup : (tup[0], int(tup[1])))

    return all_peaks


def write_to_tsv(filename, line_tuples):
    # output will also be gzipped
    with gzip.open(filename, "wb") as tsv_file:
        for line in line_tuples:
            line_to_write = ("\t".join(line) + "\n")
            tsv_file.write(line_to_write.encode())



if __name__ == "__main__":
    assert len(sys.argv) == 4
    uni_peaks_filename, bi_peaks_filename, merged_peaks_filename = sys.argv[1:]
    
    assert os.path.exists(uni_peaks_filename), uni_peaks_filename
    assert os.path.exists(bi_peaks_filename), bi_peaks_filename

    all_peaks = merge_uni_bi_peaks(uni_peaks_filename, bi_peaks_filename)
    write_to_tsv(merged_peaks_filename, all_peaks)

import random
import gzip
import sys
import os


# Assuming human chromosomes here

# Which chromosomes will be in the training, validation, and test sets:
ALL_CHROMS = ["chr" + str(i + 1) for i in range(22)] + ["chrX", "chrY"]

FOLDS = [["chr1", "chr4"],
         ["chr2", "chr13", "chr16"],
         ["chr5", "chr6", "chr20", "chr21"],
         ["chr7", "chr8", "chr9"],
         ["chr10", "chr11", "chr12"],
         ["chr3", "chr14", "chr15", "chr17"],
         ["chr18", "chr19", "chr22", "chrX", "chrY"]]


def get_chroms_for_fold(fold_num):
    test_chroms = FOLDS[fold_num]
    val_chroms = FOLDS[(fold_num + 1) % len(FOLDS)]
    train_chroms = [chrom for chrom in ALL_CHROMS if chrom not in test_chroms + val_chroms]
    return train_chroms, val_chroms, test_chroms


def split_peaks_by_chrom(peaks_filename, fold_num):
    train_chroms, val_chroms, test_chroms = get_chroms_for_fold(fold_num)
    
    train_lines = []
    val_lines = []
    test_lines = []
    with gzip.open(peaks_filename) as peaks_file:
        for line in peaks_file:
            line = line.decode().rstrip()
            chrom = line.split()[0]

            if chrom in val_chroms:
                val_lines.append(line)
            elif chrom in test_chroms:
                test_lines.append(line)
            elif chrom in train_chroms:
                train_lines.append(line)

    return train_lines, val_lines, test_lines


def write_peaks_to_file(lines_to_write, filename):
    # shuffle first, to avoid looking at overlapping examples manually later
    random.shuffle(lines_to_write)

    with gzip.open(filename, "wb") as out_file:
        out_file.write(("\n".join(lines_to_write) + "\n").encode())


def split_peaks_and_write_to_files(peaks_filename, fold_num):
    # split all peaks according to chromosome
    train_lines, val_lines, test_lines = split_peaks_by_chrom(peaks_filename, fold_num)

    # write peak sets to files (will be gzipped)
    train_filename = peaks_filename.replace(".bed.gz", "_fold" + str(fold_num + 1) + "_train.bed.gz")
    write_peaks_to_file(train_lines, train_filename)

    val_filename = peaks_filename.replace(".bed.gz", "_fold" + str(fold_num + 1) + "_val.bed.gz")
    write_peaks_to_file(val_lines, val_filename)

    if len(test_lines) > 0:
        test_filename = peaks_filename.replace(".bed.gz", "_fold" + str(fold_num + 1) + "_test.bed.gz")
        write_peaks_to_file(test_lines, test_filename)

    if "train_and_val" not in peaks_filename:
        train_and_val_lines = train_lines + val_lines
        random.shuffle(train_and_val_lines)
        train_and_val_filename = peaks_filename.replace(".bed.gz", "_fold" + str(fold_num + 1) + "_train_and_val.bed.gz")
        write_peaks_to_file(train_and_val_lines, train_and_val_filename)

    print("Done splitting peaks from " + peaks_filename + " for fold " + str(fold_num + 1) + ".")
    total_peaks = len(train_lines) + len(val_lines) + len(test_lines)
    print("Peaks in training set:", len(train_lines), " (%0.3f" % (len(train_lines) * 100 / total_peaks) + "%)")
    print("Peaks in val set:", len(val_lines), " (%0.3f" % (len(val_lines) * 100 / total_peaks) + "%)")
    print("Peaks in test set:", len(test_lines), " (%0.3f" % (len(test_lines) * 100 / total_peaks) + "%)")


if __name__ == "__main__":
    assert len(sys.argv) == 2
    peaks_filename = sys.argv[1]
    assert os.path.exists(peaks_filename), peaks_filename
    assert peaks_filename.endswith(".gz"), peaks_filename

    for fold_num in range(len(FOLDS)):
        split_peaks_and_write_to_files(peaks_filename, fold_num)

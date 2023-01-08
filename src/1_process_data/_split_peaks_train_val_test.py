import random
import gzip
import sys
import os


# Assuming human chromosomes here

# Which chromosomes will be in the training, validation, and test sets:
ALL_CHROMS = ["chr" + str(i + 1) for i in range(22)] + ["chrX", "chrY"]
VAL_CHROMS = ["chr18", "chr19", "chr20", "chr21"]  # ~10% of genome
TEST_CHROMS = ["chr13", "chr14", "chr15", "chr16"]  # ~10% of genome
TRAIN_CHROMS = [chrom for chrom in ALL_CHROMS if chrom not in VAL_CHROMS + TEST_CHROMS]


def split_peaks_by_chrom(peaks_filename):
    train_lines = []
    val_lines = []
    test_lines = []
    with gzip.open(peaks_filename) as peaks_file:
        for line in peaks_file:
            line = line.decode().rstrip()
            chrom = line.split()[0]

            if chrom in VAL_CHROMS:
                val_lines.append(line)
            elif chrom in TEST_CHROMS:
                test_lines.append(line)
            elif chrom in TRAIN_CHROMS:
                train_lines.append(line)

    return train_lines, val_lines, test_lines


def write_peaks_to_file(lines_to_write, filename):
    # shuffle first, to avoid looking at overlapping examples manually later
    random.shuffle(lines_to_write)

    with gzip.open(filename, "wb") as out_file:
        out_file.write(("\n".join(lines_to_write) + "\n").encode())


def split_peaks_and_write_to_files(peaks_filename):
    # split all peaks according to chromosome
    train_lines, val_lines, test_lines = split_peaks_by_chrom(peaks_filename)

    # write peak sets to files (will be gzipped)
    train_filename = peaks_filename.replace(".bed.gz", "_train.bed.gz")
    write_peaks_to_file(train_lines, train_filename)

    val_filename = peaks_filename.replace(".bed.gz", "_val.bed.gz")
    write_peaks_to_file(val_lines, val_filename)

    if len(test_lines) > 0:
        test_filename = peaks_filename.replace(".bed.gz", "_test.bed.gz")
        write_peaks_to_file(test_lines, test_filename)

    if "train_and_val" not in peaks_filename:
        train_and_val_lines = train_lines + val_lines
        random.shuffle(train_and_val_lines)
        train_and_val_filename = peaks_filename.replace(".bed.gz", "_train_and_val.bed.gz")
        write_peaks_to_file(train_and_val_lines, train_and_val_filename)

    print("Done splitting peaks from " + peaks_filename)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    peaks_filename = sys.argv[1]
    assert os.path.exists(peaks_filename), peaks_filename
    assert peaks_filename.endswith(".gz"), peaks_filename

    split_peaks_and_write_to_files(peaks_filename)

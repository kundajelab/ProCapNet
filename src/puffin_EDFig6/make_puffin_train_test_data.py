import pandas as pd
import numpy as np
import os
import gzip


# this script exactly follows how the puffin repo code
# splits the training and test regions, e.g.
# https://github.com/jzhoulab/puffin_manuscript/blob/main/train/train_puffin_stage1.py
# (confirmed with Ksenia)


out_dir = "puffin_data_train_val_test_split/"
os.makedirs(out_dir, exist_ok=True)


# files downloaded from the puffin repo

tsses_tsv = 'data/resources/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v5.tsv'
tsses_hc_tsv = 'data/resources/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v5.highconf.tsv'

tsses = pd.read_table(tsses_tsv, sep='\t')
tsses_hc = pd.read_table(tsses_hc_tsv, sep='\t')


# determine which TSSs are in the test set

holdoutinds = np.isin(tsses['chr'].iloc[:100000], ['chr9', 'chr8'])
hcinds = np.isin(tsses.iloc[:100000, 0].values, tsses_hc.iloc[:100000, 0].values)

which_test_TSSs = hcinds * holdoutinds
test_TSSs = tsses.iloc[:100000][which_test_TSSs]


# once TSSs are catgorized, save them in ProCapNet-friendly format

def make_bed_windows_from_dataframe(df, out_window=1000):
    chroms = df["chr"]
    strands = df["strand"]
    tss_bases = df["TSS"]
    
    window_starts = tss_bases - out_window//2
    window_ends = window_starts + out_window
    
    bed_info = zip(chroms, window_starts, window_ends, strands)
    return bed_info


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
                
                
test_bed_info = make_bed_windows_from_dataframe(test_TSSs)
write_regions_to_bed_file(test_bed_info, out_dir + "test_set_from_ksenia.bed.gz")


# train set

holdoutinds_train = ~ np.isin(tsses['chr'], ['chr9', 'chr8', "chr10"])
hcinds_train = np.isin(tsses.iloc[:, 0].values, tsses_hc.iloc[:40000, 0].values)

which_train_TSSs = hcinds_train * holdoutinds_train
train_TSSs = tsses[which_train_TSSs]

train_bed_info = make_bed_windows_from_dataframe(train_TSSs)
write_regions_to_bed_file(train_bed_info, out_dir + "train_set_from_ksenia.bed.gz")


# val set

holdoutinds_val = np.isin(tsses['chr'], ["chr10"])
hcinds_val = np.isin(tsses.iloc[:, 0].values, tsses_hc.iloc[:40000, 0].values)

which_val_TSSs = hcinds_val * holdoutinds_val
val_TSSs = tsses[which_val_TSSs]

val_bed_info = make_bed_windows_from_dataframe(val_TSSs)
write_regions_to_bed_file(val_bed_info, out_dir + "val_set_from_ksenia.bed.gz")

print("Done making puffin train-val-test split.")

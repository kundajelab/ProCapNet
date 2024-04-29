#!/bin/bash

# (this is for K562)

# these filepaths work if your directories are set up the way the setup script does it

procap_peak_bed="../../data/procap/processed/K562/peaks.bed.gz"
dnase_peak_bed="../../annotations/K562/DNase_peaks.bed.gz"

num_procap_peaks=`zcat $procap_peak_bed | wc -l`
echo "num_procap_peaks: $num_procap_peaks"

tmp="deleteme.bed"
tmp2="deleteme2.bed"
#tmp3="deleteme3.bed"

# need to gzip
zcat "$procap_peak_bed" | awk -v OFS="\t" '{ print $1, int(($2 + $3) / 2) - 500, int(($2 + $3) / 2) + 500 }' | sort -k1,1 -k2,2n > "$tmp2"
#zcat "$procap_peak_bed" > "$tmp1"

#zcat "$procap_peak_bed" | awk '{ print $3 - $2 }' | head -n50
#zcat "$dnase_peak_bed" | awk '{ print $3 - $2 }' | head -n50

bedtools intersect -a "$tmp2" -b "$dnase_peak_bed" -wa -u > "$tmp"

num_procap_peaks_overlapping_dnase_peaks=`cat $tmp | wc -l`
echo "num_procap_peaks_overlapping_dnase_peaks: $num_procap_peaks_overlapping_dnase_peaks"

frac=$((num_procap_peaks_overlapping_dnase_peaks / num_procap_peaks))
echo "frac_procap_peaks_overlapping_dnase_peaks:"

echo "$num_procap_peaks_overlapping_dnase_peaks / $num_procap_peaks" | bc -l

exit 0

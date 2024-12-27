import sys
assert len(sys.argv) == 3, len(sys.argv)  # expecting cell type, GPU ID
cell_type, GPU = sys.argv[1], sys.argv[2]

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import numpy as np
from collections import defaultdict
import gzip
import pandas as pd
import torch
from tqdm import tqdm
from pyfaidx import Fasta
import pyBigWig

import sys
sys.path.append("../2_train_models")
from data_loading import one_hot_encode
from file_configs import FoldFilesConfig
from BPNet_strand_merged_umap import Model

sys.path.append("../utils")
from misc import load_chrom_sizes, load_chrom_names


# For each cell type, we have a set of ProCapNet models trained across 7 folds,
# which are ID'd by their training timestamps below

if cell_type == "K562":
    timestamps = ["2023-05-29_15-51-40", "2023-05-29_15-58-41", "2023-05-29_15-59-09",
                  "2023-05-30_01-40-06", "2023-05-29_23-21-23", "2023-05-29_23-23-45",
                  "2023-05-29_23-24-11"]
elif cell_type == "A673":
    timestamps = ["2023-06-11_20-11-32","2023-06-11_23-42-00", "2023-06-12_03-29-06",
                  "2023-06-12_07-17-43", "2023-06-12_11-10-59", "2023-06-12_14-36-40",
                  "2023-06-12_17-26-09"]
elif cell_type == "CACO2":
    timestamps = ["2023-06-12_21-46-40", "2023-06-13_01-28-24", "2023-06-13_05-06-53",
                  "2023-06-13_08-52-39", "2023-06-13_13-12-09", "2023-06-13_16-40-41",
                  "2023-06-13_20-08-39"]
elif cell_type == "CALU3":
    timestamps = ["2023-06-14_00-43-44", "2023-06-14_04-26-48", "2023-06-14_09-34-26",
              "2023-06-14_13-03-59", "2023-06-14_17-22-28", "2023-06-14_21-03-11",
              "2023-06-14_23-58-36"]
elif cell_type == "HUVEC":
    timestamps = ["2023-06-16_21-59-35", "2023-06-17_00-20-34", "2023-06-17_02-17-07",
                  "2023-06-17_04-27-08", "2023-06-17_06-42-19", "2023-06-17_09-16-24",
                  "2023-06-17_11-09-38"] 
elif cell_type == "MCF10A":
    timestamps = ["2023-06-15_06-07-40", "2023-06-15_10-37-03", "2023-06-15_16-23-56",
                  "2023-06-15_21-44-32", "2023-06-16_03-47-46", "2023-06-16_09-41-26",
                  "2023-06-16_15-07-01"]
else:
    print("Misspelled cell type?")
    quit()
    
    
# where we will save all of the predictions generated
out_dir = "predictions/" + cell_type + "/"
os.makedirs(out_dir, exist_ok=True)


# the input sequence length and output prediction window length
in_window = 2114
out_window = 1000

genome_path = "genome/hg38.gencode_naming.withrDNA.fasta"
chrom_sizes = "genome/hg38.gencode_naming.chrom.sizes"

# load chromosome info
chrom_sizes_dict = {k : v for (k,v) in load_chrom_sizes(chrom_sizes)}
chrom_names = sorted(list(chrom_sizes_dict.keys()))


# everywhere to generate predictions for: made by CLS_collab_make_regions.ipynb

all_regions_to_predict_filepath = "regions_to_predict/TSS_windows.merged.bed.gz"



# functions that return filepaths

def get_fold_config(fold, timestamps, cell_type, model_type = "strand_merged_umap"):
    return FoldFilesConfig(cell_type, model_type, str(fold + 1), timestamps[fold], "procap")

def get_regions_to_predict_chrom_filepath(chrom):
    return "regions_to_predict/TSS_windows.merged." + chrom + ".bed.gz"




### Load in genome sequences at pre-defined regions


def read_fasta_fast(filename, include_chroms=chrom_names, verbose=True):
    chroms = {}
    print("Loading genome sequence from " + filename)
    fasta_index = Fasta(filename)
    for chrom in tqdm(include_chroms, disable=not verbose, desc="Reading FASTA"):
        chroms[chrom] = fasta_index[chrom][:].seq.upper()
    return chroms


def load_sequences(genome_path, chrom_sizes, peak_path, verbose=False):

    seqs = []
    seq_window_pad = (in_window - out_window) // 2

    assert os.path.exists(genome_path), genome_path
    sequences = read_fasta_fast(genome_path, verbose=verbose)

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    desc = "Loading Peaks"
    d = not verbose
    for _, (chrom, og_start, og_end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        # Append sequence to growing sequence list
        
        # get sequence from fasta for this chromosome
        chrom_sequence = sequences[chrom]
        
        # determine beginning and end of region (extend out by model input width)
        
        # if a single model prediction would cover the window we want to predict:
        if og_end - og_start < out_window:
            # then just make a window of len (model input size)
            mid = (og_start + og_end) // 2
            s = max(0, mid - in_window // 2)
            e = s + in_window
            assert e - s == in_window, (s, e)
        
        # otherwise, just add enough sequence to either side of the window so that
        # predictions will cover the whole window
        else:
            s = max(0, og_start - seq_window_pad)
            e = og_end + seq_window_pad
            assert s >= 0, s
            assert e - s >= in_window, (s, e)

        if isinstance(chrom_sequence, str):
            seq = one_hot_encode(chrom_sequence[s:e]).T
        else:
            seq = chrom_sequence[s:e].T

        assert seq.shape == (4, e - s), (seq.shape, s, e, e - s, chrom, og_start, og_end, chrom_sizes_dict[chrom])
        assert set(seq.flatten()) == set([0,1]), set(seq.flatten())
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert set(seq.sum(axis=0)).issubset(set([0, 1])), set(seq.sum(axis=0))
        assert seq.sum() <= e - s, seq
        seqs.append(seq)
        
    to_print = "\nPeak filepath: " + peak_path
    to_print += "\nNum. examples loaded: " + str(len(seqs))
    print(to_print)
    sys.stdout.flush()

    return seqs




### Load ProCapNet models (all folds)

def load_model(model_save_path):
    model = torch.load(model_save_path)
    model = model.eval()
    return model

def load_all_models(timestamps, cell_type):
    models = []
    for fold in range(len(timestamps)):
        model_path = get_fold_config(fold, timestamps, cell_type).model_save_path
        models.append(load_model(model_path))
    return models





### Generate predictions

# for each region:
# chunk sequence into 2114bp tiles (model input size), with small stride
# for each chunk, predict
# put predictions into giant numpy array
# merge prediction windows where they overlap


def model_predict_with_rc(model, onehot_seq):
    # this function makes a prediction for 1 model, for 1 sequence
    
    # it gets a prediction for both the original sequence and its reverse-complement,
    # then averages the two, which tends to improve accuracy
    
    model = model.cuda()
    with torch.no_grad():
        onehot_seq = onehot_seq[None, ...].cuda()
        pred_profiles, pred_logcounts = model.predict(onehot_seq)
        rc_pred_profiles, rc_pred_logcounts = model.predict(torch.flip(onehot_seq, [-1, -2]))
    
    model = model.cpu()
    
    # reverse-complement (strand-flip) BOTH the profile and counts predictions
    rc_pred_profiles = rc_pred_profiles[:, ::-1, ::-1]
    rc_pred_logcounts = rc_pred_logcounts[:, ::-1]
    
    # take the average prediction across the fwd and RC sequences
    # (profile average is in raw probs space, not logits; counts average is in log counts space)
    merged_pred_profiles = np.array([np.exp(pred_profiles), np.exp(rc_pred_profiles)]).mean(axis=0)
    merged_pred_logcounts = np.array([pred_logcounts, rc_pred_logcounts]).mean(axis=0)
    
    return merged_pred_profiles, merged_pred_logcounts


def predict_one_model(model, onehot_seq):
    with torch.no_grad():
        pred_prof, pred_logcounts = model_predict_with_rc(model, onehot_seq)
    return pred_prof.squeeze(), pred_logcounts.squeeze()


def predict_all_models(models, onehot_seq):
    # if we can, we want to just do this function when possible -- it is all-in-one
    with torch.no_grad():
        # generate preds for this seq across all model folds
        pred_profs_across_models = []
        pred_logcounts_across_models = []
        for model in models:
            pred_prof, pred_logcounts = model_predict_with_rc(model, onehot_seq)
            pred_profs_across_models.append(pred_prof)
            pred_logcounts_across_models.append(pred_logcounts)

        # average predictions across folds
        pred_prof_avg_across_models = np.mean(np.array(pred_profs_across_models), axis=0)
        pred_counts_avg_across_models = np.exp(np.mean(np.array(pred_logcounts_across_models), axis=0))

        # combine profile and counts preds into one scaled-profile prediction
        pred_prof_scaled = (pred_prof_avg_across_models * pred_counts_avg_across_models).squeeze()
        return pred_prof_scaled


def predict_one_seq(onehot_seq, models, skip = 50):
    # "skip" = the stride/offset between windows where predictions are generated
    
    onehot_seq = torch.Tensor(onehot_seq).float()
    assert len(onehot_seq.shape) == 2 and onehot_seq.shape[0] == 4, onehot_seq.shape
    
    
    # If we are lucky, the seq length will match the model's input length.
    # In that case, just predict normally (no need to tile predictions).
    
    if onehot_seq.shape[-1] == in_window:
        return predict_all_models(models, onehot_seq)
    
    # Else, we need to tile:
    
    num_seq_tiles = int(np.ceil((onehot_seq.shape[-1] - in_window) / skip + 1))
    tiles_to_do = list(skip * np.arange(0, num_seq_tiles - 1))

    assert len(tiles_to_do) > 0, (onehot_seq.shape, num_seq_tiles, onehot_seq.shape[-1] - in_window)
    # if not a nice even number of tiles to do vs. skips, tack on last window
    if tiles_to_do[-1] != onehot_seq.shape[1] - in_window:
        tiles_to_do.append(onehot_seq.shape[1] - in_window)
    
    assert len(tiles_to_do) == num_seq_tiles, (len(tiles_to_do), num_seq_tiles)
    
    
    # initialize mega-numpy-arrays to fill in in for loops below
    pred_profs_all = np.empty((len(models), num_seq_tiles, 2, onehot_seq.shape[-1] - (in_window - out_window)))
    pred_logcounts_all = np.empty((len(models), num_seq_tiles))
    pred_profs_all[:] = np.nan


    for model_i, model in enumerate(models):
        for tile_i, tile_i_skip in enumerate(tiles_to_do):
            assert tile_i_skip + in_window <= onehot_seq.shape[1], (tile_i_skip + in_window, onehot_seq.shape[1])
            
            # get sequence for this tile
            seq_tile = onehot_seq[:, tile_i_skip : tile_i_skip + in_window]
            
            # predict
            pred_prof, pred_logcounts = predict_one_model(model, seq_tile)

            goal_shape = pred_profs_all[model_i, tile_i, :, tile_i_skip : pred_prof.shape[-1] + tile_i_skip].shape
            assert goal_shape == pred_prof.shape, (goal_shape, pred_prof.shape)

            # insert predictions into mega-numpy-array
            pred_profs_all[model_i, tile_i, :, tile_i_skip : pred_prof.shape[-1] + tile_i_skip] = pred_prof
            pred_logcounts_all[model_i, tile_i] = pred_logcounts
    
    # then, first merge predictions across models
    pred_profs = pred_profs_all.mean(axis=0)
    pred_logcounts = pred_logcounts_all.mean(axis=0)
    
    # second, combine counts and profile preds into one scaled prediction per tile
    pred_profs_scaled = pred_profs * np.exp(pred_logcounts)[:, None, None]
    
    # finally, average predictions across all tiles
    preds_final = np.nanmean(pred_profs_scaled, axis=0)
    
    # (I checked by making plots and things look right)
    return preds_final






### Convert predictions from numpy arrays to bigwigs

def load_coords(bed_file):
    # loads bed file into list of regions, 1 element in list = 1 row of file
    lines = []
    if bed_file.endswith(".gz"):
        with gzip.open(bed_file) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(bed_file) as f:
            lines = [line.split() for line in f]
    
    coords = [(line[0], int(line[1]), int(line[2]), line[3]) for line in lines]
    return np.array(coords, dtype=object)


def make_track_values_dict(all_values, coords):
    track_values = defaultdict(lambda : [])

    for values, (coord_chrom, start, end, _) in zip(all_values, coords):
        assert values.shape[0] == end - start, (values.shape, start, end, end - start)
            
        for position, value in enumerate(values):
            position_offset = position + start
            track_values[position_offset] = track_values[position_offset] + [value]
    
    # take the mean at each position, so that if there was ovelap, the average value is used
    track_values = { key : sum(vals) / len(vals) for key, vals in track_values.items() }
    return track_values


def write_tracks_to_bigwigs_by_chrom(preds_dir, save_filepath,
                                     chrom_sizes):
    
    
    print("Writing predicted profiles to bigwigs.")
    print("Chrom sizes file: " + chrom_sizes)
    
    for strand_idx, strand in enumerate(["+", "-"]):
        # write separate bigwigs for svalues on the forward vs. reverse strands (in case of overlap)
        if strand == "+":
            bw_filename = save_filepath.replace(".npy", "") + ".pos.bigWig"
        else:
            bw_filename = save_filepath.replace(".npy", "") + ".neg.bigWig"
        
        print("Save path: " + bw_filename)
            
        # bigwigs need to be written in order -- so we have to go chromosome by chromosome,
        # and the chromosomes need to be numerically sorted (i.e. chr9 then chr10)
        chrom_names = load_chrom_names(chrom_sizes, filter_in=None)
        chrom_order = {chrom : i for i, chrom in enumerate(chrom_names)}

        chromosomes = sorted(list({name for name in chrom_names}),
                             key = lambda chrom_str : chrom_order[chrom_str])
            
        bw = pyBigWig.open(bw_filename, 'w')
        # bigwigs need headers before they can be written to
        # the header is just the info you'd find in a chrom.sizes file
        bw.addHeader(load_chrom_sizes(chrom_sizes))
        
        for chrom in chromosomes:
            coords_filepath = get_regions_to_predict_chrom_filepath(chrom)
            
            if not os.path.exists(coords_filepath):
                print("Skipping ", chrom)
                continue
            else:
                print("Preds loaded: " + coords_filepath)
            
            coords_this_chrom = load_coords(coords_filepath)
            
            # error if pickle is False?
            preds_this_chrom = np.load(preds_dir + "preds." + chrom + ".npy", allow_pickle=True)
            
            assert len(preds_this_chrom) == len(coords_this_chrom), (len(preds_this_chrom), len(coords_this_chrom))
            
            values_for_strand = [preds_this_chrom[i][strand_idx] for i in range(len(preds_this_chrom))]
            
            # convert arrays of scores for each peak into dict of base position : score
            # this function will average together scores at the same position from different called peaks
            track_values_dict = make_track_values_dict(values_for_strand, coords_this_chrom)
            num_entries = len(track_values_dict)
            
            starts = sorted(list(track_values_dict.keys()))
            ends = [position + 1 for position in starts]
            values_to_write = list(np.array([track_values_dict[key] for key in starts]).astype("float64"))
            
            assert len(values_to_write) == len(starts) and len(values_to_write) == len(ends)
            
            bw.addEntries([chrom for _ in range(num_entries)], 
                           starts, ends = ends, values = values_to_write)
    
        bw.close()
        
        
        
        
        
        
        
### Go

models = load_all_models(timestamps, cell_type)


for chrom in chrom_names:
    regions_to_predict_this_chrom = get_regions_to_predict_chrom_filepath(chrom)
    
    # we can ignore some scaffolds and such
    if not os.path.exists(regions_to_predict_this_chrom):
        print("Skipping ", chrom)
        continue
    
    # if starting from a run that failed part way through, don't re-run finished chroms
    if os.path.exists(out_dir + "preds." + chrom + ".npy"):
        print("Found preds for " + chrom)
        continue
        
    onehot_seqs = load_sequences(genome_path, chrom_sizes, regions_to_predict_this_chrom)
    
    preds_this_chrom = []
    for seq in tqdm(onehot_seqs):
        pred = predict_one_seq(seq, models)
        preds_this_chrom.append(pred)

    # hacky way to save ragged array
    preds_arr = np.empty(len(preds_this_chrom), dtype=object)
    preds_arr[:] = preds_this_chrom
    np.save(out_dir + "preds." + chrom + ".npy", preds_arr)


write_tracks_to_bigwigs_by_chrom(out_dir, out_dir + "preds." + cell_type, chrom_sizes)

print("Done.")


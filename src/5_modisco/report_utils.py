import os
import gzip
import h5py
import tempfile
import numpy as np
import logomaker
import vdom.helpers as vdomh
import viz_sequence
import io
import sklearn.cluster
import scipy.cluster.hierarchy
import base64
import urllib
from collections import defaultdict

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.max_colwidth = 500


# motif database files (from JASPAR?)
DATABASE_PATH = "/oak/stanford/groups/akundaje/soumyak/motifs/motifs.meme.txt"
DATABASE_DIR = "/oak/stanford/groups/akundaje/soumyak/motifs/pfms/"


def load_coords(peak_bed, in_window=2114):
    if peak_bed.endswith(".gz"):
        with gzip.open(peak_bed) as f:
            lines = [line.decode().split() for line in f]
    else:
        with open(peak_bed) as f:
            lines = [line.split() for line in f]

    coords = []
    for line in lines:
        chrom, peak_start, peak_end = line[0], int(line[1]), int(line[2])
        mid = (peak_start + peak_end) // 2
        window_start = mid - in_window // 2
        window_end = mid + in_window // 2
        
        pos_summit = int(line[-2]) if line[-2].isdigit() else None
        neg_summit = int(line[-1]) if line[-1].isdigit() else None
        
        coords.append((chrom, window_start, window_end, pos_summit, neg_summit))
    return coords


def load_modisco_results(tfm_results_path):
    return h5py.File(tfm_results_path, "r")




def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
          *ppm - (np.log(background)*background/np.log(2))[None,:])
    return np.sum(ic,axis=1)


def write_meme_file(ppm, bg, fname):
    f = open(fname, 'w')
    f.write('MEME version 4\n\n')
    f.write('ALPHABET= ACGT\n\n')
    f.write('strands: + -\n\n')
    f.write('Background letter frequencies (from unknown source):\n')
    f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
    f.write('MOTIF 1 TEMP\n\n')
    f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
    for s in ppm:
        f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))
    f.close()


def fetch_tomtom_matches(ppm, cwm, motifs_db, 
    background=[0.25, 0.25, 0.25, 0.25], tomtom_exec_path='tomtom',
    trim_threshold=0.3, trim_min_length=3):

    """Fetches top matches from a motifs database using TomTom.
    Args:
        ppm: position probability matrix- numpy matrix of dimension (N,4)
        background: list with ACGT background probabilities
        tomtom_exec_path: path to TomTom executable
        motifs_db: path to motifs database in meme format
        n: number of top matches to return, ordered by p-value
        temp_dir: directory for storing temp files
        trim_threshold: the ppm is trimmed from left till first position for which
            probability for any base pair >= trim_threshold. Similarly from right.
    Returns:
        list: a list of up to n results returned by tomtom, each entry is a
            dictionary with keys 'Target ID', 'p-value', 'E-value', 'q-value'
    """

    _, fname = tempfile.mkstemp()
    _, tomtom_fname = tempfile.mkstemp()

    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh = np.max(score) * trim_threshold  # Cut off anything less than 30% of max score
    pass_inds = np.where(score >= trim_thresh)[0]
    trimmed = ppm[np.min(pass_inds): np.max(pass_inds) + 1]

    # can be None of no base has prob>t
    if trimmed is None:
        return []

    # trim and prepare meme file
    write_meme_file(trimmed, background, fname)

    # run tomtom
    cmd = '%s -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 %s %s > %s' % (tomtom_exec_path, fname, motifs_db, tomtom_fname)

    os.system(cmd)
    tomtom_results = pd.read_csv(tomtom_fname, sep="\t", usecols=(1, 5))
    os.system('rm ' + tomtom_fname)
    os.system('rm ' + fname)
    return tomtom_results


def run_tomtom(modisco_results, output_prefix, meme_motif_db, top_n_matches=3, 
    tomtom_exec="tomtom", trim_threshold=0.3, trim_min_length=3):

    tomtom_results = {'pattern': [], 'num_seqlets': []}
    for i in range(top_n_matches):
        tomtom_results['match{}'.format(i)] = []
        tomtom_results['qval{}'.format(i)] = []

    for name in ['pos_patterns', 'neg_patterns']:
        if name not in modisco_results.keys():
            continue

        metacluster = modisco_results[name]
        for pattern_i in range(len(metacluster.keys())):
            pattern_name = "pattern_" + str(pattern_i)
            pattern = metacluster[pattern_name]
            ppm = np.array(pattern['sequence'][:])
            cwm = np.array(pattern["contrib_scores"][:])

            num_seqlets = pattern['seqlets']['n_seqlets'][:][0]
            tag = '{}.{}'.format(name, pattern_name)

            r = fetch_tomtom_matches(ppm, cwm, motifs_db=meme_motif_db,
                tomtom_exec_path=tomtom_exec, trim_threshold=trim_threshold,
                trim_min_length=trim_min_length)

            tomtom_results['pattern'].append(tag)
            tomtom_results['num_seqlets'].append(num_seqlets)

            for i, (target, qval) in r.iloc[:top_n_matches].iterrows():
                tomtom_results['match{}'.format(i)].append(target)
                tomtom_results['qval{}'.format(i)].append(qval)

            for j in range(i+1, top_n_matches):
                tomtom_results['match{}'.format(i)].append(None)
                tomtom_results['qval{}'.format(i)].append(None)

    return pd.DataFrame(tomtom_results)


def path_to_image_html(path):
    if os.path.exists(path):
        return '<img src="'+ path + '" width="240" >'
    return ""

def _plot_weights(array, path, figsize=(10,3), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 

    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'

    crp_logo = logomaker.Logo(df, ax=ax, font_name='Arial Rounded')
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

    plt.savefig(path)
    plt.close()

def make_logo(match, logo_dir, meme_motif_db):
    if match == 'NA':
        return

    background = np.array([0.25, 0.25, 0.25, 0.25])
    ppm = np.loadtxt("{}/{}.pfm".format(meme_motif_db, match), delimiter='\t')
    ppm = np.transpose(ppm)
    ic = compute_per_position_ic(ppm, background, 0.001)

    _plot_weights(ppm*ic[:, None], path='{}/{}.png'.format(logo_dir, match))


def create_modisco_logos(modisco_results, modisco_logo_dir, trim_threshold):
    names = []

    for name in ["pos_patterns", "neg_patterns"]:
        if name not in modisco_results.keys():
            continue

        metacluster = modisco_results[name]
        for pattern_i in range(len(metacluster.keys())):
            pattern_name = "pattern_" + str(pattern_i)
            pattern = metacluster[pattern_name]
            full_name = '{}.{}'.format(name, pattern_name)
            names.append(full_name)

            cwm_fwd = np.array(pattern['contrib_scores'][:])
            cwm_rev = cwm_fwd[::-1, ::-1]

            score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
            score_rev = np.sum(np.abs(cwm_rev), axis=1)

            trim_thresh_fwd = np.max(score_fwd) * trim_threshold
            trim_thresh_rev = np.max(score_rev) * trim_threshold

            pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
            pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

            start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
            start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1)

            trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

            _plot_weights(trimmed_cwm_fwd, path='{}/{}.cwm.fwd.png'.format(modisco_logo_dir, full_name))
            _plot_weights(trimmed_cwm_rev, path='{}/{}.cwm.rev.png'.format(modisco_logo_dir, full_name))

    return names

def report_motifs(modisco_h5py, proj_dir, output_dir,
                  meme_motif_db = DATABASE_PATH,
                  meme_motif_dir = DATABASE_DIR,
    suffix='../../', top_n_matches=3, trim_threshold=0.3, trim_min_length=3):

    modisco_logo_dir = output_dir + '/trimmed_logos/'
    os.makedirs(modisco_logo_dir, exist_ok=True)

    names = create_modisco_logos(modisco_h5py, modisco_logo_dir, trim_threshold)

    tomtom_df = run_tomtom(modisco_h5py, output_dir, meme_motif_db, 
        top_n_matches=top_n_matches, tomtom_exec="tomtom", 
        trim_threshold=trim_threshold, trim_min_length=trim_min_length)

    logo_dir_relative_path = suffix + modisco_logo_dir.replace(proj_dir, "")
    
    tomtom_df['modisco_cwm_fwd'] = ['{}{}.cwm.fwd.png'.format(logo_dir_relative_path, name) for name in names]
    tomtom_df['modisco_cwm_rev'] = ['{}{}.cwm.rev.png'.format(logo_dir_relative_path, name) for name in names]

    reordered_columns = ['pattern', 'num_seqlets', 'modisco_cwm_fwd', 'modisco_cwm_rev']
    for i in range(top_n_matches):
        name = "match{}".format(i)
        logos = []

        for index, row in tomtom_df.iterrows():
            if name in tomtom_df.columns:
                if pd.isnull(row[name]):
                    logos.append("NA")
                else:
                    make_logo(row[name], modisco_logo_dir, meme_motif_dir)
                    logos.append("{}{}.png".format(logo_dir_relative_path, row[name]))
            else:
                break

        tomtom_df["{}_logo".format(name)] = logos
        reordered_columns.extend([name, 'qval{}'.format(i), "{}_logo".format(name)])

    tomtom_df = tomtom_df[reordered_columns]

    return tomtom_df.to_html(escape=False,
                             formatters=dict(modisco_cwm_fwd=path_to_image_html,
                                             modisco_cwm_rev=path_to_image_html,
                                             match0_logo=path_to_image_html,
                                             match1_logo=path_to_image_html,
                                             match2_logo=path_to_image_html), 
                             index=False)




def extract_profiles_and_coords(
    seqlets, one_hot_seqs, hyp_scores, true_profs, pred_profs, pred_coords,
    in_window, out_window, input_center_cut_size, profile_center_cut_size):
    """
    From the seqlets object of a TF-MoDISco pattern's seqlets and alignments,
    extracts the predicted and observed profiles of the model, as well as the
    set of coordinates for the seqlets.
    Arguments:
        `seqlets`: a TF-MoDISco pattern's seqlets object array (N-array)
        `true_profs`: an N x T x O x 2 array of true profile counts
        `pred_profs`: an N x T x O x 2 array of predicted profile probabilities
        `pred_coords`: an N x 3 object array of coordinates for the input sequence
            underlying the predictions
        `in_window`: length of original input sequences, I
        `out_window`: length of profile predictions, O
        `input_center_cut_size`: centered cut size of SHAP scores used
        `profile_center_cut_size`: size to cut profiles to when returning them, P

    Returns an N x (T or 1) x P x 2 array of true profile counts, an
    N x (T or 1) x P x 2 array of predicted profile probabilities, and an N x 3 list
    of seqlet coordinates, where P is the profile cut size. Returned profiles are
    centered at the same center as the seqlets.
    Note that it is important that the seqlet indices match exactly with the indices
    out of the N. This should be the exact sequences in the original SHAP scores.
    """
    
    def seqlet_coord_to_input_coord(seqlet_coord):
        return seqlet_coord + ((in_window - input_center_cut_size) // 2)
        
    def convert_seqlet_coords_to_genomic_coords(coord_indexes, seqlet_starts, seqlet_ends, pred_coords):
        input_starts = seqlet_coord_to_input_coord(seqlet_starts)
        input_ends = seqlet_coord_to_input_coord(seqlet_ends)
        
        genomic_coords = []
        for coord_index, input_start, input_end in zip(coord_indexes, input_starts, input_ends):
            chrom, peak_start, _, _, _ = pred_coords[coord_index]
            genomic_coords.append([chrom, peak_start + input_start, peak_start + input_end])
            
        return genomic_coords
    
    def seqlet_coord_to_profile_coord(seqlet_coord):
        return seqlet_coord + ((in_window - input_center_cut_size) // 2) - ((in_window - out_window) // 2)
        
    coord_indexes = seqlets["example_idx"][:]
    seqlet_starts = seqlets["start"][:]
    seqlet_ends = seqlets["end"][:]
    seqlet_rcs = seqlets["is_revcomp"][:]

    # Get the coordinates of the seqlet based on the input coordinates
    seqlet_coords = convert_seqlet_coords_to_genomic_coords(coord_indexes, seqlet_starts,
                                                            seqlet_ends, pred_coords)
    
    # Get indices of profile above seqlet
    seqlet_centers = (seqlet_starts + seqlet_ends) // 2
    prof_centers = seqlet_coord_to_profile_coord(seqlet_centers)
    prof_starts = prof_centers - (profile_center_cut_size // 2)
    prof_ends = prof_starts + profile_center_cut_size
    
    # For each seqlet, fetch the true/predicted profiles
    true_seqlet_profs = []
    pred_seqlet_profs = []
    for coord_index, prof_start, prof_end, rc in zip(coord_indexes, prof_starts, prof_ends, seqlet_rcs):
        if prof_start < 0 or prof_end > out_window:
            # don't use profile for this example since indexes go past the profile end  - Kelly
            true_prof = np.full((profile_center_cut_size, true_profs.shape[-2]), np.nan)
            pred_prof = np.full((profile_center_cut_size, true_profs.shape[-2]), np.nan)
        else:
            true_prof = true_profs[coord_index, :, prof_start:prof_end].T
            pred_prof = pred_profs[coord_index, :, prof_start:prof_end].T
        
        if rc:
            true_prof = true_prof[::-1, ::-1]
            pred_prof = pred_prof[::-1, ::-1]
        
        true_seqlet_profs.append(true_prof)
        pred_seqlet_profs.append(pred_prof)  
    
    true_seqlet_profs = np.stack(true_seqlet_profs)
    pred_seqlet_profs = np.stack(pred_seqlet_profs)
    
    return true_seqlet_profs, pred_seqlet_profs, seqlet_coords, seqlet_rcs


def plot_profiles(seqlet_true_profs, seqlet_pred_profs, kmeans_clusters=5, save_path=None):
    """
    Plots the given profiles with a heatmap.
    Arguments:
        `seqlet_true_profs`: an N x O x 2 NumPy array of true profiles, either as raw
            counts or probabilities (they will be normalized)
        `seqlet_pred_profs`: an N x O x 2 NumPy array of predicted profiles, either as
            raw counts or probabilities (they will be normalized)
        `kmeans_cluster`: when displaying profile heatmaps, there will be this
            many clusters
        `save_path`: if provided, save the profile matrices here
    Returns the figure.
    """
    assert len(seqlet_true_profs.shape) == 3
    assert seqlet_true_profs.shape == seqlet_pred_profs.shape

    seqlet_true_profs = seqlet_true_profs[~np.isnan(seqlet_true_profs).any(axis=1).any(axis=1)]
    seqlet_pred_profs = seqlet_pred_profs[~np.isnan(seqlet_pred_profs).any(axis=1).any(axis=1)]

    num_profs, width, _ = seqlet_true_profs.shape

    # First, normalize the profiles along the output profile dimension
    def normalize(arr, axis=0):
        arr_sum = np.sum(arr, axis=axis, keepdims=True)
        arr_sum[arr_sum == 0] = 1  # If 0, keep 0 as the quotient instead of dividing by 0
        return arr / arr_sum
    true_profs_norm = normalize(seqlet_true_profs, axis=1)
    pred_profs_norm = normalize(seqlet_pred_profs, axis=1)

    # Compute the mean profiles across all examples
    true_profs_mean = np.mean(true_profs_norm, axis=0)
    pred_profs_mean = np.mean(pred_profs_norm, axis=0)

    # Perform k-means clustering on the predicted profiles, with the strands pooled
    kmeans_clusters = max(5, num_profs // 50)  # Set number of clusters based on number of profiles, with minimum
    kmeans = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)
    cluster_assignments = kmeans.fit_predict(
        np.reshape(pred_profs_norm, (pred_profs_norm.shape[0], -1))
    )

    # Perform hierarchical clustering on the cluster centers to determine optimal ordering
    kmeans_centers = kmeans.cluster_centers_
    cluster_order = scipy.cluster.hierarchy.leaves_list(
        scipy.cluster.hierarchy.optimal_leaf_ordering(
            scipy.cluster.hierarchy.linkage(kmeans_centers, method="centroid"), kmeans_centers
        )
    )

    # Order the profiles so that the cluster assignments follow the optimal ordering
    cluster_inds = []
    for cluster_id in cluster_order:
        cluster_inds.append(np.where(cluster_assignments == cluster_id)[0])
    cluster_inds = np.concatenate(cluster_inds)

    # Compute a matrix of profiles, normalized to the maximum height, ordered by clusters
    def make_profile_matrix(flat_profs, order_inds):
        matrix = flat_profs[order_inds]
        maxes = np.max(matrix, axis=1, keepdims=True)
        maxes[maxes == 0] = 1  # If 0, keep 0 as the quotient instead of dividing by 0
        return matrix / maxes
    true_matrix = make_profile_matrix(true_profs_norm, cluster_inds)
    pred_matrix = make_profile_matrix(pred_profs_norm, cluster_inds)
    
    if save_path:
        np.savez_compressed(
            true_profs_mean=true_profs_mean, pred_profs_mean=pred_profs_mean,
            true_matrix=true_matrix, pred_matrix=pred_matrix
        )

    # Create a figure with the right dimensions
    mean_height = 4
    heatmap_height = min(num_profs * 0.004, 8)
    fig_height = mean_height + (2 * heatmap_height)
    fig, ax = plt.subplots(
        3, 2, figsize=(16, fig_height), sharex=True,
        gridspec_kw={
            "width_ratios": [1, 1],
            "height_ratios": [mean_height / fig_height, heatmap_height / fig_height, heatmap_height / fig_height]
        }
    )

    # Plot the average predictions
    ax[0, 0].plot(true_profs_mean[:, 0], color="darkslateblue")
    ax[0, 0].plot(-true_profs_mean[:, 1], color="darkorange")
    ax[0, 1].plot(pred_profs_mean[:, 0], color="darkslateblue")
    ax[0, 1].plot(-pred_profs_mean[:, 1], color="darkorange")

    # Set axes on average predictions
    max_mean_val = max(np.max(true_profs_mean), np.max(pred_profs_mean))
    mean_ylim = max_mean_val * 1.05  # Make 5% higher
    ax[0, 0].set_title("True profiles")
    ax[0, 0].set_ylabel("Average probability")
    ax[0, 1].set_title("Predicted profiles")
    for j in (0, 1):
        ax[0, j].set_ylim(-mean_ylim, mean_ylim)
        ax[0, j].label_outer()

    # Plot the heatmaps
    ax[1, 0].imshow(true_matrix[:, :, 0], interpolation="nearest", aspect="auto", cmap="Blues")
    ax[1, 1].imshow(pred_matrix[:, :, 0], interpolation="nearest", aspect="auto", cmap="Blues")
    ax[2, 0].imshow(true_matrix[:, :, 1], interpolation="nearest", aspect="auto", cmap="Oranges")
    ax[2, 1].imshow(pred_matrix[:, :, 1], interpolation="nearest", aspect="auto", cmap="Oranges")

    # Set axes on heatmaps
    for i in (1, 2):
        for j in (0, 1):
            ax[i, j].set_yticks([])
            ax[i, j].set_yticklabels([])
            ax[i, j].label_outer()
    width = true_matrix.shape[1]
    delta = 100
    num_deltas = (width // 2) // delta
    labels = list(range(max(-width // 2, -num_deltas * delta), min(width // 2, num_deltas * delta) + 1, delta))
    tick_locs = [label + max(width // 2, num_deltas * delta) for label in labels]
    for j in (0, 1):
        ax[2, j].set_xticks(tick_locs)
        ax[2, j].set_xticklabels(labels)
        ax[2, j].set_xlabel("Distance from seqlet center (bp)")

    fig.tight_layout()
    plt.show()
    
    
    # Create a figure with the right dimensions
    fig2, ax = plt.subplots(
        1, 2, figsize=(16, mean_height), sharex=True,
        gridspec_kw={"width_ratios": [1, 1]}
    )

    # Plot the average predictions
    mid = true_profs_mean.shape[0] // 2
    zoom_width = 60
    start = mid - zoom_width // 2
    end = mid + zoom_width // 2
    ax[0].plot(true_profs_mean[start:end, 0], color="darkslateblue")
    ax[0].plot(-true_profs_mean[start:end, 1], color="darkorange")
    ax[1].plot(pred_profs_mean[start:end, 0], color="darkslateblue")
    ax[1].plot(-pred_profs_mean[start:end, 1], color="darkorange")

    # Set axes on average predictions
    max_mean_val = max(np.max(true_profs_mean[start:end]), np.max(pred_profs_mean[start:end]))
    mean_ylim = max_mean_val * 1.05  # Make 5% higher
    ax[0].set_title("True profiles")
    ax[0].set_ylabel("Average probability")
    ax[1].set_title("Predicted profiles")
    
    delta = 10
    num_deltas = (zoom_width // 2) // delta
    labels = list(range(max(-zoom_width // 2, -num_deltas * delta), min(zoom_width // 2, num_deltas * delta) + 1, delta))
    tick_locs = [label + max(zoom_width // 2, num_deltas * delta) for label in labels]
    
    for j in (0, 1):
        ax[j].set_ylim(-mean_ylim, mean_ylim)
        ax[j].label_outer()
        ax[j].set_xticks(tick_locs)
        ax[j].set_xticklabels(labels)
        ax[j].set_xlabel("Distance from seqlet center (bp)")

    fig2.tight_layout(w_pad=4, rect=(0.1, 0, 0.95, 1))
    plt.show()
    
    return fig


def get_summit_distances(seqlet_coords, seqlet_rcs, peak_coords):
    peak_chroms, _, _, peak_pos_summits, peak_neg_summits = zip(*peak_coords)
    
    # determine if peak is bidirectional, uni on + strand, or uni on - strand
    peak_strands = []
    for i in range(len(peak_coords)):
        pos_summit = peak_pos_summits[i]
        neg_summit = peak_neg_summits[i]
        assert pos_summit is not None or neg_summit is not None, (pos_summit, neg_summit)
        if pos_summit is None:
            peak_strands.append("-")
        elif neg_summit is None:
            peak_strands.append("+")
        else:
            peak_strands.append("Both")
            
    peak_table = pd.DataFrame({"chrom" : peak_chroms,
                               "summit_pos" : peak_pos_summits,
                               "summit_neg" : peak_neg_summits,
                               "summit_strands" : peak_strands})

    # associate each seqlet with its nearest peak, get distance
    
    dists_to_nearest_peak = []
    strands_of_nearest_peak = []
    for chrom, start, end in seqlet_coords:
        midpoint = (start + end) // 2
        peaks_on_chrom = peak_table[peak_table["chrom"] == chrom]

        # make parallel arrays of peak summits (all types) and peak type (uni+, uni-, bidirectional)
        
        all_peak_summits = np.concatenate([
            np.array(peaks_on_chrom["summit_pos"].values),
            np.array(peaks_on_chrom["summit_neg"].values),
            np.array((peaks_on_chrom["summit_pos"] + peaks_on_chrom["summit_neg"]).values) // 2 
        ])
        
        all_peak_strands = np.concatenate([
            np.array(["+"] * len(peaks_on_chrom)),
            np.array(["-"] * len(peaks_on_chrom)),
            np.array(["Both"] * len(peaks_on_chrom)) 
        ])
        
        # remove where there are nans
        # (if peaks is not bidirectional, one strand's summit is nan)
        
        all_peak_strands = all_peak_strands[~ np.isnan(all_peak_summits)]
        all_peak_summits = all_peak_summits[~ np.isnan(all_peak_summits)]
        assert len(all_peak_summits.shape) == 1, all_peak_summits.shape
        assert all_peak_summits.shape == all_peak_strands.shape, (all_peak_summits.shape, all_peak_strands.shape)
        
        # then get min distance from seqlets to any summit, and keep track of peak type
        
        dist_arr = midpoint - all_peak_summits
        
        min_dist = dist_arr[np.argmin(np.abs(dist_arr))]
        min_dist_strand = all_peak_strands[np.argmin(np.abs(dist_arr))]

        dists_to_nearest_peak.append(min_dist)
        strands_of_nearest_peak.append(min_dist_strand)
    
    # finally, separate out peak-to-seqlet distances by peak type, seqlet orientation
    
    dists = defaultdict(lambda : dict())

    for strand in ["+", "-", "Both"]:
        peaks_on_strand = peak_table[peak_table["summit_strands"] == strand]
        
        fwd_dists = []
        rc_dists = []
        for rc, dist, peak_strand in zip(seqlet_rcs, dists_to_nearest_peak, strands_of_nearest_peak):
            if peak_strand == strand:
                if rc:
                    rc_dists.append(dist)
                else:
                    fwd_dists.append(dist)

        dists[strand]["fwd"] = np.array(fwd_dists)
        dists[strand]["rc"] = np.array(rc_dists)
        
    return dists


def plot_summit_dists(summit_dists, plot_width = 200):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, peak_type in zip(axes, summit_dists.keys()):
        peak_type_to_name = {"+": "Pos. Strand", "-": "Neg. Strand", "Both" : "Bidirectional"}
        peak_type_name = peak_type_to_name[peak_type]
        
        dists_fwd = summit_dists[peak_type]["fwd"]
        dists_rc = summit_dists[peak_type]["rc"]
            
        dists_fwd = [dist for dist in dists_fwd if dist < plot_width and dist > - plot_width]
        dists_rc = [dist for dist in dists_rc if dist < plot_width and dist > - plot_width]
        
        bins = np.arange(-plot_width, plot_width, 10)
        ax.hist(dists_fwd, bins=bins, color="darkslateblue", alpha = 0.5, label = "Fwd")
        ax.hist(dists_rc, bins=bins, color="darkorange", alpha = 0.5, label = "RC")
        ax.set_title("Distribution of Seqlets Around\n" + peak_type_name + " Peak Summits")
        ax.set_xlabel("Distance from Nearest Summit, bp")
        ax.legend(frameon=False)
    
    fig.tight_layout()
    plt.show()


def pfm_info_content(track, pseudocount=0.001):
    """
    Given an L x 4 track, computes information content for each base and
    returns it as an L-array.
    """
    background = np.array([0.25, 0.25, 0.25, 0.25])
    num_bases = track.shape[1]
    # Normalize track to probabilities along base axis
    track_norm = (track + pseudocount) / (np.sum(track, axis=1, keepdims=True) + (num_bases * pseudocount))
    ic = track_norm * np.log2(track_norm / np.expand_dims(background, axis=0))
    return np.sum(ic, axis=1)


def pfm_to_pwm(pfm):
    ic = pfm_info_content(pfm)
    return pfm * np.expand_dims(ic, axis=1)


def trim_motif_by_ic(pfm, motif, min_ic=0.2, pad=0):
    """
    Given the PFM and motif (both L x 4 arrays) (the motif could be the
    PFM itself), trims `motif` by cutting off flanks of low information
    content in `pfm`. `min_ic` is the minimum required information
    content. If specified this trimmed motif will be extended on either
    side by `pad` bases.
    If no base passes the `min_ic` threshold, then no trimming is done.
    """
    # Trim motif based on information content
    ic = pfm_info_content(pfm)
    pass_inds = np.where(ic >= min_ic)[0]  # Cut off flanks with less than min_ic IC
    
    if not pass_inds.size:
        return motif

    # Expand trimming to +/- pad bp on either side
    start, end = max(0, np.min(pass_inds) - pad), min(len(pfm), np.max(pass_inds) + pad + 1)
    return motif[start:end]


def figure_to_vdom_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    img_src_str = 'data:image/png;base64,' + urllib.parse.quote(string)
    return vdomh.div(vdomh.img(src=img_src_str), style={"display":"inline-block"})


def plot_all_metaclusters(tfm_obj, one_hot_seqs, hyp_scores, true_profs, pred_profs, coords,
                in_window, out_window, score_center_size,
                profile_display_center_size):

    for pattern_type in ["pos_patterns", "neg_patterns"]:
        patterns = tfm_obj[pattern_type]
        num_patterns = len(patterns)
        
        for pattern_i in range(num_patterns):
            pattern_name = "pattern_" + str(pattern_i)
            pattern = patterns[pattern_name]
            seqlets = pattern["seqlets"]
            num_seqlets = seqlets["n_seqlets"][:]

            display(vdomh.h4("Pattern %d/%d" % (pattern_i, num_patterns)))
            display(vdomh.p("%d seqlets" % num_seqlets))

            pfm = pattern["sequence"][:]
            hcwm = pattern["hypothetical_contribs"][:]
            cwm = pattern["contrib_scores"][:]

            pfm_fig = viz_sequence.plot_weights(pfm, subticks_frequency=10, return_fig=True)
            hcwm_fig = viz_sequence.plot_weights(hcwm, subticks_frequency=10, return_fig=True)
            cwm_fig = viz_sequence.plot_weights(cwm, subticks_frequency=10, return_fig=True)
            pfm_fig.tight_layout()
            hcwm_fig.tight_layout()
            cwm_fig.tight_layout()

            motif_table = vdomh.table(
                vdomh.tr(
                    vdomh.td("Sequence (PFM)"),
                    vdomh.td(figure_to_vdom_image(pfm_fig))
                ),
                vdomh.tr(
                    vdomh.td("Hypothetical contributions (hCWM)"),
                    vdomh.td(figure_to_vdom_image(hcwm_fig))
                ),
                vdomh.tr(
                    vdomh.td("Actual contributions (CWM)"),
                    vdomh.td(figure_to_vdom_image(cwm_fig))
                )
            )
            display(motif_table)
            plt.close("all")  # Remove all standing figures
            
            seqlet_true_profs, seqlet_pred_profs, seqlet_coords, rcs = extract_profiles_and_coords(
                seqlets, one_hot_seqs, hyp_scores, true_profs, pred_profs, coords,
                in_window, out_window, score_center_size,
                profile_display_center_size)

            plot_profiles(seqlet_true_profs, seqlet_pred_profs)

            summit_dists = get_summit_distances(seqlet_coords, rcs, coords)
            plot_summit_dists(summit_dists)

    

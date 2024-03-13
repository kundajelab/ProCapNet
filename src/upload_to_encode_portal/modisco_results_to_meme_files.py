import sys
import os
from collections import defaultdict
import numpy as np


assert len(sys.argv) == 4, len(sys.argv)
modisco_results_path = sys.argv[1]
meme_files_dest_dir = sys.argv[2]
meme_files_prefix = sys.argv[3]

if not meme_files_dest_dir.endswith("/"):
    meme_files_dest_dir = meme_files_dest_dir + "/"

os.makedirs(meme_files_dest_dir, exist_ok=True)

sys.path.append("../5_modisco")
from report_utils import load_modisco_results

sys.path.append("../figure_notebooks")
from other_motif_utils import compute_per_position_ic


def load_ppms_cwms(modisco_results):
    motifs = defaultdict(lambda : [])
    for pattern_group in ['pos_patterns', 'neg_patterns']:
        if pattern_group not in modisco_results.keys():
            continue

        metacluster = modisco_results[pattern_group]
        for pattern_i in range(len(metacluster.keys())):
            pattern_name = "pattern_" + str(pattern_i)
            pattern = metacluster[pattern_name]
            ppm = np.array(pattern['sequence'][:])
            cwm = np.array(pattern["contrib_scores"][:])
            
            pwm = ppm * compute_per_position_ic(ppm)[:, None]

            motifs[pattern_group].append((ppm, pwm, cwm))
    return motifs


def trim_all_motif_matrices_by_pwm_and_cwm(modisco_motifs, trim_threshold=0.3, pad=2):
    motifs_trimmed = defaultdict(lambda : [])
    for pattern_group, motifs in modisco_motifs.items():
        for motif in motifs:
            ppm, pwm, cwm = motif

            assert ppm.shape == pwm.shape
            assert ppm.shape == cwm.shape
            assert cwm.shape[-1] == 4

            starts = []
            ends = []
            for motif in [pwm, cwm]:
                trim_thresh = np.max(motif) * trim_threshold
                pass_inds = np.where(motif >= trim_thresh)[0]

                _start = max(np.min(pass_inds) - pad, 0)
                _end = min(np.max(pass_inds) + pad + 1, len(motif) + 1)

                starts.append(_start)
                ends.append(_end)

            start = np.min(starts)
            end = np.max(ends)

            motif_trimmed = (motif[start:end] for motif in [ppm, pwm, cwm])
            motifs_trimmed[pattern_group].append(motif_trimmed)
        
    return motifs_trimmed


def write_meme_file(ppm, motif_name, fname):
    background_freqs = [0.25] * 4
    
    f = open(fname, 'w')
    f.write('MEME version 4\n\n')
    f.write('ALPHABET= ACGT\n\n')
    f.write('strands: + -\n\n')
    f.write('Background letter frequencies (from unknown source):\n')
    f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(background_freqs))
    f.write(motif_name + '\n\n')
    f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
    for s in ppm:
        f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))
    f.close()


def write_meme_files_all_motifs(dest_dir, motifs, pos_or_neg_patterns, files_prefix):
    assert pos_or_neg_patterns in ["pos", "neg"], pos_or_neg_patterns
    
    for motif_i, motif,  in enumerate(motifs):
        ppm, pwm, cwm = motif
        
        assert ppm.shape == pwm.shape
        assert ppm.shape == cwm.shape
        assert len(cwm.shape) == 2 and cwm.shape[-1] == 4
        
        fname_root = dest_dir + meme_files_prefix + "."
        fname_root += pos_or_neg_patterns + ".motif_" + str(motif_i+1) + "."
        
        if pos_or_neg_patterns == "pos":
            motif_name = "TF-Modisco Positive Pattern " + str(motif_i+1)
        else:
            motif_name = "TF-Modisco Negative Pattern " + str(motif_i+1)
        
        write_meme_file(ppm, motif_name, fname_root + "ppm.fwd.meme")
        write_meme_file(ppm[::-1, ::-1], motif_name, fname_root + "ppm.rev.meme")
        write_meme_file(pwm, motif_name, fname_root + "pwm.fwd.meme")
        write_meme_file(pwm[::-1, ::-1], motif_name, fname_root + "pwm.rev.meme")
        write_meme_file(cwm, motif_name, fname_root + "cwm.fwd.meme")
        write_meme_file(cwm[::-1, ::-1], motif_name, fname_root + "cwm.rev.meme")
        
        
        
modisco_results = load_modisco_results(modisco_results_path)
modisco_motifs = load_ppms_cwms(modisco_results)
modisco_motifs_trimmed = trim_all_motif_matrices_by_pwm_and_cwm(modisco_motifs)

if "pos_patterns" in modisco_motifs_trimmed.keys():
    write_meme_files_all_motifs(meme_files_dest_dir, modisco_motifs_trimmed["pos_patterns"], "pos", meme_files_prefix)
    
if "neg_patterns" in modisco_motifs_trimmed.keys():
    write_meme_files_all_motifs(meme_files_dest_dir, modisco_motifs_trimmed["neg_patterns"], "neg", meme_files_prefix)

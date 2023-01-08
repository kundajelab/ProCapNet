import h5py
import numpy as np

import sys
sys.path.append("../2_train_models")
sys.path.append("../5_modisco")

from utils import load_json

assert len(sys.argv) == 2, len(sys.argv)  # expecting config file input
config_file = sys.argv[1]

config = load_json(config_file)
results_save_path = config["results_save_path"]


def convert_new_to_old(new_format_filename, old_format_filename):
    '''This function does the opposite of the convert function.
    Given the filepath to a tfmodisco-lite-formatted file (arg 1),
    it saves the same information in the original tfmodisco format
    to file (arg 2).
    
    This function assumes that metacluster0 should be the positive
    patterns (patterns formed from positive importance scores, and 
    that metacluster1 is the negative patterns. It also assumes that
    there is only 1 task (the standard use-case).
    
    This function does not fill in all of the information for the
    original modisco format, in part because some of that information
    is no longer in the new modisco format. But the info converted is
    sufficient to run motif hit calling using the original modisco algorithm.
    '''
    
    print("Converting " + new_format_filename + " to old modisco format...")
    old_f = h5py.File(old_format_filename, "w")
    new_f = h5py.File(new_format_filename, "r")
    
    old_f.create_dataset("task_names", data=["task0"])
    
    old_fmt_grp = old_f.create_group('metacluster_idx_to_submetacluster_results')
    

    patterns_group_name_to_metacluster_name = {"pos_patterns" : 'metacluster_0',
                                              "neg_patterns" : 'metacluster_1'}
    
    for patterns_group_name in ['pos_patterns', 'neg_patterns']:
        
        # if the results include pos/neg patterns...
        
        if patterns_group_name in new_f.keys():
            metacluster_name = patterns_group_name_to_metacluster_name[patterns_group_name]
            metacluster_seqlet_strings = []
            
            # new format
            new_patterns_grp = new_f[patterns_group_name]
            
            # old format
            old_metacluster_grp = old_fmt_grp.create_group(metacluster_name)
            old_patterns_grp = old_metacluster_grp.create_group('seqlets_to_patterns_result')
            
            # these needed to avoid error / silent failure
            old_patterns_grp.attrs["success"] = True
            old_patterns_grp.attrs["total_time_taken"] = 1.0
            
            
            # if there are any patterns for this hit (should always be???)...
            
            if len(new_patterns_grp.keys()) > 0:
                pattern_names = list(new_patterns_grp.keys())
                pattern_names = sorted(pattern_names, key = lambda name : int(name.split("_")[1]))
                
                old_patterns_subgrp = old_patterns_grp.create_group("patterns")
                
                # for each modisco hit...
                
                for pattern in pattern_names:
                    pattern_grp = new_patterns_grp[pattern]
                    
                    # new format
                    sequence = pattern_grp["sequence"]
                    contrib_scores = pattern_grp["contrib_scores"]
                    hypothetical_contribs = pattern_grp["hypothetical_contribs"]
                    
                    # old format
                    old_pattern_grp = old_patterns_subgrp.create_group(pattern)
                    old_pattern_grp.create_dataset("sequence/fwd", data=sequence)
                    old_pattern_grp.create_dataset("task0_contrib_scores/fwd", data=contrib_scores)
                    old_pattern_grp.create_dataset("task0_hypothetical_contribs/fwd", data=hypothetical_contribs)
                    
                    seqlets_grp = pattern_grp['seqlets']
                    
                    # in the old format, seqlets were stored as a list of strings
                    
                    seqlet_strings = []
                    for i in range(len(seqlets_grp['example_idx'])):
                        # new format: separate arrays for each seqlet attribute,
                        # where each array is len(# seqlets)
                        example_idx = str(seqlets_grp['example_idx'][i])
                        start = str(seqlets_grp['start'][i])
                        end = str(seqlets_grp['end'][i])
                        rc = str(seqlets_grp['is_revcomp'][i])
                        seqlet_str = "example:" + example_idx + ",start:" + start + ",end:" + end + ",rc:" + rc
                        seqlet_strings.append(seqlet_str)
                    metacluster_seqlet_strings.extend(seqlet_strings)
                    
                    # old format
                    
                    old_seq_align_grp = old_pattern_grp.create_group("seqlets_and_alnmts")
                    old_seq_align_grp.create_dataset('seqlets', data=seqlet_strings)
                    
                    # dummy data to avoid error: alignments are all 0s
                    # (must be len(# seqlets), and must be ints)
                    old_seq_align_grp.create_dataset('alnmts', data=np.zeros((len(seqlet_strings),)), dtype="i")
                    
                    
                    # repeat the process for each pattern/cluster (above) for each subpattern/subcluster (below)
                    
                    subcluster_names = [k for k in pattern_grp.keys() if k.startswith("subcluster_")]
                    if len(subcluster_names) > 0:
                        old_subpatterns_grp = old_pattern_grp.create_group("subcluster_to_subpattern")
                        
                        for subpattern in subcluster_names:
                            subpattern_grp = pattern_grp[subpattern]
                            sequence = subpattern_grp["sequence"]
                            contrib_scores = subpattern_grp["contrib_scores"]
                            hypothetical_contribs = subpattern_grp["hypothetical_contribs"]
                    
                            old_subpattern_grp = old_subpatterns_grp.create_group(subpattern)
                            old_subpattern_grp.create_dataset("sequence/fwd", data=sequence)
                            old_subpattern_grp.create_dataset("task0_contrib_scores/fwd", data=contrib_scores)
                            old_subpattern_grp.create_dataset("task0_hypothetical_contribs/fwd", data=hypothetical_contribs)
                    
                            seqlets_grp = subpattern_grp['seqlets']
                    
                            seqlet_strings = []
                            for i in range(len(seqlets_grp['example_idx'])):
                                example_idx = str(seqlets_grp['example_idx'][i])
                                start = str(seqlets_grp['start'][i])
                                end = str(seqlets_grp['end'][i])
                                rc = str(seqlets_grp['is_revcomp'][i])
                                seqlet_str = "example:" + example_idx + ",start:" + start + ",end:" + end + ",rc:" + rc
                                seqlet_strings.append(seqlet_str)

                            old_seq_align_grp = old_subpattern_grp.create_group("seqlets_and_alnmts")
                            old_seq_align_grp.create_dataset('seqlets', data=seqlet_strings)
                            old_seq_align_grp.create_dataset('alnmts', data=np.zeros((len(seqlet_strings),)), dtype="i")
                    
                        old_subpatterns_grp.create_dataset("subcluster_names", data=subcluster_names)
                    
                old_patterns_subgrp.create_dataset("all_pattern_names", data=pattern_names)
                
            # required to avoid error: a collection of seqlets for the entire metacluster
            
            old_metacluster_grp.create_dataset("seqlets", data=metacluster_seqlet_strings)

    old_f.close()
    new_f.close()
    
    print("Saved at " + new_format_filename + ".")
                
        
if __name__ == "__main__":
    convert_new_to_old(results_save_path,
                       results_save_path.replace("modisco_results.h", "old_fmt_modisco_results.h"))
    
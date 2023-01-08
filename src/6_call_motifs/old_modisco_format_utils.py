import modisco
from modisco.tfmodisco_workflow import workflow, seqlets_to_patterns
from collections import OrderedDict
import h5py


class SubMetaclusterResults2(workflow.SubMetaclusterResults):

    def __init__(self, metacluster_size, seqlets, seqlets_to_patterns_result):
        self.metacluster_size = metacluster_size
        self.seqlets = seqlets
        self.seqlets_to_patterns_result = seqlets_to_patterns_result

    @classmethod
    def from_hdf5(cls, grp, track_set):
        seqlet_coords = modisco.util.load_seqlet_coords(dset_name="seqlets", grp=grp)
        seqlets = track_set.create_seqlets(coords=seqlet_coords)
        
        seqlets_to_patterns_result =\
            seqlets_to_patterns.SeqletsToPatternsResults.from_hdf5(
                grp=grp["seqlets_to_patterns_result"],
                track_set=track_set) 
        
        return cls(metacluster_size=len(grp["seqlets_to_patterns_result"]["patterns"]),
                   seqlets=seqlets,
                   seqlets_to_patterns_result=seqlets_to_patterns_result) 
    

class TfModiscoResults2(workflow.TfModiscoResults):

    def __init__(self,
                 task_names,
                 metacluster_idx_to_submetacluster_results,
                 **kwargs):
        self.task_names = task_names
        self.metacluster_idx_to_submetacluster_results =\
            metacluster_idx_to_submetacluster_results

        self.__dict__.update(**kwargs)

    @classmethod
    def from_hdf5(cls, grp, track_set):
        task_names = modisco.util.load_string_list(dset_name="task_names",
                                           grp=grp)
        metacluster_idx_to_submetacluster_results = OrderedDict()
        metacluster_idx_to_submetacluster_results_group =\
            grp["metacluster_idx_to_submetacluster_results"]
        
        for metacluster_idx in metacluster_idx_to_submetacluster_results_group:
            metacluster_idx_to_submetacluster_results[metacluster_idx] =\
             SubMetaclusterResults2.from_hdf5(
                grp=metacluster_idx_to_submetacluster_results_group[
                     metacluster_idx],
                track_set=track_set)

        return cls(task_names=task_names,
                   metacluster_idx_to_submetacluster_results=
                    metacluster_idx_to_submetacluster_results)
    
    
def import_tfmodisco_results(tfm_results_path, hyp_scores, one_hot_seqs):
    """
    Imports the TF-MoDISco results object.
    Arguments:
        `tfm_results_path`: path to HDF5 containing TF-MoDISco results
        `hyp_scores`: hypothetical importance scores used for this run
        `one_hot_seqs`: input sequences used for this run
    """ 
    # Everything should already be cut to `input_center_cut_size`
    act_scores = hyp_scores * one_hot_seqs
    
    track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
        task_names=["task0"],
        contrib_scores={"task0": act_scores},
        hypothetical_contribs={"task0": hyp_scores},
        one_hot=one_hot_seqs
    )
    
    with h5py.File(tfm_results_path,"r") as f:
        return TfModiscoResults2.from_hdf5(f, track_set=track_set)
    
    
def get_patterns_from_modisco_results(tfm_results_obj, metacluster = "metacluster_0"):
    return tfm_results_obj.metacluster_idx_to_submetacluster_results[metacluster].seqlets_to_patterns_result.patterns

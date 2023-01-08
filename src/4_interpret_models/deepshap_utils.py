import sys
import numpy as np
import torch
from tqdm import trange
from captum.attr import DeepLiftShap

sys.path.append("../2_train_models")
from data_loading import extract_peaks
from dinuc_shuffle import dinuc_shuffle
from write_bigwigs import write_scores_to_bigwigs
from utils import ensure_parent_dir_exists


class ProfileModelWrapper(torch.nn.Module):
    # this wrapper assumes:
    # 1) the model's profile head outputs pre-softmax logits
    # 2) the profile output has the last axis as the profile-length dimension
    # 3) the softmax should be applied over both strands at the same time
    #      (a ala Jacob's bpnetlite implementation of BPNet)
    # 4) the profile head is the first of two model outputs
    
    def __init__(self, model):
        super(ProfileModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, X):
        logits, _ = self.model(X)
        logits = logits.reshape(logits.shape[0], -1)
        mean_norm_logits = logits - torch.mean(logits, axis = -1, keepdims = True)
        softmax_probs = torch.nn.Softmax(dim=-1)(mean_norm_logits.detach())
        return (mean_norm_logits * softmax_probs).sum(axis=-1)
    
    
class CountsModelWrapper(torch.nn.Module):
    # this wrapper assumes the counts head is the second of two model outputs
    
    def __init__(self, model):
        super(CountsModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, X):
        _, logcounts = self.model(X)
        return logcounts
    
    
class StrandedProfileModelWrapper(torch.nn.Module):
    # this wrapper assumes:
    # 1) the model's profile head outputs pre-softmax logits
    # 2) the profile output has the last axis as the profile-length dimension
    # 3) the softmax should be applied individual to each strand
    # 4) the profile head is the first of two model outputs
    
    def __init__(self, model):
        super(StrandedProfileModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, X):
        logits, _ = self.model(X)
        # take mean over profile axis, strands are still separate
        mean_norm_logits = logits - torch.mean(logits, axis = -1, keepdims = True)
        # take mean over profile axis, strands are still separate
        softmax_probs = torch.nn.Softmax(dim=-1)(mean_norm_logits.detach())
        # sum over all bases and both strands
        return (mean_norm_logits * softmax_probs).sum(axis=(-1,-2))


class StrandedCountsModelWrapper(torch.nn.Module):
    # this wrapper assumes the counts head is the second of two model outputs
    
    def __init__(self, model):
        super(StrandedCountsModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, X):
        _, logcounts = self.model(X)
        return logcounts.sum(axis=-1)

    
    
    
def get_attributions(sequences, prof_explainer, count_explainer, num_shufs = 25):
    assert len(sequences.shape) == 3 and sequences.shape[1] == 4, sequences.shape
    prof_attrs = []
    count_attrs = []

    with torch.no_grad():
        for i in trange(len(sequences)):
            # use a batch of 1 so that reference is generated for each seq 
            seq = torch.tensor(sequences[i : i + 1]).float()

            # create a reference of dinucleotide shuffled sequences
            ref_seqs = dinuc_shuffle(seq[0], num_shufs).float().cuda()

            seq = seq.cuda()
            # calculate attributions according to profile task
            prof_attrs_batch = prof_explainer.attribute(seq, ref_seqs)
            prof_attrs.append(prof_attrs_batch.cpu().numpy())

            # calculate attributions according to counts task
            count_attrs_batch = count_explainer.attribute(seq, ref_seqs)
            count_attrs.append(count_attrs_batch.cpu().numpy())

    prof_attrs = np.concatenate(prof_attrs)
    count_attrs = np.concatenate(count_attrs)
    return prof_attrs, count_attrs



def save_deepshap_results(onehot_seqs, scores, peak_path,
                          scores_path, onehot_scores_path,
                          chrom_sizes):
    assert len(onehot_seqs.shape) == 3 and onehot_seqs.shape[1] == 4, onehot_seqs.shape
    assert len(scores.shape) == 3 and scores.shape[1] == 4, scores.shape
    
    ensure_parent_dir_exists(scores_path)
    ensure_parent_dir_exists(onehot_scores_path)
    
    # save profile attributions
    scores_onehot = scores * onehot_seqs
    np.save(scores_path, scores)
    np.save(onehot_scores_path, scores_onehot)

    # write scores to bigwigs -- flatten the one-hot encoding of scores
    write_scores_to_bigwigs(np.sum(scores_onehot, axis = 1),
                            peak_path, scores_path, chrom_sizes)
    
    
    
    
def run_deepshap(genome_path, chrom_sizes, plus_bw_path, minus_bw_path,
                 peak_path, model_path, prof_scores_path, prof_onehot_scores_path,
                 count_scores_path, count_onehot_scores_path,
                 in_window=2114, out_window=1000, stranded=False, save=True):
    
    print("Running deepSHAP.\n")
    print("genome_path:", genome_path)
    print("chrom_sizes:", chrom_sizes)
    print("plus_bw_path:", plus_bw_path)
    print("minus_bw_path:", minus_bw_path)
    print("peak_path:", peak_path)
    print("model_path:", model_path)
    print("Stranded model:", stranded)
    
    print("prof_scores_path:", prof_scores_path)
    print("prof_onehot_scores_path:", prof_onehot_scores_path)
    print("count_scores_path:", count_scores_path)
    print("count_onehot_scores_path:", count_onehot_scores_path)

    print("in_window:", in_window)
    print("out_window:", out_window, "\n")
    
    
    onehot_seqs, _ = extract_peaks(genome_path, chrom_sizes,
                                   plus_bw_path,
                                   minus_bw_path, peak_path,
                                   in_window=in_window,
                                   out_window=out_window,
                                   max_jitter=0, verbose=True)

    model = torch.load(model_path)
    model.eval()
    model = model.cuda()

    if stranded:
        prof_shap_explainer = DeepLiftShap(StrandedProfileModelWrapper(model))
        count_shap_explainer = DeepLiftShap(StrandedCountsModelWrapper(model))
    else:
        prof_shap_explainer = DeepLiftShap(ProfileModelWrapper(model))
        count_shap_explainer = DeepLiftShap(CountsModelWrapper(model))

    prof_attrs, count_attrs = get_attributions(onehot_seqs,
                                               prof_shap_explainer,
                                               count_shap_explainer)

    if save:
        save_deepshap_results(onehot_seqs, prof_attrs, peak_path,
                              prof_scores_path, prof_onehot_scores_path,
                              chrom_sizes)
    
        save_deepshap_results(onehot_seqs, count_attrs, peak_path,
                              count_scores_path, count_onehot_scores_path,
                              chrom_sizes)
    else:
        return prof_attrs, count_attrs

from data_loading import *
from torch.utils.data import Sampler
import numpy as np
import torch
from math import ceil

class MultiSourceSampler(Sampler):
    
    ''' This class works with the PyTorch Data Loader setup to allow
    the model to be trained on batches of data coming from multiple sources.
    The fraction of each batch coming from each data source can be specified.
    '''
    
    def __init__(self, train_generator, source_totals, source_fracs,
                 batch_size):
        
        source_batch_sizes = [ceil(batch_size * frac) for frac in source_fracs]
        # if the int truncation above doesn't create nums
        # that add up to the desired total ...
        if sum(source_batch_sizes) != batch_size:
            source_batch_sizes[0] += batch_size - sum(source_batch_sizes)
        self.source_batch_sizes = source_batch_sizes
        
        self.source_totals = source_totals
        
        self.range_ends = []
        total_so_far = 0
        for source_total in source_totals:
            total_so_far += source_total
            self.range_ends.append(total_so_far)
            
        self.range_starts = [0] + self.range_ends[:-1]
        
        # going to assume source 0 is the real peaks,
        # which we want to sample (close to) 100% of 
        self.num_batches = source_totals[0] // self.source_batch_sizes[0]
        self.len = self.num_batches * source_totals[0]
        
    def __len__(self):
        return self.len
    
    def __iter__(self):
        peak_indices_permuted = torch.randperm(self.range_ends[0]).tolist()

        for batch_i in range(self.num_batches):
            # first source (actual peaks)
            source_0_batch_size = self.source_batch_sizes[0]
            yield from peak_indices_permuted[batch_i * source_0_batch_size : (batch_i + 1) * source_0_batch_size]
            
            for source_i, source_batch_size in enumerate(self.source_batch_sizes):
                if source_i == 0:
                    continue  # we did the 0th source separately above
                    
                start = self.range_starts[source_i]
                end = self.range_ends[source_i]

                yield from torch.randint(low = start, high = end,
                                         size = (source_batch_size,),
                                         dtype = torch.int64).tolist()

                
def load_data_loader(genome_path, chrom_sizes, plus_bw_path, minus_bw_path,
                     peak_paths, source_fracs, mask_bw_path=None,
                     in_window=2114, out_window=1000, max_jitter=0,
                     batch_size=64, generator_random_seed=0):
    
    sequences_all_sources = []
    profiles_all_sources = []
    
    if mask_bw_path:
        masks_all_sources = []
    
        for peak_path in peak_paths:
            sequences, profiles, masks = extract_peaks(genome_path,
                                                       chrom_sizes,
                                                       plus_bw_path,
                                                       minus_bw_path,
                                                       peak_path,
                                                       mask_bw_path=mask_bw_path,
                                                       in_window=in_window,
                                                       out_window=out_window,
                                                       max_jitter=max_jitter,
                                                       verbose=True)
            sequences_all_sources.append(sequences)
            profiles_all_sources.append(profiles)
            masks_all_sources.append(masks)
    
        train_masks = np.concatenate(masks_all_sources)
    
    else:
        for peak_path in peak_paths:
            sequences, profiles = extract_peaks(genome_path,
                                                chrom_sizes,
                                                plus_bw_path,
                                                minus_bw_path,
                                                peak_path,
                                                in_window=in_window,
                                                out_window=out_window,
                                                max_jitter=max_jitter,
                                                verbose=True)
            sequences_all_sources.append(sequences)
            profiles_all_sources.append(profiles)
    
        train_masks = None
            
    train_sequences = np.concatenate(sequences_all_sources)
    train_signals = np.concatenate(profiles_all_sources)

    gen = DataGenerator(sequences=train_sequences,
                        signals=train_signals,
                        masks=train_masks,
                        in_window=in_window,
                        out_window=out_window,
                        random_state=generator_random_seed)

    multi_sampler = MultiSourceSampler(gen,
                                       [a.shape[0] for a in sequences_all_sources],
                                       source_fracs, batch_size)

    train_data = torch.utils.data.DataLoader(gen,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             sampler=multi_sampler)
    return train_data
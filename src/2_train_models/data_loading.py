### Code here is modified from Jacob's Schreiber's
### implementation of BPNet, called BPNet-lite:
### https://github.com/jmschrei/bpnet-lite/


import numpy as np
import torch
import os, sys
from tqdm import tqdm
from utils import load_chrom_names



def one_hot_encode(sequence, alphabet=['A','C','G','T'], dtype='int8', 
    desc=None, verbose=False, **kwargs):
    """Converts a string or list of characters into a one-hot encoding.

    This function will take in either a string or a list and convert it into a
    one-hot encoding. If the input is a string, each character is assumed to be
    a different symbol, e.g. 'ACGT' is assumed to be a sequence of four 
    characters. If the input is a list, the elements can be any size.

    Although this function will be used here primarily to convert nucleotide
    sequences into one-hot encoding with an alphabet of size 4, in principle
    this function can be used for any types of sequences.

    Parameters
    ----------
    sequence : str or list
        The sequence to convert to a one-hot encoding.

    alphabet : set or tuple or list, optional
        A pre-defined alphabet. If None is passed in, the alphabet will be
        determined from the sequence, but this may be time consuming for
        large sequences. Default is ACGT.

    dtype : str or numpy.dtype, optional
        The data type of the returned encoding. Default is int8.

    desc : str or None, optional
        The title to display in the progress bar.

    verbose : bool or str, optional
        Whether to display a progress bar. If a string is passed in, use as the
        name of the progressbar. Default is False.

    kwargs : arguments
        Arguments to be passed into tqdm. Default is None.

    Returns
    -------
    ohe : numpy.ndarray
        A binary matrix of shape (alphabet_size, sequence_length) where
        alphabet_size is the number of unique elements in the sequence and
        sequence_length is the length of the input sequence.
    """
    
    # these characters will be encoded as all-zeros
    ambiguous_nucs = ["Y", "R", "W", "S", "K", "M", "D", "V", "H", "B", "X", "N"]

    d = verbose is False

    sequence = sequence.upper()
    if isinstance(sequence, str):
        sequence = list(sequence)

    alphabet = alphabet or np.unique(sequence)
    alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

    ohe = np.zeros((len(sequence), len(alphabet)), dtype=dtype)
    for i, char in tqdm(enumerate(sequence), disable=d, desc=desc, **kwargs):
        if char in alphabet:
            idx = alphabet_lookup[char]
            ohe[i, idx] = 1
        else:
            assert char in ambiguous_nucs, char

    return ohe


def read_fasta_fast(filename, chrom_sizes=None, include_chroms=None, verbose=True):
    """Read in a FASTA file and output a dictionary of sequences.

    This function will take in the path to a FASTA-formatted file and output
    a string containing the sequence for each chromosome. Optionally,
    the user can specify a set of chromosomes to include or exclude from
    the returned dictionary.

    Parameters
    ----------
    filename : str
        The path to the FASTA-formatted file to open.
        
    chrom_sizes: str, optional
        Path to the 2-column tsv file containing chromosome names and lengths.
        If None and include_chroms is also None, will assume hg38 chromosomes. 

    include_chroms : set or tuple or list, optional
        The exact names of chromosomes in the FASTA file to include, excluding
        all others. If None, include all chromosomes (except those specified by
        exclude_chroms). Default is None.

    verbose : bool or str, optional
        Whether to display a progress bar. If a string is passed in, use as the
        name of the progressbar. Default is False.

    Returns
    -------
    chroms : dict
        A dictionary of one-hot encoded sequences where the keys are the names
        of the chromosomes (exact strings from the header lines in the FASTA file)
        and the values are the strings.
    """
    
    from pyfaidx import Fasta

    if include_chroms is None:
        if chrom_sizes is None:
            print("Assuming human chromosomes in read_fasta_fast.")
            include_chroms = ["chr" + str(i + 1) for i in range(22)]
            include_chroms.extend(["chrX", "chrY"])
        else:
            include_chroms = load_chrom_names(chrom_sizes)

    chroms = {}
    print("Loading genome sequence from " + filename)
    fasta_index = Fasta(filename)
    for chrom in tqdm(include_chroms, disable=not verbose, desc="Reading FASTA"):
        chroms[chrom] = fasta_index[chrom][:].seq.upper()
    return chroms


class DataGenerator(torch.utils.data.Dataset):
    """A data generator for BPNet inputs.

    This generator takes in an extracted set of sequences, output signals,
    and control signals, and will return a single element with random
    jitter and reverse-complement augmentation applied. Jitter is implemented
    efficiently by taking in data that is wider than the in/out windows by
    two times the maximum jitter and windows are extracted from that.
    Essentially, if an input window is 1000 and the maximum jitter is 128, one
    would pass in data with a length of 1256 and a length 1000 window would be
    extracted starting between position 0 and 256. This  generator must be 
    wrapped by a PyTorch generator object.

    Parameters
    ----------
    sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
        A one-hot encoded tensor of `n` example sequences, each of input 
        length `in_window`. See description above for connection with jitter.

    signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
        The signals to predict, usually counts, for `n` examples with
        `t` output tasks (usually 2 if stranded, 1 otherwise), each of 
        output length `out_window`. See description above for connection 
        with jitter.
        
    masks: torch.tensor, shape=(n, t, out_window+2*max_jitter), optional
        The mask tensor of `n` examples, each of length `out_window`.

    in_window: int, optional
        The input window size. Default is 2114.

    out_window: int, optional
        The output window size. Default is 1000.

    reverse_complement: bool, optional
        Whether to reverse complement-augment half of the data. Default is True.

    random_state: int or None, optional
        Whether to use a deterministic seed or not.
    """

    def __init__(self, sequences, signals, masks=None, in_window=2114, out_window=1000,
        reverse_complement=True, random_state=None):
        self.in_window = int(in_window)
        self.out_window = int(out_window)

        self.reverse_complement = reverse_complement
        self.random_state = np.random.RandomState(random_state)

        self.signals = signals
        self.sequences = sequences
        self.masks = masks
        
        self.max_jitter = (sequences.shape[-1] - self.in_window) // 2

        assert len(signals) == len(sequences), (len(signals), len(sequences))
        if masks is not None:
            assert len(signals) == len(masks), (len(signals), len(masks))

        assert signals.shape[-1] == self.out_window + 2 * self.max_jitter, (signals.shape, self.out_window, self.max_jitter)
        assert sequences.shape[-1] == self.in_window + 2 * self.max_jitter, (sequences.shape, self.in_window, self.max_jitter)
        if masks is not None:
            assert masks.shape[-1] == self.out_window + 2 * self.max_jitter, (signals.shape, self.out_window, self.max_jitter)

        assert sequences.shape[1] == 4, sequences.shape
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert np.max(sequences.sum(axis=(1,2))) == self.in_window + 2 * self.max_jitter, np.max(sequences.sum(axis=(1,2)))
        assert set(np.sum(sequences, axis=1).flatten()).issubset(set([0,1]))
        
        to_print = "Data generator loaded " + str(len(sequences)) + " sequences of len " + str(self.in_window)
        to_print += ", profile len " + str(self.out_window) + ", with max_jitter " + str(self.max_jitter)
        to_print += ".\nRC enabled? " + str(self.reverse_complement)
        to_print += "\nMask loaded? " + str(self.masks is not None)
        print(to_print)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.max_jitter == 0:
            j = 0
        else:
            j = self.random_state.randint(self.max_jitter*2)

        X = self.sequences[idx][:, j:j+self.in_window]
        y = self.signals[idx][:, j:j+self.out_window]
        if self.masks is not None:
            m = self.masks[idx][:, j:j+self.out_window]

        if self.reverse_complement and np.random.choice(2) == 1:
            X = X[::-1][:, ::-1]
            y = y[::-1][:, ::-1]
            if self.masks is not None:
                m = m[:, ::-1]  # one strand

        X = torch.tensor(X.copy(), dtype=torch.float32)
        y = torch.tensor(y.copy())
        if self.masks is not None:
            m = torch.tensor(m.copy(), dtype=torch.bool)
            return X, y, m
        return X, y


def extract_peaks(sequences, chrom_sizes, plus_bw_path, minus_bw_path, peak_path,
                  mask_bw_path=None, in_window=2114, out_window=1000, max_jitter=0, verbose=False):
    """Extract data directly from fasta and bigWig files.

    This function will take in the file path to a fasta file and stranded
    signal and control files as well as other parameters. It will then
    extract the data to the specified window lengths with jitter added to
    each side for efficient jitter extraction. If you don't want jitter,
    set that to 0.

    Parameters
    ----------
    sequence_path: str or dictionary
        Either the path to a fasta file to read from or a dictionary where the
        keys are the unique set of chromosoms and the values are one-hot
        encoded sequences as numpy arrays or memory maps.
        
    chrom_sizes: str
        Path to the 2-column tsv file containing chromosome names and lengths.

    plus_bw_path: str
        Path to the bigWig containing the signal values on the positive strand.

    minus_bw_path: str
        Path to the bigWig containing the signal values on the negative strand.

    peak_path: str
        Path to a peak bed file. The file can have more than three columns as
        long as the first three columns are (chrom, start, end).
        
    mask_bw_path: str, optional
        Path to the bigWig containing values to load as a binary mask.

    in_window: int, optional
        The input window size. Default is 2114.

    out_window: int, optional
        The output window size. Default is 1000.

    max_jitter: int, optional
        The maximum amount of jitter to add, in either direction, to the
        midpoints that are passed in. Default is 0.

    verbose: bool, optional
        Whether to display a progress bar while loading. Default is False.

    Returns
    -------
    seqs: numpy.ndarray, shape=(n, 4, in_window+2*max_jitter)
        The extracted sequences in the same order as the chrom and mid arrays.

    signals: numpy.ndarray, shape=(n, 2, out_window+2*max_jitter)
        The extracted stranded signals in the same order as the chrom and mid
        arrays.
    """
    
    import pyBigWig
    import pandas as pd

    seqs, signals, masks = [], [], []
    in_width, out_width = in_window // 2, out_window // 2

    if isinstance(sequences, str):
        assert os.path.exists(sequences), sequences
        sequences = read_fasta_fast(sequences, chrom_sizes, verbose=verbose)

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    assert os.path.exists(plus_bw_path), plus_bw_path
    assert os.path.exists(minus_bw_path), minus_bw_path
    plus_bw = pyBigWig.open(plus_bw_path, "r")
    minus_bw = pyBigWig.open(minus_bw_path, "r")
    if mask_bw_path is not None:
        assert os.path.exists(mask_bw_path), mask_bw_path
        mask_bw = pyBigWig.open(mask_bw_path, "r")

    desc = "Loading Peaks"
    d = not verbose
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        start = mid - out_width - max_jitter
        end = mid + out_width + max_jitter
        assert start > 0, start

        sequence = sequences[chrom]

        # Load plus strand signal
        plus_sig = plus_bw.values(chrom, start, end, numpy=True)
        plus_sig = np.nan_to_num(plus_sig)

        # Load minus strand signal
        minus_sig = minus_bw.values(chrom, start, end, numpy=True)
        minus_sig = np.nan_to_num(minus_sig)

        # Append signal to growing signal list
        assert len(plus_sig) == end - start, (len(plus_sig), start, end)
        assert len(minus_sig) == end - start, (len(minus_sig), start, end)
        signals.append(np.array([plus_sig, minus_sig]))

        if mask_bw_path is not None:
            try:
                mask = mask_bw.values(chrom, start, end, numpy=True)
            except:
                print("Mask not loading for all examples.", chrom, start, end)
                mask = np.zeros((end - start,))

            # binarize to be only 1s and 0s
            mask = np.nan_to_num(mask).astype(int)
            assert len(mask) == end - start, (len(mask), start, end)
            assert set(mask.flatten()).issubset(set([0,1])), set(mask.flatten())
            # double-save the mask to broadcast to strand axis
            masks.append(np.array([mask, mask]))

        # Append sequence to growing sequence list
        s = mid - in_width - max_jitter
        e = mid + in_width + max_jitter

        if isinstance(sequence, str):
            seq = one_hot_encode(sequence[s:e]).T
        else:
            seq = sequence[s:e].T

        assert seq.shape == (4, e - s), (seq.shape, s, e)
        assert set(seq.flatten()) == set([0,1]), set(seq.flatten())
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert set(seq.sum(axis=0)).issubset(set([0, 1])), set(seq.sum(axis=0))
        assert seq.sum() <= e - s, seq
        seqs.append(seq)

    seqs = np.array(seqs)
    signals = np.array(signals)
    
    assert len(seqs) == len(signals), (seqs.shape, signals.shape)
    assert seqs.shape[1] == 4 and seqs.shape[2] == in_window + 2 * max_jitter, seqs.shape
    assert signals.shape[1] == 2 and signals.shape[2] == out_window + 2 * max_jitter, signals.shape
    
    to_print = "== In Extract Peaks =="
    to_print += "\nPeak filepath: " + peak_path
    to_print += "\nSequence length (with jitter): " + str(seqs.shape[-1])
    to_print += "\nProfile length (with jitter): " + str(signals.shape[-1])
    to_print += "\nMax jitter applied: " + str(max_jitter)
    to_print += "\nNum. Examples: " + str(len(seqs))
    to_print += "\nMask loaded? " + str(mask_bw_path is not None)
    print(to_print)
    sys.stdout.flush()

    if mask_bw_path is not None:
        masks = np.array(masks)
        assert masks.shape[1] == 2 and masks.shape[2] == out_window + 2 * max_jitter, masks.shape
        return seqs, signals, masks

    return seqs, signals


def extract_sequences(sequences, chrom_sizes, peak_path, in_window=2114, verbose=False):
    """Extract data directly from fasta files.

    This function will take in the file path to a fasta file and a bed file
    containing regions to get sequences for. It will then extract sequences
    with the specified window lengths, centered at the bed regions' midpoints.

    Parameters
    ----------
    sequence_path: str or dictionary
        Either the path to a fasta file to read from or a dictionary where the
        keys are the unique set of chromosoms and the values are one-hot
        encoded sequences as numpy arrays or memory maps.
        
    chrom_sizes: str
        Path to the 2-column tsv file containing chromosome names and lengths.

    peak_path: str
        Path to a peak bed file. The file can have more than three columns as
        long as the first three columns are (chrom, start, end).
        
    in_window: int, optional
        The input window size. Default is 2114.

    verbose: bool, optional
        Whether to display a progress bar while loading. Default is False.

    Returns
    -------
    seqs: numpy.ndarray, shape=(n, 4, in_window+2*max_jitter)
        The extracted sequences in the same order as the chrom and mid arrays.
    """
    
    import pandas as pd

    seqs = []
    in_width = in_window // 2

    if isinstance(sequences, str):
        assert os.path.exists(sequences), sequences
        sequences = read_fasta_fast(sequences, chrom_sizes, verbose=verbose)

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    desc = "Loading Peaks"
    d = not verbose
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        s = mid - in_width
        e = mid + in_width
        assert s > 0, start

        sequence = sequences[chrom]

        if isinstance(sequence, str):
            seq = one_hot_encode(sequence[s:e]).T
        else:
            seq = sequence[s:e].T

        assert seq.shape == (4, e - s), (seq.shape, s, e)
        assert set(seq.flatten()) == set([0,1]), set(seq.flatten())
        # the following asserts allow for [0,0,0,0] as a valid base encoding
        assert set(seq.sum(axis=0)).issubset(set([0, 1])), set(seq.sum(axis=0))
        assert seq.sum() <= e - s, seq
        seqs.append(seq)

    seqs = np.array(seqs)
    assert seqs.shape[1] == 4 and seqs.shape[2] == in_window, seqs.shape
    
    to_print = "== In Extract Sequences =="
    to_print += "\nPeak filepath: " + peak_path
    to_print += "\nSequence length: " + str(seqs.shape[-1])
    to_print += "\nNum. Examples: " + str(len(seqs))
    print(to_print)
    sys.stdout.flush()

    return seqs


def extract_observed_profiles(plus_bw_path, minus_bw_path, peak_path,
                              out_window=1000, verbose=False):
    """Extract data directly from bigWig files.

    This function will take in the file path to stranded bigwig files and a 
    bed file of regions. It will then extract the data to the specified window
    lengths.

    Parameters
    ----------
    plus_bw_path: str
        Path to the bigWig containing the signal values on the positive strand.

    minus_bw_path: str
        Path to the bigWig containing the signal values on the negative strand.

    peak_path: str
        Path to a peak bed file. The file can have more than three columns as
        long as the first three columns are (chrom, start, end).

    out_window: int, optional
        The output window size. Default is 1000.

    verbose: bool, optional
        Whether to display a progress bar while loading. Default is False.

    Returns
    -------
    signals: numpy.ndarray, shape=(n, 2, out_window)
        The extracted stranded signals in the same order as the peak bed file.
    """
    
    import pyBigWig
    import pandas as pd

    signals = []
    out_width = out_window // 2

    names = ['chrom', 'start', 'end']
    assert os.path.exists(peak_path), peak_path
    peaks = pd.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
        header=None, index_col=False, names=names)

    assert os.path.exists(plus_bw_path), plus_bw_path
    assert os.path.exists(minus_bw_path), minus_bw_path
    plus_bw = pyBigWig.open(plus_bw_path, "r")
    minus_bw = pyBigWig.open(minus_bw_path, "r")

    desc = "Loading Profiles"
    d = not verbose
    for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
        mid = start + (end - start) // 2
        start = mid - out_width
        end = mid + out_width
        assert start > 0, start
        
        # Load plus strand signal
        plus_sig = plus_bw.values(chrom, start, end, numpy=True)
        plus_sig = np.nan_to_num(plus_sig)

        # Load minus strand signal
        minus_sig = minus_bw.values(chrom, start, end, numpy=True)
        minus_sig = np.nan_to_num(minus_sig)

        # Append signal to growing signal list
        assert len(plus_sig) == end - start, (len(plus_sig), start, end)
        assert len(minus_sig) == end - start, (len(minus_sig), start, end)
        signals.append(np.array([plus_sig, minus_sig]))

    signals = np.array(signals)
    assert signals.shape[1] == 2 and signals.shape[2] == out_window, signals.shape
    
    to_print = "== In Extract Profiles =="
    to_print += "\nPeak filepath: " + peak_path
    to_print += "\nProfile length: " + str(signals.shape[-1])
    to_print += "\nNum. Examples: " + str(len(signals))
    print(to_print)
    sys.stdout.flush()

    return signals


def load_data_loader(genome_path, chrom_sizes, plus_bw_path, minus_bw_path, peak_path,
                     mask_bw_path=None, in_window=2114, out_window=1000, max_jitter=0,
                     batch_size=64):
    
    if mask_bw_path:
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
    else:
        sequences, profiles = extract_peaks(genome_path,
                                            chrom_sizes,
                                            plus_bw_path,
                                            minus_bw_path,
                                            peak_path,
                                            in_window=in_window,
                                            out_window=out_window,
                                            max_jitter=max_jitter,
                                            verbose=True)
        masks = None
    
    gen = DataGenerator(sequences=sequences,
                        signals=profiles,
                        masks=masks,
                        in_window=in_window,
                        out_window=out_window,
                        random_state=0)

    data_loader = torch.utils.data.DataLoader(gen,
                                              batch_size=batch_size,
                                              pin_memory=True)
    return data_loader

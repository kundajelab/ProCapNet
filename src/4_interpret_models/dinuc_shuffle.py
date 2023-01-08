# Code borrowed, modified from Jacob Schreiber
# https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/attributions.py


import numba
import numpy
import torch

@numba.jit('void(int64, int64[:], int64[:], int32[:, :], int32[:,], int32[:, :], float32[:, :, :])')
def _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, counters, shuffled_sequences):
	"""An internal function for fast shuffling using numba."""

	for i in range(n_shuffles):
		for char in chars:
			n = next_idxs_counts[char]

			next_idxs_ = numpy.arange(n)
			next_idxs_[:-1] = numpy.random.permutation(n-1)  # Keep last index same
			next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

		idx = 0
		shuffled_sequences[i, idxs[idx], 0] = 1
		for j in range(1, len(idxs)):
			char = idxs[idx]
			count = counters[i, char]
			idx = next_idxs[char, count]

			counters[i, char] += 1
			shuffled_sequences[i, idxs[idx], j] = 1


def dinuc_shuffle(sequence, n_shuffles=10, random_state=None):
	"""Given a one-hot encoded sequence, dinucleotide shuffle it.
	This function takes in a one-hot encoded sequence (not a string) and
	returns a set of one-hot encoded sequences that are dinucleotide
	shuffled. The approach constructs a transition matrix between
	nucleotides, keeps the first and last nucleotide constant, and then
	randomly at uniform selects transitions until all nucleotides have
	been observed. This is a Eulerian path. Because each nucleotide has
	the same number of transitions into it as out of it (except for the
	first and last nucleotides) the greedy algorithm does not need to
	check at each step to make sure there is still a path.
	This function has been adapted to work on PyTorch tensors instead of
	numpy arrays. Code has been adapted from
	https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
	Parameters
	----------
	sequence: torch.tensor, shape=(k, -1)
		The one-hot encoded sequence. k is usually 4 for nucleotide sequences
		but can be anything in practice.
	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Default is 10.
	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 
	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(n_shuffles, k, -1)
		The shuffled sequences.
	"""

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	chars, idxs = torch.unique(sequence.argmax(axis=0), return_inverse=True)
	chars, idxs = chars.numpy(), idxs.numpy()

	next_idxs = numpy.zeros((len(chars), sequence.shape[1]), dtype=numpy.int32)
	next_idxs_counts = numpy.zeros(max(chars)+1, dtype=numpy.int32)

	for char in chars:
		next_idxs_ = numpy.where(idxs[:-1] == char)[0]
		n = len(next_idxs_)

		next_idxs[char][:n] = next_idxs_ + 1
		next_idxs_counts[char] = n

	shuffled_sequences = numpy.zeros((n_shuffles, *sequence.shape), dtype=numpy.float32)
	counters = numpy.zeros((n_shuffles, len(chars)), dtype=numpy.int32)

	_fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, 
		counters, shuffled_sequences)
	
	shuffled_sequences = torch.from_numpy(shuffled_sequences)
	return shuffled_sequences


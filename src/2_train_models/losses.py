### Code here is from Jacob's Schreiber's
### implementation of BPNet, called BPNet-lite:
### https://github.com/jmschrei/bpnet-lite/

# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains the losses used by BPNet for training.
"""

import torch

def MNLLLoss(logps, true_counts):
    """A loss function based on the multinomial negative log-likelihood.

    This loss function takes in a tensor of normalized log probabilities such
    that the sum of each row is equal to 1 (e.g. from a log softmax) and
    an equal sized tensor of true counts and returns the probability of
    observing the true counts given the predicted probabilities under a
    multinomial distribution. Can accept tensors with 2 or more dimensions
    and averages over all except for the last axis, which is the number
    of categories.

    Adapted from Alex Tseng.

    Parameters
    ----------
    logps: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories. 

    true_counts: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories.

    Returns
    -------
    loss: float
        The multinomial log likelihood loss of the true counts given the
        predicted probabilities, averaged over all examples and all other
        dimensions.
    """

    logps = logps.reshape(logps.shape[0], -1)
    true_counts = true_counts.reshape(true_counts.shape[0], -1)

    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return -torch.mean(log_fact_sum - log_prod_fact + log_prod_exp)

def log1pMSELoss(log_predicted_counts, true_counts):
    """A MSE loss on the log(x+1) of the inputs.

    This loss will accept tensors of predicted counts and a vector of true
    counts and return the MSE on the log of the labels. The squared error
    is calculated for each position in the tensor and then averaged, regardless
    of the shape.

    Note: The predicted counts are in log space but the true counts are in the
    original count space.

    Parameters
    ----------
    log_predicted_counts: torch.tensor, shape=(n, ...)
        A tensor of log predicted counts where the first axis is the number of
        examples. Important: these values are already in log space.

    true_counts: torch.tensor, shape=(n, ...)
        A tensor of the true counts where the first axis is the number of
        examples.

    Returns
    -------
    loss: float
        The MSE loss on the log of the two inputs, averaged over all examples
        and all other dimensions.
    """

    log_true = torch.log(true_counts+1)
    return torch.nn.MSELoss()(log_predicted_counts, log_true)


import scipy.ndimage
import numpy as np

def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = torch.tensor(kernel).cuda()

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(input_tensor, kernel, padding=sigma)

    return torch.squeeze(smoothed, dim=1)


from losses import smooth_tensor_1d

def fourier_att_prior_loss(status, input_grads, freq_limit = 150,
                           limit_softness = 0.2,
                           att_prior_grad_smooth_sigma = 3
    ):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `status`: a B-tensor, where B is the batch size; each entry is 1 if
            that example is to be treated as a positive example, and 0
            otherwise
        `input_grads`: a B x L x 4 tensor, where B is the batch size, L is
            the length of the input; this needs to be the gradients of the
            input with respect to the output; this should be
            *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    assert len(input_grads.shape) == 3, input_grads.shape
    if input_grads.shape[1] == 4:
        input_grads = input_grads.swapaxes(1,2)
    assert input_grads.shape[-1] == 4, input_grads.shape
    
    abs_grads = torch.sum(torch.abs(input_grads), dim=2)

    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(
        abs_grads, att_prior_grad_smooth_sigma
    )

    # Only do the positives
    pos_grads = grads_smooth[status == 1]
    
    # Loss for positives
    if pos_grads.nelement():
        pos_fft = torch.fft.rfft(pos_grads, dim=1)
        pos_mags = torch.abs(pos_fft)
        pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
        pos_mags = pos_mags / pos_mag_sum

        # Cut off DC
        pos_mags = pos_mags[:, 1:]

        # Construct weight vector
        weights = torch.ones_like(pos_mags).cuda()
        if limit_softness is None:
            weights[:, freq_limit:] = 0
        else:
            x = torch.arange(1, pos_mags.size(1) - freq_limit + 1).float().cuda()
            weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights

        # Add up along frequency axis to get score
        pos_score = torch.sum(pos_weighted_mags, dim=1)
        pos_loss = 1 - pos_score
        return torch.mean(pos_loss)
    else:
        return torch.zeros(1).cuda()
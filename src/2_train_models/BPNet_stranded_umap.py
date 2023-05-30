### Code here is modified from Jacob's Schreiber's
### implementation of BPNet, called BPNet-lite:
### https://github.com/jmschrei/bpnet-lite/

import time 
import numpy as np
import torch

from losses import MNLLLoss, log1pMSELoss
from performance_metrics import pearson_corr, multinomial_log_probs, compute_performance_metrics

import sys
sys.path.append("../utils")
from misc import ensure_parent_dir_exists

torch.backends.cudnn.benchmark = True

class Model(torch.nn.Module):
    """A basic BPNet model with stranded profile and total count prediction.
    This is a reference implementation for BPNet. The model takes in
    one-hot encoded sequence, runs it through: 
    (1) a single wide convolution operation 
    THEN 
    (2) a user-defined number of dilated residual convolutions
    THEN
    (3a) profile predictions done using a very wide convolution layer 
    AND
    (3b) total count prediction done using an average pooling on the output
    from 2 followed by a dense layer.
    
    Parameters
    ----------
    model_save_prefix: str
        filepath to save model and performance metrics to.
    n_filters: int, optional
        The number of filters to use per convolution. Default is 64.
    n_layers: int, optional
        The number of dilated residual layers to include in the model.
        Default is 8.
    n_outputs: int, optional
        The number of outputs from the model. Generally either 1 or 2 
        depending on if the data is unstranded or stranded. Default is 2.
    alpha: float, optional
        The weight to put on the count loss.
    trimming: int or None, optional
        The amount to trim from both sides of the input window to get the
        output window. This value is removed from both sides, so the total
        number of positions removed is 2*trimming.
    """

    def __init__(self, model_save_path, n_filters=64, n_layers=8, n_outputs=2,  
        alpha=1, trimming=None):
        super(Model, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.alpha = alpha
        self.trimming = trimming or 2 ** n_layers
        self.model_save_path = model_save_path
        self.train_metrics = []

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])

        self.deconv_kernel_size = 75  # will need in forward() to crop padding
        self.fconv = torch.nn.Conv1d(n_filters, n_outputs,
                                    kernel_size=self.deconv_kernel_size)

        self.relus = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(0, self.n_layers+1)])
        self.linear = torch.nn.Linear(n_filters, 2)

    def forward(self, X):
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.relus[0](self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.relus[i+1](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        X = X[:, :, start - self.deconv_kernel_size//2 : end + self.deconv_kernel_size//2]

        y_profile = self.fconv(X)

        X = torch.mean(X, axis=2)
        y_counts = self.linear(X)

        return y_profile, y_counts

    def predict(self, X, batch_size=64, logits = False):
        with torch.no_grad():
            starts = np.arange(0, X.shape[0], batch_size)
            ends = starts + batch_size

            y_profiles, y_counts = [], []
            for start, end in zip(starts, ends):
                X_batch = X[start:end]

                y_profiles_, y_counts_ = self(X_batch)
                if not logits:  # apply softmax
                    y_profiles_ = self.log_softmax(y_profiles_)
                y_profiles.append(y_profiles_.cpu().detach().numpy())
                y_counts.append(y_counts_.cpu().detach().numpy())

            y_profiles = np.concatenate(y_profiles)
            y_counts = np.concatenate(y_counts)
            return y_profiles, y_counts

    def log_softmax(self, y_profile):
        return torch.nn.LogSoftmax(dim=-1)(y_profile)

    def save_metrics(self):
        ensure_parent_dir_exists(self.model_save_path)
        
        metrics_filename = self.model_save_path.replace(".model", "_metrics.tsv")
        with open(metrics_filename, "w") as metrics_file:
            for line in self.train_metrics:
                metrics_file.write(line + "\n")

    def save_model_arch_to_txt(self, filepath):
        ensure_parent_dir_exists(filepath)
        print(self, file=open(filepath, "w"))

    def fit_generator(self, training_data, optimizer, X_valid=None, 
        y_valid=None, max_epochs=100, batch_size=64, 
        validation_iter=100, early_stop_epochs=10, verbose=True, save=True):

        if X_valid is not None and y_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=torch.float32).cuda()
            y_valid = np.swapaxes(np.expand_dims(y_valid, 1),2,3)
            y_valid_counts = y_valid.sum(axis=2)

        columns = "Epoch\tIteration\tTraining Time\tValidation Time\t"
        columns += "Train MNLL\tTrain Count log1pMSE\t"
        columns += "Val MNLL\tVal JSD\tVal Profile Pearson\tVal Count Pearson\tVal Count log1pMSE"
        columns += "\tSaved?"
        if verbose:
            print(columns)
            self.train_metrics.append(columns)
        if save:
            ensure_parent_dir_exists(self.model_save_path)

        start = time.time()
        iteration = 0
        best_loss = float("inf")

        for epoch in range(max_epochs):
            tic = time.time()
            time_to_early_stop = False

            for X, y, mask in training_data:
                X, y, mask = X.cuda(), y.cuda(), mask.cuda()

                # Clear the optimizer and set the model to training mode
                optimizer.zero_grad()
                self.train()

                # Run forward pass
                y_profile, y_counts = self(X)
                
                # Calculate the profile and count losses
                # Here we apply the mask to the profile task only
                
                profile_loss = 0
                for y_profile_i, y_i, mask_i in zip(y_profile, y, mask):
                    # Apply mask before softmax, so masked bases do not influence softmax
                    y_profile_i = torch.masked_select(y_profile_i, mask_i)[None,...]
                    y_i = torch.masked_select(y_i, mask_i)[None,...]
                    y_profile_i = torch.nn.LogSoftmax(dim=-1)(y_profile_i)
                    profile_loss_i = MNLLLoss(y_profile_i, y_i)
                    profile_loss += profile_loss_i
                
                # divide by batch size to get the mean loss for this batch
                profile_loss = profile_loss / y.shape[0]
                
                # Calculate the counts loss
                count_loss = log1pMSELoss(y_counts, y.sum(dim=2))

                # Extract the losses for logging
                profile_loss_ = profile_loss.item()
                count_loss_ = count_loss.item()

                # Mix losses together and update the model
                loss = profile_loss + self.alpha * count_loss
                loss.backward()
                optimizer.step()

                # Report measures if desired
                if verbose and iteration % validation_iter == 0 and X_valid is not None and y_valid is not None:
                    train_time = time.time() - start

                    with torch.no_grad():
                        self.eval()

                        tic = time.time()
                        y_profile, y_counts = self.predict(X_valid)
                        valid_time = time.time() - tic

                        y_profile = np.swapaxes(np.expand_dims(y_profile, 1),2,3)
                        y_counts = np.expand_dims(y_counts, 1)

                        measures = compute_performance_metrics(y_valid, y_profile, 
                            y_valid_counts, y_counts, 7, 81)

                        line = "{}\t{}\t{:4.4}\t{:4.4}\t".format(epoch, iteration,
                            train_time, valid_time)

                        line += "{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}".format(
                            profile_loss_, count_loss_,
                            measures['nll'].mean(), 
                            np.nanmean(measures['jsd']),
                            np.nanmean(measures['profile_pearson']),
                            measures['count_pearson'].mean(), 
                            measures['count_mse'].mean()
                        )

                        valid_loss = measures['nll'].mean() + self.alpha * measures['count_mse'].mean()
                        line += "\t{}".format(valid_loss < best_loss)

                        print(line, flush=True)
                        self.train_metrics.append(line)

                        start = time.time()

                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            epoch_of_best_loss = epoch

                            if save:
                                self = self.cpu()
                                torch.save(self, self.model_save_path)
                                self = self.cuda()
                        else:
                            if epoch_of_best_loss <= epoch - early_stop_epochs:
                                time_to_early_stop = True
                                break
                iteration += 1

            if time_to_early_stop:
                break

        if save:
            self.save_metrics()

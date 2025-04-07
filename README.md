# ProCapNet: Dissecting the *cis*-regulatory syntax of transcription initiation with deep learning

This repository contains all of the code used for data downloading and processing, model training, evaluation, and interpretation, and downstream analyses used in the ProCapNet paper (preprint: https://www.biorxiv.org/content/10.1101/2024.05.28.596138v2).

In this project, we trained a neural network to predict transcription initiation (measured by PRO-cap experiments) at base-resolution using the BPNet model framework. We then interpreted the model to discover initiation-predictive sequence motifs, characterize how the epistasis between these motifs regulates the rate and positioning of transcription initiation, investigate the sequence logic behind enhancer-RNA, and more.

---

### Quick-access Roadmap: Why Are You Here?

**0. I want to make a model prediction or get model importance / contribution scores for my favorite DNA sequence using ProCapNet.**

This repository is optimized for reproducing the work in the paper; you might be more interested in using the following stand-alone Colab notebook, which downloads ProCapNet already-trained models from the ENCODE portal and shows how to make predictions or generate scores: https://colab.research.google.com/drive/18H0cUVEksnDKV0STLuemrI1rW7YDj4Gw?usp=sharing 

**1. I want to download the same dataset(s) you used.**

See `src/0_download_files/0.1_download_data.sh` -- this script shows how most data files were downloaded from the ENCODE portal, including the PRO-cap experiments, DNase peaks, candidate cis-regulatory element annotations, and RAMPAGE experiments, in all cell types used in the paper. For the chromatin accessibility and histone modification datasets used in the cross-cell-type analysis, see `src/0_download_files/0.2_download_histone_marks.sh`. For ENCODE accession IDs for each experiment, see the Supplemental Table in the manuscript.

**2. I want to know how you processed the model training data.**

PRO-cap data itself was processed from BAMs to bigWigs using this script: `src/1_process_data/1.0_process_bams.sh`.

Model training used both PRO-cap peaks ("positive" examples) and DNase peaks not overlapping PRO-cap peaks ("negative" examples). These two sets of loci were processed using the scripts `src/1_process_data/1.1_process_peaks.sh` and `src/1_process_data/1.2_process_dnase_peaks.sh`, respectively. The second script relies on outputs from the first.

**3. I want to know the exact model architecture you used.**

See `src/2_train_models/BPNet_strand_merged_umap.py` for the model architecture and training loop. All model training and evaluation is implemented in PyTorch and largely adapted from Jacob Schreiber's `bpnet-lite` repository (https://github.com/jmschrei/bpnet-lite). If you are looking for any implementation of BPNet, rather than ProCapNet specifically, check out `bpnet-lite` instead. The ProCapNet implementation is modified from `bpnet-lite` in two main ways: 1) while the model makes two-stranded predictions, the loss functions are effectively applied jointly across the two strands; and 2) ProCapNet uses mappability-aware training, where the model is not penalized for mispredictions on bases that are not uniquely mappable by sequencing reads. See the Methods section of the paper for further details.

Model hyperparameters are stored in `2_train_models/hyperparams.py`.

**4. I want to see how final model predictions or importance scores / contribution scores / DeepSHAP scores / sequence attributions were generated.**

See the scripts in `src/3_eval_models`, particularly `src/3_eval_models/eval_utils.py`, for the former.

See the scripts in `src/4_interpret_models`, particularly the `get_attributions()` function in `src/4_interpret_models/deepshap_utils.py`, for the latter.

**5. I want to know how you identified motifs from contribution scores.**

Motif identification consists of two steps: 1) running TF-MoDISco, and 2) calling instances of the motif patterns found by TF-MoDISco.

We ran TF-MoDISco using tf-modiscolite, the sped-up and cleaned-up implementation by Jacob Schreiber (https://github.com/jmschrei/tfmodisco-lite, soon to be moved to the official TF-MoDISco repository). See the function `_run_modisco()` in the file `src/5_modisco/modiscolite_utils.py` for the parameters and inputs used.

Motif instance calling was performed using the following script: `src/6_call_motifs/call_motifs_script.py` Thresholds in that script were tuned manually to minimize false-positive and false-negative rates.

**6. I want to see how a specific model evaluation, analysis, or other result in the paper was produced.**

See `src/figure_notebooks/`. Each figure and table in the paper has an associated jupyter notebook (the analyses for some supplementary figures and tables are contained within the main figure's notebook). The notebooks contain all the code to go from the outputs of the folders numbered 1-6 in `src/` (model prediction, contribution scoring, motif calling) to the plots in the figures themselves.

**7. I want to re-run your entire workflow.**

See below!

---

## Running Everything From Scratch

The script `run_everything.sh` exists as a roadmap of what order to run all of the code in. I would probably not literally run this script, because that would take weeks on a good GPU; instead I would recommend running the commands inside it one-by-one.

### Installation & Setup

First, make a directory for everything to happen in, and move into that directory. Then download this repository and move inside it.

```
mkdir -p "/users/me/procapnet"
cd "/users/me/procapnet"
git clone http://git@github.com/kundajelab/ProCapNet.git
cd ProCapNEt
```

The script `setup_project_directory.sh` will build the directory structure for all the raw + processed data, all the models saved after training, and all model outputs.

You will also probably want to set up a conda environment or similar way of having all the correct Python packages. See `conda_env_spec_file.txt` for every package and version used to run everything.

### From Downloading Data to Calling Motifs

From here on out, every step consists of just running the script `runall.sh` inside of each of the folders in `src/` numbered 1 through 6.

For example, to populate your new data directory with data, and then process that data, you would run these scripts:
```
./src/0_download_files/0_runall.sh
./src/1_process_data/1_runall.sh
```
Note that in some cases, there are additional *optional* `runall.sh` scripts, which are for if you are looking to reproduce a specific, singular result far downstream of training models. For instance, `./src/1_process_data/1_runall.sh` processes all the data you need to train a model, get contribution scores and motifs, etc., while `./src/1_process_data/1_runall_optional_annotations.sh` will also process all the extra data needed to run the model evaluations stratified by various region classifications from Figure 1 of the paper. These optional scripts are also included in `run_everything.sh`, but you can skip running them if you don't need to produce the results that depend on them.

Note #2: model training, prediction, and contribution score generation are expecting to be run on a GPU, and the `runall.sh` scripts for those steps expect an input argument specifying the ID of the GPU to use. If you're not sure what GPU ID to use, 0 is a good guess. 

Note #3: trained models are saved using unique identifiers -- namely, timestamp strings. To point the model prediction and contribution score scripts at the correct models, you will need to supply the timestamps of the models you trained. So for example, in `src/3_eval_models/3_runall.sh`, you would edit this line:
```
timestamps=( "2023-05-29_15-51-40" "2023-05-29_15-58-41" "2023-05-29_15-59-09" "2023-05-30_01-40-06" "2023-05-29_23-21-23" "2023-05-29_23-23-45" "2023-05-29_23-24-11" )
```
### Re-Creating Plots, Figures, Tables, and Stats

Any jupyter notebook in `src/figure_notebooks/` can be run once you have run the scripts in folders 1-6. They require the same conda environment as the scripts -- the package `nb_conda_kernels` was used for running notebooks inside conda environments. The notebooks show plots inside the notebooks themselves, but also save every plot to the `figures/` folder in high resolution.

Note that you may need to point notebooks to the correct GPU and correct model timestamps in the same way as with earlier scripts, but in the case of the notebooks, you'll need to edit the info directly in the first few cells of each notebook.

That's it!

---

### Code credits:
- Jacob Schreiber (@jmschrei)'s bpnetlite, tf-modiscolite, and tangermeme repositories are the backbone of this repository
- Alex Tseng (@amtseng)'s code for model training and evaluation in PyTorch is incorporated into bpnet-lite
- Otherwise code is authored by Kelly Cochran (@kellycochran)

### Repositories this project uses
- https://github.com/jmschrei/bpnet-lite
- https://github.com/jmschrei/tfmodisco-lite
- https://github.com/jmschrei/tangermeme


## Primary Dependencies
- Python ~ 3.9
- Pytorch (GPU version) v1.12 (py3.9_cuda11.6_cudnn8.3.2_0)
- numpy v1.22
- pybigwig v0.3.18
- pyfaidx v0.7
- Captum v0.5 (for deepshap)
- modisco-lite v2.0.0
- bedtools v2.30
- scikit-learn v1.1.2
- scipy v1.10
- pandas v1.4.3
- h5py v3.7
- meme v5.4.1
- statsmodels v0.13.2
- logomaker v0.8
- matplotlib v3.5.2
- jupyter_core v4.10, IPython v 8.4, and nb_conda_kernels v2.3.1

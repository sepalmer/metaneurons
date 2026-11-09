# metaneurons

This repository contains the code necessary to reproduce plots and calculations in *Coarse-graining reveals collective predictive information in a sensory population* (https://www.biorxiv.org/content/10.1101/2025.10.18.683195v1.abstract). 

# Contents
## `analysis-suite`

The `analysis-suite` folder contains utilities used throughout the rest of the project. Data in this project are handled by classes defined in `datasets.py`, which enforces a convention that any `Dataset` can be represented as a 2d numpy ndarray object, with indices `[feature, data point]` and shape `(N, P)`. For any given `Dataset` `D`, this representation can be accessed with `D.X`. Generally speaking, other methods in `analysis-suite` expect to receive numpy ndarrays in this format, not `Dataset` objects. Common analyses such as PCA or CCA, or the central tool of the project, Information Bottleneck (IB), are located in `analyses.py`. Useful transformations are included as methods in dataset classes. Note that a special dataset class `DataNatScenes` is used to load and process the raw data. 

We included the `pyentropy` package explicitly in this directory. It is available at https://github.com/robince/pyentropy.


## `analyses`

The `analyses` folder contains the Jupyter notebooks that generate the data behind the paper's figures, grouped into subfolders by topic (e.g. `IB curves`, `info decay timescale`, `cross-stim performance`). Notebooks import `analysis-suite` by appending it to `sys.path`, and write their results to a `saved` folder next to the notebook. Each notebook is set to the parameters used for the paper, and a single run writes every file its figures need (notebooks that cover several stimuli loop over them). Saved results and data are not included in this repository; raw data are expected in a `data` folder at the top level of the project, where `DataNatScenes` loads them from `data/nat_scenes_data`.

## `paper plots`

The `paper plots` folder contains one notebook per figure (or group of related panels) in the paper. These notebooks load results from the `saved` folders in `analyses`, so the corresponding analyses must be run first. Figures are written to `paper plots/plots`. All paths are relative to the repository, so notebooks can be run in place without editing.

| Plot notebook | Generating notebook(s) in `analyses` |
|---|---|
| `cross-stim hold one out` | `cross-stim performance/hold one out` |
| `cross-stim pairwise` | `cross-stim performance/run all pairwise` |
| `DMD spectra` | `basis meta neuron bank/K sweeps` |
| `IB and DMD modes` | `basis meta neuron bank/buffer sweeps` |
| `independent critic ansatz` | `architecture tests/Autocorrelations only - independent critic ansatz`, `noise correlations/trial shuffle comparisons` |
| `Info vs K` | `IB curves/all neurons small K`, `info decay vs K/1 2 4 8` |
| `Info vs K nonlinear comparison` | `IB curves/all neurons small K`, `IB curves/nonlinear features small K` |
| `Info vs prediction interval` | `info decay timescale/all neurons`, `info decay timescale/all neurons shuffled` |
| `Meta neuron timescales` | `basis meta neuron bank/K sweeps` |
| `Multiplexing` | `multiplexed prediction/(1,B,1) K=1`, `K=2`, `K=4`, `K=8`, `multiplexed prediction/longer training bound` |
| `Single neuron synergy` | `synergy and scaling/self-prediction dependence` |
| `Trial shuffle effect vs interval` | `noise correlations/dependence on timescale` |
| `Trial shuffles` | `architecture tests/Autocorrelations only - Trial shuffles`, `noise correlations/trial shuffle comparisons` |
| `Trial variation plot` | none (raw data only) |

`DMD spectra` and `Meta neuron timescales` also read the raw data directly.

# Requirements

Package versions are listed in `requirements.txt`.


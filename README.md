# metaneurons

This repository contains the code necessary to reproduce plots and calculations in *Coarse-graining reveals collective predictive information in a sensory population* (https://www.biorxiv.org/content/10.1101/2025.10.18.683195v1.abstract). 

# Contents
## `analysis-suite`

The `analysis-suite` folder contains utilities used throughout the rest of the project. Data in this project are handled by classes defined in `datasets.py`, which enforces a convention that any `Dataset` can be represented as a 2d numpy ndarray object, with indices `[feature, data point]` and shape `(N, P)`. For any given `Dataset` `D`, this representation can be accessed with `D.X`. Generally speaking, other methods in `analysis-suite` expect to receive numpy ndarrays in this format, not `Dataset` objects. Common analyses such as PCA or CCA, or the central tool of the project, Information Bottleneck (IB), are located in `analyses.py`. Useful transformations are included as methods in dataset classes. Note that a special dataset class `DataNatScenes` is used to load and process the raw data. 

We included the `pyentropy` package explicitly in this directory. It is available at https://github.com/robince/pyentropy.


## `analyses`

The `analyses` folder contains Jupyter notebooks that run the calculations in the paper, grouped into subfolders by topic (e.g. `IB curves`, `info decay timescale`, `cross-stim performance`). Notebooks import `analysis-suite` by appending it to `sys.path`, and write their results to a `saved` folder next to the notebook. Saved results and data are not included in this repository; raw data are expected in a `data` folder at the top level of the project.

## `paper plots`

The `paper plots` folder contains one notebook per figure (or group of related panels) in the paper. These notebooks load results from the `saved` folders in `analyses`, so the corresponding analyses must be run first. Figures are written to `paper plots/plots`. Note that some notebooks set `project_folder = ...`, which should be replaced with the path to this repository.

# Requirements

Package versions are listed in `requirements.txt`.


# `metaneurons/analyses` vs `retina-past-future-ib/analyses`

Compared on 2026-11-09. "New" is metaneurons and "old" is retina-past-future-ib (rpfib).
Notebooks were compared on code cells only, with outputs, comments and magics removed.
Each new notebook was matched to its closest old notebook across the whole tree, so moved
or renamed notebooks are caught. Data files were compared by path, size and MD5.

## At a glance

| | Old | New |
|---|---:|---:|
| Notebooks | 121 | 63 |
| Data files, same path in both and byte-identical | 183 | 183 |
| Data files, same path but contents differ | 8 | 8 |
| Data files only in that project | 108 (3.6 GB) | 86 (4.4 GB) |

- **14 notebooks are new analyses** (section 1).
- **6 existing notebooks changed what they compute** (section 2).
- **Shared code changed the critic argument convention** (section 3).
- The other new notebooks are the same analysis with path or parameter changes (section 7).
- Most of the old tree was dropped rather than edited (section 5).

"Paper plot" below means a notebook in `paper plots/` reads the output. "None" means no plot
notebook does, though the analysis could still support numbers quoted in the text.

## 1. New analyses (no real counterpart in rpfib)

### Stimulus predictability: new folder, 7 notebooks + `receptive_fields.py`

Applies the IB / InfoNCE machinery to the **stimulus itself** rather than to neural activity:
how well does the movie predict its own future?

| Notebook | What it does | Output | Paper plot |
|---|---|---|---|
| `nat_receptive_fields` | STA receptive fields from the natural movies, thresholded into a union pixel mask | `M_masked_mov{i}.pkl` | None |
| `bin_receptive_fields` | Same, from the binary checkerboard stimulus | `M_masked_checkerboard.pkl` (3.1 GB) | None |
| `rbm_receptive_fields_for_mask` | Masks built from RBM-derived receptive fields (thresholds 0.5, 0.8) | `M_masked_mov{i}_thresh{t}.pkl` | None |
| `rbm_receptive_field_latents` | Projects movies through RBM receptive fields to get latent variables | `M_latent_mov{i}.pkl` | None |
| `stimulus predictability from mask` | Direct estimate and IB curve (K = 10, 200) on masked pixels | not saved | None |
| `stimulus predictability from latents` | IB curve on latents, K up to M.N, plus train/test critic validation | not saved | None |
| `estimates` | Latent-stimulus MI across all 5 movies, at 3 batch sizes | not saved | None |

### Cross-stimulus generalization: 2 new notebooks

The old project had only one exploratory notebook, carried over as `basic pairwise comparison`.

| Notebook | What it does | Output | Paper plot |
|---|---|---|---|
| `run all pairwise` | Trains the coarse-grainer on each movie and tests on each other movie, 10 repeats, 16 iterations | `pairwise_10_repeats_16_iter[_5_buffer].dat` | `cross-stim pairwise` (the no-buffer file) |
| `hold one out` | Trains on 4 movies and tests on the held-out one | `hold_one_out_4_repeats_{1,8,16}_iter.dat` | `cross-stim hold one out` (16-iter file) |

### Critic and estimator validation: 4 new notebooks in `architecture tests`

| Notebook | What it does | Output | Paper plot |
|---|---|---|---|
| `critic function validation` | Train/test sweeps over epochs, hidden dim and embed dim (notes: test optimum around 5-10 epochs, hidden dim 40-80, embed dim sloppy) | not saved | None |
| `linear critic` | Linear critic across all stimuli, plus epoch validation and a look at critic weights | `linear critic (1, 0, 1).dat` | None |
| `quadratic correlation structure` | Gaussian (second-order) information estimate with regularization sweep and cross-entropy validation | `Gaussian information (…).dat` | None |
| `swap_xy` | Tests whether the critic's argument order changes the MI estimate (see section 3) | not saved | None |

### Markovianity across movies: 1 new notebook

`Markovianity/all movies estimates`: V_width 1 vs 10 for every movie, across iteration counts.
Output is `all stim estimates.dat`. No paper plot.

## 2. Existing notebooks whose analysis changed

Same file (or a renamed file), but the quantity computed is different.

| Notebook | Old | New | Paper plot |
|---|---|---|---|
| `architecture tests/Autocorrelations only` | Earlier critic and setup | Rewritten `AutocorrelationOnlyCritic` with per-neuron sub-critics, pure `f(x)` and `g(y)` terms, a separable-critic control, a trial-shuffle option, buffer 5 | `independent critic ansatz`, `Trial shuffles` |
| `info decay timescale/all neurons` | Uncompressed (K = N), one buffer value, `trim(20)` | **K = 20 meta-neurons**, buffers 0-29, `trim(60)`, 16 iterations, trial-shuffle option; saves `all neurons, high-iter.dat` | `Info vs prediction interval` |
| `Markovianity/all neurons opticflow` | Direct `train_NCE` on raw activity | **IB with K = 20 coarse-graining**; V_width 1, 10, 20; saves `all opticflow K=20.dat` | None |
| `Markovianity/small opticflow small dt` | Direct `train_NCE`, 10 neurons | **IB with K = 20**, neurons 70-72; **writes over the old filename** (see section 4) | None |
| `Markovianity/small opticflow mid dt weakest` | (variant of `mid dt`) | New variant on the 10 weakest neurons (83-92) | None |
| `mutual info vs group size/random groups` (old: `random groups all`) | K = group size, buffer 19 (tau = 20) | **K = 20 fixed**, buffer 0 (tau = 1). Would write `random groups all tau=1.dat`, but the data on disk is `random groups high iter with cg tau=1.dat` | None |

## 3. Shared code (`analysis-suite`)

Not strictly in `analyses/`, but it affects how every notebook behaves.

- **Critic argument convention renamed.** `analyses.py` now calls `critic(h, y)` rather than
  `critic(y, h)`, and the `einsum` in `models.py` flips to match. The two old conventions
  cancelled, so the score matrix is T-by-Y in both versions and only the names change: `_g`
  now takes T and `_h` takes Y. Results are not affected. The `*_swap_xy` functions exist for
  `swap_xy.ipynb` and are not the old implementation.
- **New in `tools.py`:** `record_parameters` and helpers. Notebooks now write a
  `parameters.txt` next to their saved data, which accounts for most of the `parameters.txt`
  files listed as new.
- **Removed:** `IB_test_train`.

## 4. Data files with the same path but different contents

| File | Observation |
|---|---|
| `Markovianity/saved/small opticflow small dt.dat` | 18.5 MB → 7.4 MB. **Overwritten by the changed analysis in section 2**, so the old result exists only in rpfib. |
| `Markovianity/saved/small opticflow mid dt.dat` | Same size, different bytes. Could not be unpickled for comparison without `analysis-suite` on the path. |
| `multiplexed prediction/.../longer training (1,B,1) K={1,2,4,8}.dat` | Same size and shape, values differ. Checked K=1 and K=8: mean test MI differs by about 1-3%, consistent with a re-run. K=2 and K=4 not inspected. |
| `multiplexed prediction/.../longer training bound.dat` | 63 MB → 127 MB, consistent with `N_training_repeats` going 5 → 10. |
| `mutual info vs group size/pym_estimate.py` | Small code file, not data. |

## 5. In rpfib but not carried over

Exploratory lines of work with no counterpart in metaneurons:

| Old folder or notebooks | Topic |
|---|---|
| `Dynamical mode decomposition/` (7) | DMD / Koopman analyses. `paper plots/DMD spectra` and `IB and DMD modes` still exist in metaneurons and read the basis bank directly. |
| `autocorrelation structure/` (5) | Autocorrelation vs MI |
| `meta neuron activity properties/` (4), `meta neuron nesting/` (1), `meta neuron synergy/` (2), `meta neuron time composition/` (4) | Meta-neuron characterisation |
| `multiple meta neuron shapes/` (4), `single meta neuron shapes/` (3) | Meta-neuron geometry |
| `NCE bound testing/` (2), `simulations/` (4 notebooks + scripts) | Estimator tests and toy models |
| `multiplexed prediction`: basis-meta-neuron K variants (4), `(3,B,3) K=4`, `compare shuffles`, `head start bound` | Multiplexing variants |
| `mutual info vs group size`: `cov sorted*` (3), `sorted up to 10`, `up to 10`, `small groups vs plugin`, `compare with cae` | Group-size variants |
| `IB curves`: `subset sweeps`, `subset sweep bounds`, `compare subset sweeps`, `compare CCA to infoNCE`, `compare with cae`, `reproduce figure 3D`, `reproduction run 1` (2), `nonlinear features` | IB curve variants |
| `architecture tests`: `complex encoding`, `dropout effect` (2), `embed layer marginals`, `simple decoding` | Architecture variants |
| `noise correlations/no trial matched pairs`, `info decay timescale/asdf`, `template` | Miscellaneous |
| Empty notebooks: `IB curves/N=10`, `N=25`, `N=50`, `using basis mns` (2), `architecture tests/with pure terms`, `meta neuron synergy/time synergy` | No code |

**Opticflow data not carried over:** 15 files in `info decay timescale/saved/opticflow/`
(`big`, `small sorted at dt=…`, and their shuffled versions) exist only in rpfib. The notebooks
I checked (`big groups`, `small groups`) load `bar` in their plot cells, and the `bar` files are
present. No `paper plots` notebook reads the missing files; the paper figure uses
`all neurons, high-iter.dat`.

## 6. Observed in passing (for the cleanup step)

- `paper plots/independent critic ansatz` and `Trial shuffles` look for filenames containing
  `hidden_dim:3, embed_dim:3`, but the files on disk are named `hidden_dim 3, embed_dim 3`.
  This is one instance of the filename mismatch you described.
- `IB curves/nonlinear features small K` now builds its filename with `H_…_L_…`, but
  `paper plots/Info vs K nonlinear comparison` loads `nonlinear features small K.dat`.
- `analyses/paper.tex` is a stray copy of a manuscript inside the analyses folder.

## 7. Minor differences

### Parameter or code tweaks

| Notebook | Change |
|---|---|
| `IB curves/(7,0,1) opticflow` | vs old `(7,0,1)`: K grid changed (`arange(5,200,5)` → `arange(93)[::5]+3`), `embed_dim` 8 → 30, output renamed `(7,0,1).dat`; plot cell loads `(4,0,1).dat` |
| `IB curves/all neurons small K` | movie index 4 → 3 |
| `IB curves/all neurons` | `iterations`: 1 → 16 |
| `IB curves/nonlinear features small K` | `N_training_repeats`: 10 → 2; `hidden_dim`: 32 → 100; `layers`: 2 → 3; `experiment_name`: 'nonlinear features small K' → f'nonlinear features small K; `stim`: 'opticflow' → 'fish'; `y`: ∅ → np.mean(mi_averages_test, ax ; 2 other changed line(s) |
| `Markovianity/small opticflow mid dt` | `embed_dim`: 8 → 30; `iterations`: 1 → 8 ; 1 other changed line(s) |
| `basis meta neuron bank/K sweeps` | `N_training_repeats`: 1000 → 400; `K_array`: np.array([50]) → np.array([20]); `embed_dim`: 100 → 30; `iterations`: 1 → 5 ; 5 other changed line(s) |
| `basis meta neuron bank/buffer sweeps` | `V_width`: 1 → 7; `N_training_repeats`: 400 → 200; `K`: 50 → 60 |
| `basis meta neuron bank/collective mode viewer` | Plotting rewritten (shared color scale, sorted modes, PCA comparison); `K` 10 → 50 |
| `cross-stim performance/basic pairwise comparison` (closest old: `cross-stim performance/basic pairwise comparison`) | `embed_dim`: 8 → 30; `iterations`: 1 → 16 |
| `meta neurons vs dt/(1,B,1) opticflow` | `mode_index`: 9 → 0 |
| `multiplexed prediction/(1,B,1) K=1` | Reload-and-plot cells removed |
| `multiplexed prediction/(1,B,1) K=8` | `stim`: 'opticflow' → 'fish' |
| `multiplexed prediction/longer training bound` | `K`: 8 → 20; `N_training_repeats`: 5 → 10; `iterations`: 1 → 5 |
| `synergy and scaling/self-prediction dependence` | movie index 4 → 0 |

### Only load paths changed, or `record_parameters` added, or identical

- **IB curves** (9): `(1,0,1) opticflow small K`, `(2,0,1) opticflow`, `(2,0,2) opticflow`, `(2,0,3) opticflow`, `(2,1,2) opticflow`, `(2,2,2) opticflow`, `10 most active`, `10 random`, `4 most active`
- **Markovianity** (4): `all neurons bar`, `all opticflow big dt`, `all opticflow mid dt`, `small opticflow big dt`
- **basis meta neuron bank** (1): `buffer sweep, global train-test`
- **info decay timescale** (6): `big groups`, `big sorted at dt=50ms`, `small groups`, `small sorted at dt=100ms`, `small sorted at dt=150ms`, `small sorted at dt=50ms`
- **info decay vs K** (3): `1 2 4 8`, `4-10`, `bound for 1 2 4 8`
- **meta neurons vs dt** (2): `relationship with autocorrelation`, `sloppiness at large B`
- **multiplexed prediction** (2): `(1,B,1) K=2`, `(1,B,1) K=4`
- **noise correlations** (2): `dependence on timescale`, `trial shuffle comparisons`

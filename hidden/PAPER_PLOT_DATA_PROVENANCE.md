# Paper plot data provenance

Generated 2026-11-09. Traces every generated data file loaded by `paper plots/*.ipynb` to
the notebook that produces it.

> **Update (2026-11-09): all unresolved names were fixed.** The generators in U1-U6 now write the
> names the plots load, and the three autocorrelation data files (U4, U5, #14) were renamed from
> spaces to colons, with their `parameters.txt` sections updated. The tables below record the
> state before that fix.
>
> **Update (2026-11-09): generators restructured so each needs one run.** `all neurons small K`,
> `nonlinear features small K` and `info decay timescale/all neurons` now loop over `mov_ind`;
> `dependence on timescale` loops over `shuffle_trial_per_time`. The `(shuffled) all neurons,
> high-iter.dat` files now come from `info decay timescale/all neurons shuffled`.
> `architecture tests/Autocorrelations only` is replaced for the paper by
> `Autocorrelations only - Trial shuffles` (U4) and
> `Autocorrelations only - independent critic ansatz` (U5, #14).
>
> **Update (2026-11-09): generator parameters set to the paper values** recorded in
> `parameters.txt`, so the "settings differ" table below no longer applies.

**Rule.** A file is *resolved* if an active (not commented-out) save expression in a
`metaneurons/analyses` notebook can produce the exact name the plot loads, once that notebook's
parameters are set. Otherwise it is *unresolved*. Notebooks were read, not run.

**Totals.** Resolved: 14 entries, 53 file loads, 47 unique files. Unresolved: 6 entries, 10 file
loads, 9 unique files. Commented-out loads are not counted.

**Provenance** comes from each folder's `parameters.txt`: *recorded* was written at generation
time, *inferred* was recovered later from the file's contents, and *unknown* has no notebook
listed.

## Resolved

`{stim}` = `bushes, water, leaves, fish, opticflow`. Data paths are relative to `analyses/`.

| # | Plot notebook | Generating notebook (`analyses/`) | Data file(s) | Files | Provenance |
|---:|---|---|---|---:|---|
| 1 | `Info vs K` | `IB curves/all neurons small K` | `IB curves/saved/{stim}/all neurons small K large iter.dat` | 5 | inferred |
| 2 | `Info vs K` | `info decay vs K/1 2 4 8` | `info decay vs K/saved/opticflow/1 2 4 8 93.dat` | 1 | inferred |
| 3 | `Info vs K nonlinear comparison` | `IB curves/all neurons small K` | same 5 files as #1 | 5 | inferred |
| 4 | `Info vs prediction interval` | `info decay timescale/all neurons` | `info decay timescale/saved/{stim}/all neurons, high-iter.dat` | 5 | inferred |
| 5 | `Info vs prediction interval` | `info decay timescale/all neurons` | `info decay timescale/saved/{stim}/(shuffled) all neurons, high-iter.dat` | 5 | inferred |
| 6 | `Multiplexing` | `multiplexed prediction/(1,B,1) K={1,2,4,8}` (one per K) | `multiplexed prediction/saved/opticflow/longer training (1,B,1) K={1,2,4,8}.dat` | 4 | recorded |
| 7 | `Multiplexing` | `multiplexed prediction/longer training bound` | `multiplexed prediction/saved/opticflow/longer training bound.dat` | 1 | recorded |
| 8 | `Single neuron synergy` | `synergy and scaling/self-prediction dependence` | `synergy and scaling/saved/opticflow/self-prediction dependence, N=50, tau=1.dat` | 1 | inferred |
| 9 | `Trial shuffle effect vs interval` | `noise correlations/dependence on timescale` | `noise correlations/saved/opticflow/0 - 22.dat`, `(shuffled trial per time) 0 - 22.dat` | 2 | inferred |
| 10 | `cross-stim hold one out` | `cross-stim performance/hold one out` | `cross-stim performance/saved/hold_one_out_4_repeats_16_iter.dat` | 1 | inferred |
| 11 | `DMD spectra` | `basis meta neuron bank/K sweeps` | `basis meta neuron bank/saved/opticflow/(7,0,1) K=50.dat` | 1 | inferred |
| 12 | `Meta neuron timescales` | `basis meta neuron bank/K sweeps` | same file as #11 | 1 | inferred |
| 13 | `IB and DMD modes` | `basis meta neuron bank/buffer sweeps` | `basis meta neuron bank/saved/opticflow/(1,{0..19},1) K=10.dat` | 20 | inferred |
| 14 | `independent critic ansatz` | `architecture tests/Autocorrelations only` | `architecture tests/saved/opticflow/autocorrelations only (1, 0, 1), hidden_dim:3, embed_dim:3, layers:2, iter:16, shuffle_trial_per_time:True, with pure.dat` (**see note**) | 1 | inferred |

**Note on #14.** The code produces this name, but the file on disk uses spaces instead of colons
(`hidden_dim 3, embed_dim 3, layers 2, iter 16, shuffle_trial_per_time True, with pure.dat`), so
the plot's load fails as written.

### Generators whose current settings differ from the data

The save name can be produced, but the notebook is not currently set up to reproduce the file.
Where not listed (#1-3, #6-7), the settings checked already match.

| # | Notebook | Change needed to regenerate |
|---:|---|---|
| 4 | `info decay timescale/all neurons` | `shuffle_time` True → False for the unshuffled files |
| 5 | `info decay timescale/all neurons` | `V_E_buffer_array` `arange(30)` → `[0, 14, 29]` |
| 8 | `self-prediction dependence` | `mov_ind` 0 → 4 (opticflow) |
| 9 | `dependence on timescale` | `shuffle_trial_per_time` True → False for `0 - 22.dat` |
| 10 | `hold one out` | `N_training_repeats` 1 → 4; iterations 8 → 16 |
| 11, 12 | `K sweeps` | `V_width` 12 → 7; `K_array` `[20]` → `[50]`; repeats 400 → 1000; `embed_dim` 30 → 100; iterations 5 → 1 |
| 13 | `buffer sweeps` | `V_width` 7 → 1; `K` 60 → 10; repeats 200 → 400; `embed_dim` 30 → 8; buffers `[0]` → 0-19 |
| 14 | `Autocorrelations only` | `V_E_buffer` 5 → 0; `shuffle_trial_per_time` False → True |

`#11-12` could also come from `buffer sweeps`, whose save template is identical; `parameters.txt`
attributes the file to `K sweeps`. For #1 and #4-5, `mov_ind` must be set to each stimulus in turn.

## Unresolved

| # | Plot notebook | Data file (as loaded) | On disk? | Closest code and why it fails |
|---:|---|---|---|---|
| U1 | `Info vs K nonlinear comparison` | `IB curves/saved/{stim}/nonlinear features small K.dat` (5 files) | Yes | `IB curves/nonlinear features small K` now appends ` H_{hidden} L_{layers}` to the name. The old literal name is in `../rpfib-pre-sync-backup-2026-11-09/analyses/IB curves/nonlinear features small K.ipynb` |
| U2 | `Trial shuffles` | `noise correlations/saved/opticflow/trial shuffle comparisons (1, 0, 1) N=93 high-iter.dat` | Yes | `noise correlations/trial shuffle comparisons`: the `high-iter` save line is commented out; the active line omits it |
| U3 | `independent critic ansatz` | same file as U2 | Yes | Same as U2 |
| U4 | `Trial shuffles` | `architecture tests/saved/opticflow/autocorrelations only (1, 0, 1), hidden_dim:3, embed_dim:3, layers:2 with pure.dat` | No: exists with spaces instead of colons | `architecture tests/Autocorrelations only`: the active name always includes `iter:` and `shuffle_trial_per_time:`. `parameters.txt` lists notebook as unknown |
| U5 | `independent critic ansatz` (cells 2, 9) | `architecture tests/saved/opticflow/autocorrelations only (1, 0, 1), hidden_dim:3, embed_dim:3, layers:2, iter:16 with pure.dat` | No: exists with spaces instead of colons | Same notebook: the active name always includes `shuffle_trial_per_time:`. `parameters.txt` lists notebook as unknown |
| U6 | `cross-stim pairwise` | `cross-stim performance/saved/pairwise_10_repeats_16_iter.dat` | Yes | `cross-stim performance/run all pairwise`: the active name always appends `_{V_E_buffer}_buffer`. `parameters.txt` lists notebook as unknown |

## Raw inputs (not generated, not counted)

`DMD spectra`, `Meta neuron timescales` and `Trial variation plot` read the experimental data
through `ds.DataNatScenes` from `data/nat_scenes_data` (`mov_ind=4`). `Trial variation plot`
loads nothing else.

## Files needed to keep (generating notebooks)

`IB curves/all neurons small K`, `info decay vs K/1 2 4 8`, `info decay timescale/all neurons`,
`multiplexed prediction/(1,B,1) K={1,2,4,8}`, `multiplexed prediction/longer training bound`,
`synergy and scaling/self-prediction dependence`, `noise correlations/dependence on timescale`,
`cross-stim performance/hold one out`, `basis meta neuron bank/K sweeps`,
`basis meta neuron bank/buffer sweeps`, `architecture tests/Autocorrelations only`.

Also needed once the unresolved names are fixed: `IB curves/nonlinear features small K`,
`noise correlations/trial shuffle comparisons`, `cross-stim performance/run all pairwise`.

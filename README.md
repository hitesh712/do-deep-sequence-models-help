# Do deep sequence models help on Windows API-call traces?

Code, data splits, saved results, trained models and analysis scripts for

> Ogbonna Chinweike and Hitesh Kag, "Do Deep Sequence Models Help on Windows API-Call Traces? A Duplicate-Controlled
> Comparison with Count-Based Baselines", submitted to *Computers & Security*.

The study compares a 1D-CNN, a Bi-LSTM, two Transformers, Random Forest, XGBoost, TF-IDF + logistic regression and a
1-nearest-neighbour baseline on the Oliveira Windows API-call dataset after removing exact duplicates and grouping
near-duplicate sequences, with validation-only tuning, ten deterministic seeds and repeated group-aware cross-validation.

## Contents

| Path | Contents |
|---|---|
| `code/rerun_v4.py` (`rerun_v4.ipynb`) | All training and evaluation, in tiers A–E (see `code/README_how_to_run.md`) |
| `analysis/` | Scripts `00`–`06` that turn the saved results into every table, figure and number of the paper |
| `results_v4/` | Saved results of the paper's runs: tuned hyperparameters, per-run predictions (`runs/`, `cv/`), summaries, seed-42 models (`models/`) |
| `revision_step1_audit/` | Duplicate and near-duplicate audit (see its README) |
| `revision_step2/` | Clean dataset, component audit and group-aware split assignment (see its README) |
| `reference_outputs/` | Outputs of the analysis scripts (tables, figures, statistics) for comparison |
| `sample_analysis_data.txt` | 100 MalAPI-2019 malware sequences used for the cross-dataset sanity check |
| `oliveira_calls.json` | Names of the API-call ids 0–306 of the Oliveira dataset |
| `MANIFEST.md` | Which script produces which table or figure |
| `CHANGES_v5.md` | Changes made in revision v5 |

## Reproduce the tables and figures (minutes, from the saved results)

```
pip install -r requirements.txt
python analysis/00_verify_split.py
python analysis/01_near_duplicates.py
python analysis/02_stats_bootstrap_strata_sensitivity.py
python analysis/03_adversarial_saliency_lift.py      # loads the saved Keras models (TensorFlow 2.20, Keras 3.13)
python analysis/04_make_figures.py
python analysis/05_summary_tables.py
python analysis/06_revision_analyses.py
```

Run the commands from the repository root. Outputs are written to `analysis_out/` and can be compared with
`reference_outputs/`.

## Re-run the experiments (about 15–20 h on one CPU)

See `code/README_how_to_run.md`. Use a new results folder, for example `python code/rerun_v4.py A --results results_new`;
with the default `results_v4/` the cached runs are reused. Tier D also needs the raw Oliveira file
`dynamic_api_call_sequence_per_malware_100_0_306.csv` (Kaggle or IEEE DataPort, see below) in the repository root.

## Reproducibility notes

- All runs used a CPU (no GPU) with Python 3.11.9, TensorFlow 2.20.0 and Keras 3.13.0; deep models use deterministic
  TensorFlow operations and fixed seeds.
- XGBoost results depend on the number of CPU threads. The paper's runs used 8 threads; the code fixes
  `XGB_THREADS = 8`, and with it scikit-learn 1.7.2 and XGBoost 3.2.0 reproduce every stored classical-model result exactly.
- The script that generated the candidate pairs and the split assignment is not included. The assignment is provided as
  data, and `analysis/00_verify_split.py` checks that its groups are exactly the connected components of the provided
  pair file and that no group crosses the training, validation and test splits.

## Data sources and licences

- Code: MIT licence (`LICENSE`).
- A. Oliveira, "Malware Analysis Datasets: API Call Sequences", Kaggle and IEEE DataPort, doi:10.21227/tqqm-aq14,
  licensed CC BY 4.0. The files in `revision_step1_audit/`, `revision_step2/` and `results_v4/dedup_splits.npz` are
  derived from it.
- `sample_analysis_data.txt`: 100 sequences from the MalAPI-2019 dataset of F. O. Catak et al.
  (https://github.com/ocatak/malware_api_class), MIT licence; see `THIRD_PARTY_NOTICES.md`.

## Citation

See `CITATION.cff`. Please cite the article above when using this code or these splits.

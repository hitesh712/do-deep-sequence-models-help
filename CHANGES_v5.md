# Revision v5 (28 Oct 2026)

## Verified
Every number in manuscript v4 and its supplement was recomputed from `results_v4/` and the split files. Tables 1–11 and S1–S4,
the exhaustive near-duplicate counts, the sensitivity analysis, the saliency deletion curves and the padding results all
reproduce. Scripts `00`–`03` reproduce `reference_outputs/` to floating-point precision.

## Code changes
- `code/rerun_v4.py`, tier E: consecutive repeated calls in the external sequences are collapsed before the first 100 calls
  are taken (the Oliveira encoding has no consecutive repeats; in the v4 run 61% of external positions repeated the
  previous call). 22 of the 100 sequences are shorter than 100 calls afterwards and are padded with id 307.
  Recall changes: Bi-LSTM 98→99%, Transformer 92→91%, CNN 95→82%, Random Forest 98→80%, TF-IDF + LogReg 98→95%,
  XGBoost 100→78% (XGBoost value provisional, see below).
- `analysis/02`: the AP bootstrap now uses all 2,000 resamples (v4 used the first 1,000); Table 4 AP intervals updated.
- `analysis/04`: Fig. 1 relabelled (two Transformers; "5 folds x 2" instead of the ambiguous "5x2"); Fig. 7 legend moved
  below the panels and y-axes widened so no error bar is clipped; Fig. S2 x-axis follows the data.
- `analysis/06_revision_analyses.py` (new): two-way ANOVA of the 2 x 2 ablation; benign AP / ROC-AUC per weighting strength;
  macro-F1 on the 338 test sequences more than 20 edits from every training sequence (ten seeds per model); padding attack
  on the seed-42 Random Forest, TF-IDF + LogReg and 1-NN; components of the exhaustive graph at 2, 5 and 10 edits.
- `requirements.txt`, `README.md`, `MANIFEST.md`, `code/README_how_to_run.md` updated.

## Repository layout (29 Oct 2026)
The package is self-contained for publication: `results_v4/` (the paper's runs, with the tier-E result of v5),
`revision_step1_audit/` and `revision_step2/` (audit, clean dataset, split) and the MalAPI-2019 sample are included, so
the analysis scripts run from the repository root after cloning. Added `LICENSE` (MIT, code), `THIRD_PARTY_NOTICES.md`
(CC BY 4.0 Oliveira data, MIT MalAPI-2019 sample), `CITATION.cff` and `.gitignore`. Internal planning notes from the
earlier split package and the superseded v3 scripts are not part of the public repository (they remain in the
authors' working folder).

## XGBoost reproducibility (resolved 29 Oct 2026)
XGBoost results depend on the CPU thread count, not on the library version (3.0.5, 3.1.3 and 3.2.0 give identical
models for the same thread count). The stored results were produced with 8 threads (`n_jobs=-1` on an 8-thread
machine); with 8 threads every stored XGBoost run is reproduced exactly. `rerun_v4.py` (and the notebook) now fix
`XGB_THREADS = 8`, `06` uses 8 threads, and `requirements.txt` pins scikit-learn 1.7.2 and XGBoost 3.2.0.
Final XGBoost values: external set 76% (threshold 0.64), far-subset macro-F1 0.883 ± 0.011 over ten seeds (still the
highest), padding row added to Table S3. The manuscript and supplement are v5.1 in `../manuscript/`.

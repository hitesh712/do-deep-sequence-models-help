# Which script produces which paper table / figure

Run from the repository root: `python analysis/0X_name.py` (the scripts import `analysis/common.py`; run them in
numeric order). Outputs go to `analysis_out/`.

| Script | Produces (in `analysis_out/`) | Paper items |
|---|---|---|
| `00_verify_split.py` | `split_verification_rebuilt.json` (components rebuilt from the pair file equal the stored components; no component spans two splits; exclusion counts) | Sec. 3.2.1, Table 1 |
| `01_near_duplicates.py` | `near_duplicate_audit.json`, `near_duplicate_exhaustive.json`, `nearest_distance.npz`, `test_flagged_within2.npy` | Table 1 (pair counts), Sec. 4.1, Fig. 3 |
| `02_stats_bootstrap_strata_sensitivity.py` | `cv_nb_holm.csv`, `cv_ci.csv`, `bootstrap_seed42.csv`, `strata_seed42.csv`, `sensitivity.json` | Tables 4–6, Sec. 4.3–4.4, Table S1 |
| `03_adversarial_saliency_lift.py` | `adv_seed42_all_models.csv`, `saliency_deletion.csv`, `top_saliency_calls.csv`, `lift_top_calls.csv`, `attack_diag.json` | Table S3 (deep-model rows), Table 9, Table 10, Fig. 9, Sec. 4.7 |
| `04_make_figures.py` | `figures/*.png` | Figs. 1–9, S1, S2 |
| `05_summary_tables.py` | `tables/*.csv` | Tables 2, 3, 7, 8, 11 |
| `06_revision_analyses.py` | `anova_2x2.csv`, `weighting_threshold_free.csv`, `far_subset_ten_seeds.csv`, `adv_seed42_classical.csv`, `models_classical/`, `exhaustive_components.csv` | Sec. 4.4 (sequences far from the training data), Sec. 4.5 (threshold-free measures), Sec. 4.6 (two-way ANOVA), Table S3 (classical rows), Sec. 5 (exhaustive components) |

Training and tuning (all model results, the leakage experiment, the 2 x 2 ablation incl. adversarial recall of all four conditions, the external set) are produced by `code/rerun_v4.py` (tiers A-E). Table S4 and Fig. S2 come from tier E (`results_v4/step8_crossdataset_wilson_intervals.csv`); `reference_outputs/step8_crossdataset_wilson_intervals_v5.csv` is the v5 run with consecutive repeats collapsed.

Not included: the script that built the candidate pairs, components and split assignment; the assignment is provided as
data in `revision_step2/` and checked by `00_verify_split.py`.

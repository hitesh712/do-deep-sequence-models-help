# Running the experiments

Run every command from the repository root. The inputs are already in place: `revision_step2/` (clean dataset and
split), `code/oliveira_calls.json` and `sample_analysis_data.txt`. Tier D also needs the raw Oliveira file
`dynamic_api_call_sequence_per_malware_100_0_306.csv` in the root (Kaggle or IEEE DataPort, doi:10.21227/tqqm-aq14).

Install the packages in `requirements.txt` first.

## Tiers

| Command | What it does | Time on one CPU |
|---|---|---|
| `python code/rerun_v4.py check --results results_new` | Environment check and determinism test (the "determinism test" line must say OK) | 2 min |
| `python code/rerun_v4.py A --results results_new` | Main comparison: learning rate and threshold tuned on validation, 10 seeds, all models, 1-NN row | ~7 h |
| `python code/rerun_v4.py B --results results_new` | Repeated group cross-validation (needs A) | ~4 h |
| `python code/rerun_v4.py C --results results_new` | Weighting sweep, weighting vs threshold, 2×2, adversarial recall for all four conditions (needs A) | ~2 h |
| `python code/rerun_v4.py D --results results_new` | Naive → exact-dedup → group split leakage test (needs A and the raw Oliveira file) | ~2 h |
| `python code/rerun_v4.py E --results results_new` | The 100-sequence external set (needs A) | 1 min |

Run A first; B–E can then run in any order. Every run is cached in `<results>/runs`, so a stopped tier resumes where
it stopped, and progress is logged to `<results>/run_log.txt`. Without `--results` the scripts use `results_v4/`, which
already holds the paper's runs: the deep-model runs are then loaded from the cache, while the classical models
(seconds) and the 1-NN baseline are recomputed and their stored results overwritten.

## Options (top of `rerun_v4.py`)
- `LEAK_MODELS`: remove `"transformer"` to save about 1.5 h in tier D.
- `CV_REPEATS = 1` halves tier B (the paper uses 2).
- `--no-determinism` if the determinism check fails on your hardware.
- `XGB_THREADS = 8`: keep it at 8 to reproduce the paper's XGBoost results (they depend on the thread count).

## Changes relative to the first version of the experiments
1. Random Forest and XGBoost use API n-gram counts (1–3 grams, 5,000 features), not raw positional ids.
2. The Bi-LSTM uses max-pooling over time.
3. Learning rate (CNN, Bi-LSTM, Transformers, XGBoost; C for LogReg) and the decision threshold are chosen on the validation split only.
4. The 1-NN edit-distance baseline is a row in the main comparison and the CV.
5. Weighting strength is chosen on validation and compared with moving the threshold.
6. Leakage test with the same pipeline and the same sizes on naive, exact-dedup and group splits.
7. Deterministic training is switched on.
8. XGBoost uses subsample/colsample 0.8 so its 10 seeds differ; TF-IDF + LogReg is deterministic and run once.
9. (v5) Tier E collapses consecutive repeated calls in the external sequences before taking the first 100 calls, as in the Oliveira encoding.
10. (v5) XGBoost uses a fixed 8 threads.

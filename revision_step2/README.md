# Clean dataset and group-aware split

| File | Contents |
|---|---|
| `grouped_clean_dataset.csv` | The 13,132 unique sequences of the benchmark (585 benign, 12,547 malware): one row per exact sequence, taken from its first occurrence in the raw Oliveira file (`source_row_index` = 0-based row in `dynamic_api_call_sequence_per_malware_100_0_306.csv`), with `component_id`, `exact_sequence_id`, the label and the 100 call ids. |
| `group_component_audit.csv` | One row per connected component of the screened near-duplicate graph (pairs within two edits): unique sequences, raw rows, labels, and whether the component is in the benchmark (`included_primary`). 33 components with 74 sequences are excluded: 15 contain an exact-sequence group with both labels, 18 link sequences with different labels. |
| `group_aware_split_assignments.csv` | Split of each retained sequence. Whole components stay in one split (StratifiedGroupKFold): 9,380 training, 1,876 validation and 1,876 test sequences (417 / 84 / 84 benign). |
| `group_aware_split_summary.csv` | Counts and class prevalence per split. |

`analysis/00_verify_split.py` checks that the components are exactly the connected components of
`../revision_step1_audit/near_duplicate_pairs_levenshtein_le2.csv` and that no component crosses splits. Because the
candidate screen is not exhaustive, 14.4% of test sequences still have a training sequence within two edits; see the
paper, Sections 4.1 and 4.4.

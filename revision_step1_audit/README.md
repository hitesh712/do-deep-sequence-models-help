# Duplicate and near-duplicate audit

Derived from the Oliveira API-call dataset (43,876 rows of 100 call ids; CC BY 4.0).

| File | Contents |
|---|---|
| `exact_sequence_label_conflicts.csv` | The 1,641 raw rows of the 15 exact-sequence groups that carry both labels. These groups are excluded from the benchmark rather than resolved by majority vote. |
| `near_duplicate_pairs_levenshtein_le2.csv` | 2,481 pairs of unique sequences within two Levenshtein edits found by the candidate screen, with their distances, labels and tokens. |
| `near_duplicate_audit_summary.csv` | Counts of the candidate screen. |

The candidate screen split each sequence into three exact positional blocks (34, 33 and 33 calls), compared sequences
sharing a block, and verified the 91,423 candidate pairs with the Levenshtein distance; 31 high-frequency blocks were
skipped. The screen is not exhaustive: an exhaustive search over the 13,132 retained sequences
(`analysis/01_near_duplicates.py`) finds 4,970 pairs within two edits, of which the screen found 2,430 (48.9%; the other
51 screened pairs involve excluded sequences). The split in `../revision_step2/` is built from the screened pairs; the
paper reports how much the missed pairs matter.

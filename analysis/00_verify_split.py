"""Checks that the stored split is consistent with the stored near-duplicate graph (it does NOT regenerate the split:
the script that created the candidate pairs and the assignments, create_group_aware_splits.py, is a separate file)."""
from common import *
cd=load_clean(); X=cd[COLS].to_numpy(np.int32); key={tuple(r):i for i,r in enumerate(X)}
pairs=pd.read_csv(S1+'near_duplicate_pairs_levenshtein_le2.csv'); ca=pd.read_csv(S2+'group_component_audit.csv'); summ=pd.read_csv(S2+'group_aware_split_summary.csv')
par=list(range(len(X)))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
used=0; same=0
for a,b in zip(pairs.sequence_a_tokens,pairs.sequence_b_tokens):
    ta=tuple(int(t) for t in a.split()); tb=tuple(int(t) for t in b.split())
    if ta in key and tb in key:
        used+=1; i,j=key[ta],key[tb]; par[find(i)]=find(j)
roots=np.array([find(i) for i in range(len(X))]); rebuilt=pd.Series(roots).groupby(roots).ngroup().to_numpy()
stored=cd.component_id.to_numpy()
# components agree if the partition of sequences is identical
agree=pd.crosstab(rebuilt,stored); identical=(agree.gt(0).sum(1).max()==1) and (agree.gt(0).sum(0).max()==1)
span=cd.groupby('component_id').split.nunique().max()
res={'retained_sequences':len(X),'candidate_pairs_in_file':len(pairs),'pairs_with_both_sequences_retained':used,
 'components_rebuilt_from_pairs':int(len(set(roots))),'components_stored':int(len(set(stored))),'rebuilt_partition_identical_to_stored':bool(identical),
 'max_splits_spanned_by_one_component':int(span),
 'components_excluded_total':int((ca.included_primary==0).sum()),'excluded_with_conflicting_exact_group':int(ca[ca.included_primary==0].exact_label_conflict.sum()),
 'excluded_sequences_total':int(ca[ca.included_primary==0].unique_sequences.sum()),
 'split_counts':cd.split.value_counts().to_dict(),'benign_per_split':cd[cd.malware==0].split.value_counts().to_dict(),'summary_file':summ.to_dict('records')}
json.dump(res,open(OUT+'split_verification_rebuilt.json','w'),indent=1,default=str); print(json.dumps(res,indent=1,default=str))

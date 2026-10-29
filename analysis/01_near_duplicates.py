"""Edit-distance audit of the split, exhaustive search for pairs within 2 edits, recall of the candidate screen, sensitivity mask."""
from common import *
import pickle
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
cd=load_clean(); X=cd[COLS].to_numpy(np.int32); y=cd.malware.to_numpy(); sp=cd.split.to_numpy()
tr,va,te=sp=='train',sp=='validation',sp=='test'
S=seq_str(X); Str=[S[i] for i in np.where(tr)[0]]
def nearest(idx,chunk=250):
    Q=[S[i] for i in idx]; dm=np.empty(len(Q),int)
    for i in range(0,len(Q),chunk): dm[i:i+chunk]=cdist(Q[i:i+chunk],Str,scorer=Levenshtein.distance,dtype=np.int32,workers=-1).min(1)
    return dm
dte=nearest(np.where(te)[0]); dva=nearest(np.where(va)[0]); yte=y[te]; yva=y[va]
aud={'min_cross_split_edit_distance_test_to_train':int(dte.min()),'min_cross_split_edit_distance_val_to_train':int(dva.min()),'median_nearest_edit_distance_test':float(np.median(dte)),'sequence_length':100}
for name,dm,yy in [('test',dte,yte),('val',dva,yva)]:
    for th in [0,1,2,3,5,10,20]:
        aud[f'{name}_frac_within_{th}_edits']=float((dm<=th).mean()); aud[f'{name}_benign_frac_within_{th}_edits']=float((dm[yy==0]<=th).mean())
json.dump(aud,open(OUT+'near_duplicate_audit.json','w'),indent=2); np.savez(OUT+'nearest_distance.npz',dmin_test=dte,y_test=yte,dmin_val=dva,y_val=yva)
# exhaustive search for pairs within 2 edits among the retained sequences
N=len(S); pairs=[]
for i in range(0,N,300):
    D=cdist(S[i:i+300],S,scorer=Levenshtein.distance,score_cutoff=2,dtype=np.uint8,workers=-1); a,b=np.where(D<=2)
    pairs+=[(i+p,int(q),int(D[p,q])) for p,q in zip(a,b) if q>i+p]
key={tuple(r):i for i,r in enumerate(X)}; cand=pd.read_csv(S1+'near_duplicate_pairs_levenshtein_le2.csv'); cset=set()
for a,b in zip(cand.sequence_a_tokens,cand.sequence_b_tokens):
    ta=tuple(int(t) for t in a.split()); tb=tuple(int(t) for t in b.split())
    if ta in key and tb in key: i,j=key[ta],key[tb]; cset.add((min(i,j),max(i,j)))
P=set((a,b) for a,b,d in pairs); cross=[(a,b) for a,b in P if sp[a]!=sp[b]]; trte=[(a,b) for a,b in P if {sp[a],sp[b]}=={'train','test'}]
flag=set()
for a,b in P:
    if {sp[a],sp[b]}=={'train','test'}: flag.add(a if sp[a]=='test' else b)
test_idx=np.where(te)[0]; flagged=np.array([i in flag for i in test_idx]); np.save(OUT+'test_flagged_within2.npy',flagged)
ex={'exhaustive_pairs_le2_among_retained':len(P),'at_distance_1':sum(d==1 for _,_,d in pairs),'at_distance_2':sum(d==2 for _,_,d in pairs),
 'candidate_pairs_in_file':len(cand),'candidate_pairs_with_both_retained':len(cset),'candidate_pairs_found_by_exhaustive':len(P&cset),'candidate_recall':len(P&cset)/len(P),
 'candidate_pairs_not_confirmed_by_exhaustive':len(cset-P),'cross_split_pairs':len(cross),'train_test_pairs':len(trte),
 'test_sequences_with_train_neighbour_le2':int(flagged.sum()),'test_fraction_with_train_neighbour_le2':float(flagged.mean()),'benign_test_with_train_neighbour_le2':int((flagged&(yte==0)).sum())}
json.dump(ex,open(OUT+'near_duplicate_exhaustive.json','w'),indent=2); print(json.dumps(aud,indent=1)); print(json.dumps(ex,indent=1))

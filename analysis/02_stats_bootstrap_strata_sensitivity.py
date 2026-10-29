"""Cross-validation tests (Nadeau-Bengio + Holm), component bootstrap, distance strata, near-duplicate sensitivity."""
from common import *
import itertools
from scipy import stats
from sklearn.metrics import average_precision_score
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
S3=json.load(open(R4+'step3_all_models.json')); MODELS=['cnn','bilstm','transformer','transformer_1layer','random_forest','xgboost','tfidf_logreg','nn1']
cd=load_clean(); X=cd[COLS].to_numpy(np.int32); y=cd.malware.to_numpy(); sp=cd.split.to_numpy()
Xtr,ytr,Xte,yte=X[sp=='train'],y[sp=='train'],X[sp=='test'],y[sp=='test']; gte=cd.component_id.to_numpy()[sp=='test']
nd=np.load(OUT+'nearest_distance.npz'); dmin=nd['dmin_test']; flagged=np.load(OUT+'test_flagged_within2.npy')
# ---- CV: Nadeau-Bengio corrected paired t-test, Holm over all pairs
cv=pd.read_csv(R4+'step4B_repeated_group_cv_per_run.csv'); piv=cv.pivot_table(index=['repeat','fold'],columns='model',values='macro_f1'); assert piv.shape==(10,7)
rows=[]
for a,b in itertools.combinations(piv.columns,2):
    dl=(piv[a]-piv[b]).values; n=len(dl); t=dl.mean()/np.sqrt((1/n+0.25)*dl.var(ddof=1)) if dl.var(ddof=1)>0 else np.nan
    rows.append({'model_a':a,'model_b':b,'mean_diff_a_minus_b':float(dl.mean()),'a_wins':int((dl>0).sum()),'p_raw':float(2*stats.t.sf(abs(t),n-1)) if not np.isnan(t) else 1.0})
nb=pd.DataFrame(rows); nb['p_holm']=holm(nb.p_raw.values); nb['sig']=nb.p_holm<0.05; nb.to_csv(OUT+'cv_nb_holm.csv',index=False)
tc=stats.t.ppf(0.975,9); ci=[]
for m in piv.columns:
    r={'model':m}
    for col in ['macro_f1','benign_recall','benign_average_precision','macro_f1_at0.5']:
        v=cv[cv.model==m].sort_values(['repeat','fold'])[col].values
        if np.all(np.isnan(v)): continue
        h=tc*np.sqrt((1/len(v)+0.25)*v.var(ddof=1)); r[col+'_mean']=v.mean(); r[col+'_lo']=v.mean()-h; r[col+'_hi']=v.mean()+h
    ci.append(r)
pd.DataFrame(ci).to_csv(OUT+'cv_ci.csv',index=False)
# ---- seed-42 predictions at the validation-tuned thresholds; 1-NN recomputed
thr={m:S3[m]['per_seed'][0]['threshold'] for m in MODELS}; assert S3['cnn']['per_seed'][0]['seed']==42
probs={m:np.load(R4+f'{m}_seed42_probs.npy') for m in MODELS if m!='nn1'}
S=seq_str(Xtr); Q=seq_str(Xte); arg=np.empty(len(Q),int)
for i in range(0,len(Q),250): arg[i:i+250]=cdist(Q[i:i+250],S,scorer=Levenshtein.distance,dtype=np.int32,workers=-1).argmin(1)
probs['nn1']=ytr[arg].astype(float); pred={m:(probs[m]>=thr[m]).astype(int) for m in MODELS}
for m in MODELS: assert abs(macro_f1(yte,pred[m])-S3[m]['per_seed'][0]['macro_f1'])<1e-9, m   # reproduces the stored seed-42 results
# ---- component bootstrap (2,000 resamples of whole components)
uniq,inv=np.unique(gte,return_inverse=True); members=[np.where(inv==i)[0] for i in range(len(uniq))]
rng=np.random.default_rng(42); idxs=[np.concatenate([members[j] for j in rng.integers(0,len(uniq),len(uniq))]) for _ in range(2000)]; brows=[]
for m in MODELS:
    pr=pred[m]; f=np.array([macro_f1(yte[ix],pr[ix]) for ix in idxs]); br=np.array([((pr[ix]==0)&(yte[ix]==0)).sum()/max(1,(yte[ix]==0).sum()) for ix in idxs])
    r={'model':m,'macro_f1':macro_f1(yte,pr),'f1_lo':np.percentile(f,2.5),'f1_hi':np.percentile(f,97.5),'benign_recall':float(((pr==0)&(yte==0)).sum()/(yte==0).sum()),'br_lo':np.percentile(br,2.5),'br_hi':np.percentile(br,97.5)}
    if m!='nn1':
        ap=[average_precision_score((yte[ix]==0).astype(int),1-probs[m][ix]) for ix in idxs if 0<(yte[ix]==0).sum()<len(ix)]
        r.update({'benign_ap':float(average_precision_score((yte==0).astype(int),1-probs[m])),'ap_lo':np.percentile(ap,2.5),'ap_hi':np.percentile(ap,97.5)})
    brows.append(r)
pd.DataFrame(brows).to_csv(OUT+'bootstrap_seed42.csv',index=False)
# ---- strata by distance to the nearest training sequence, and sensitivity to the flagged test sequences
bins=[(0,5,'<=5 edits'),(6,10,'6-10'),(11,20,'11-20'),(21,100,'>20')]; srows=[]
for m in ['cnn','bilstm','xgboost','random_forest','transformer']:
    for lo,hi,lab in bins:
        msk=(dmin>=lo)&(dmin<=hi); nbn=int(((yte==0)&msk).sum())
        srows.append({'model':m,'distance_bin':lab,'n':int(msk.sum()),'n_benign':nbn,'accuracy':float((pred[m][msk]==yte[msk]).mean()),'benign_recall':float(((pred[m]==0)&(yte==0)&msk).sum()/nbn) if nbn else np.nan})
pd.DataFrame(srows).to_csv(OUT+'strata_seed42.csv',index=False); keep=~flagged
sens={m:{'all':macro_f1(yte,pred[m]),'sub':macro_f1(yte[keep],pred[m][keep])} for m in MODELS}; sens['n_sub']=int(keep.sum()); sens['n_benign_sub']=int((yte[keep]==0).sum())
json.dump(sens,open(OUT+'sensitivity.json','w'),indent=1); print(nb[nb.sig].round(4).to_string()); print({m:(round(v['all'],3),round(v['sub'],3)) for m,v in sens.items() if isinstance(v,dict)})

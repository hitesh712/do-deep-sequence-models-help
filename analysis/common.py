import os, json, numpy as np, pandas as pd
R4='results_v4/'; S1='revision_step1_audit/'; S2='revision_step2/'; OUT='analysis_out/'
os.makedirs(OUT, exist_ok=True)
COLS=[f't_{i}' for i in range(100)]
def load_clean():
    """Clean dataset + split assignments, in the same order as the experiments (train/val/test masks)."""
    cd=pd.read_csv(S2+'grouped_clean_dataset.csv'); sp=pd.read_csv(S2+'group_aware_split_assignments.csv')
    cd=cd.merge(sp[['source_row_index','component_id','split']],on='source_row_index',suffixes=('','_a'),validate='one_to_one')
    return cd
def seq_str(X): return ["".join(chr(256+int(t)) for t in r) for r in X]
def holm(p):
    p=np.asarray(p,float); m=len(p); o=np.argsort(p); adj=np.empty(m); run=0.0
    for r,i in enumerate(o): run=max(run,min(1.0,(m-r)*p[i])); adj[i]=run
    return adj
def macro_f1(y,pred):
    f=[]
    for c in (0,1):
        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum(); f.append(0.0 if tp==0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(f))
def wilson(s,n,z=1.959963984540054):
    p=s/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [float(max(0,c-h)),float(min(1,c+h))]

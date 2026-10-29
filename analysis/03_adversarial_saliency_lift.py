"""Padding attack on the saved seed-42 deep models (Table S3), CNN saliency deletion (Fig. 9, Table 9), top-saliency calls and lift (Table 10)."""
from common import *
import collections, warnings; warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw); self.att=layers.MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim//num_heads)
        self.ffn=keras.Sequential([layers.Dense(ff_dim,activation='relu'),layers.Dense(embed_dim)]); self.ln1=layers.LayerNormalization(epsilon=1e-6); self.ln2=layers.LayerNormalization(epsilon=1e-6); self.drop1=layers.Dropout(rate); self.drop2=layers.Dropout(rate)
    def call(self,x,training=False,return_attention=False):
        a,sc=self.att(x,x,return_attention_scores=True); a=self.drop1(a,training=training); o1=self.ln1(x+a); f=self.drop2(self.ffn(o1),training=training); o2=self.ln2(o1+f); return (o2,sc) if return_attention else o2
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kw):
        super().__init__(**kw); self.token_emb=layers.Embedding(vocab_size,embed_dim); self.pos_emb=layers.Embedding(seq_len,embed_dim)
    def call(self,x): return self.token_emb(x)+self.pos_emb(tf.range(start=0,limit=tf.shape(x)[-1],delta=1))
CUSTOM={'TransformerBlock':TransformerBlock,'PositionalEmbedding':PositionalEmbedding}
kp=lambda m,X: m.predict(X,verbose=0,batch_size=512)[:,1]
cd=load_clean(); X=cd[COLS].to_numpy(np.int32); y=cd.malware.to_numpy(); sp=cd.split.to_numpy()
Xtr,ytr,Xva,yva,Xte,yte=X[sp=='train'],y[sp=='train'],X[sp=='validation'],y[sp=='validation'],X[sp=='test'],y[sp=='test']
S3=json.load(open(R4+'step3_all_models.json')); calls=json.load(open('oliveira_calls.json'))
def make_adv(X,ids,n,seed):   # identical to rerun_v4.py
    r=np.random.default_rng(seed); Xa=np.asarray(X).copy(); n=min(n,Xa.shape[1]-20)
    for i in range(len(Xa)):
        pos=r.choice(np.arange(20,Xa.shape[1]),size=n,replace=False); Xa[i,pos]=r.choice(ids,size=n,replace=True)
    return Xa
TOP=[c for c,_ in collections.Counter(Xtr[ytr==0].flatten().tolist()).most_common(15)]; XM,XB=Xte[yte==1],Xte[yte==0]; NL=[0,5,10,20,30]
ATT={0:(XM,XB)}
for n in NL[1:]: ATT[n]=(make_adv(XM,TOP,n,2026+n),make_adv(XB,TOP,n,4026+n))
def best_thr(yv,p):
    best=None
    for t in np.round(np.arange(0.02,0.99,0.01),2):
        s=macro_f1(yv,(p>=t).astype(int))
        if best is None or s>best[1]+1e-12 or (abs(s-best[1])<1e-12 and abs(t-.5)<abs(best[0]-.5)): best=(float(t),s)
    return best[0]
models={n:keras.models.load_model(R4+f'models/{n}_seed42.keras',custom_objects=CUSTOM) for n in ['cnn','bilstm','transformer','transformer_1layer']}
models['hardened_cnn']=keras.models.load_model(R4+'models/hardened_cnn_seed42.keras',custom_objects=CUSTOM)
thr={m:S3[m]['per_seed'][0]['threshold'] for m in models if m in S3}; thr['hardened_cnn']=best_thr(yva,kp(models['hardened_cnn'],Xva))
rows=[]
for name,m in models.items():
    for n in NL:
        pm,pb=kp(m,ATT[n][0]),kp(m,ATT[n][1]); s=int((pm>=thr[name]).sum()); lo,hi=wilson(s,len(pm))
        rows.append({'model':name,'n_injected':n,'malware_recall':s/len(pm),'wilson_low':lo,'wilson_high':hi,'benign_recall_padded':float((pb<thr[name]).mean()),'threshold':thr[name]})
pd.DataFrame(rows).to_csv(OUT+'adv_seed42_all_models.csv',index=False)
json.dump({'top15_overlap_with_malware_top15':len(set(TOP)&set(c for c,_ in collections.Counter(Xtr[ytr==1].flatten().tolist()).most_common(15))),'share_pos20plus_already_decoy':float(np.isin(XM[:,20:],TOP).mean())},open(OUT+'attack_diag.json','w'))
# ---- saliency deletion on the seed-42 CNN
PAD=307; cnn0=models['cnn']; emb_layer=next(l for l in cnn0.layers if isinstance(l,layers.Embedding)); emb_model=keras.Model(cnn0.input,emb_layer.output); dense=cnn0.layers[-1]
tin=layers.Input(shape=(100,emb_layer.output.shape[-1])); x_=tin
for L in cnn0.layers[2:-1]: x_=L(x_)
trunk=keras.Model(tin,x_); logits=lambda E: tf.matmul(trunk(E,training=False),dense.kernel)+dense.bias
pE=lambda E,bs=256: np.concatenate([tf.nn.softmax(logits(tf.constant(E[i:i+bs])),axis=-1).numpy()[:,1] for i in range(0,len(E),bs)])
E_test=emb_model.predict(Xte,verbose=0,batch_size=512); N=len(Xte); sal=np.zeros((N,100))
for i in range(0,N,256):
    E=tf.constant(E_test[i:i+256])
    with tf.GradientTape() as t: t.watch(E); lg=logits(E); mg=lg[:,1]-lg[:,0]
    sal[i:i+256]=np.abs(t.gradient(mg,E).numpy()).sum(-1)
base=pE(E_test); bp=(base>=0.5).astype(int); assert np.abs(base-np.load(R4+'cnn_seed42_probs.npy')).max()<1e-4
cnt=collections.Counter(Xtr.flatten().tolist()); flog=np.log1p(np.array([cnt.get(t,0) for t in range(308)],float))
def fm(seq,ranked,k,r):
    sel=list(ranked[:k]); sf=flog[np.asarray(seq,int)[sel]]; avail=np.setdiff1d(np.arange(len(seq)),np.asarray(sel,int)); ch=[]
    for f in sf:
        if len(avail)==0: break
        cf=flog[np.asarray(seq,int)[avail]]; dd=np.abs(cf-f); tied=avail[np.isclose(dd,dd.min())]; pk=int(r.choice(tied)); ch.append(pk); avail=avail[avail!=pk]
    return np.asarray(ch,int)
rk=np.argsort(-sal,axis=1); srows=[]
for k in [0,1,3,5,10,20,40]:
    for mode in ['saliency','random','frequency_matched']:
        r=np.random.default_rng(9000+k); pos=[]
        for i in range(N): pos.append(np.array([],int) if k==0 else rk[i,:k] if mode=='saliency' else r.choice(100,size=k,replace=False) if mode=='random' else fm(Xte[i],rk[i],k,r))
        Xt=Xte.copy(); Ez=E_test.copy()
        for i,ps in enumerate(pos):
            if len(ps): Xt[i,ps]=PAD; Ez[i,ps,:]=0.0
        for mt,pm in [('token_307',kp(cnn0,Xt)),('zero_embedding',pE(Ez))]:
            conf=np.where(bp==1,pm,1-pm); flip=((pm>=.5).astype(int)!=bp); se=1.96*conf.std(ddof=1)/np.sqrt(N)
            srows.append({'k_masked':k,'masking':mode,'mask_type':mt,'n':N,'mean_confidence':float(conf.mean()),'ci_low':float(conf.mean()-se),'ci_high':float(conf.mean()+se),'flip_rate':float(flip.mean())})
pd.DataFrame(srows).to_csv(OUT+'saliency_deletion.csv',index=False)
t1=collections.Counter(calls[int(Xte[i,rk[i,0]])] for i in range(N)); top=pd.DataFrame(t1.most_common(10),columns=['api_call','count']); top.to_csv(OUT+'top_saliency_calls.csv',index=False)
# ---- lift of the top-saliency calls (share in malware / share in benign, Katz 95% interval with 0.5 correction)
lrows=[]
for nm in top.api_call.head(5):
    tok=calls.index(nm); pres=np.any(X==tok,axis=1); a=int(pres[y==0].sum()); n0=int((y==0).sum()); b=int(pres[y==1].sum()); n1=int((y==1).sum())
    lr=np.log(((b+.5)/(n1+.5))/((a+.5)/(n0+.5))); se=np.sqrt(1/(b+.5)-1/(n1+.5)+1/(a+.5)-1/(n0+.5))
    lrows.append({'call':nm,'benign_with':a,'benign_n':n0,'malware_with':b,'malware_n':n1,'share_ben':a/n0,'share_mal':b/n1,'lift_corrected':float(np.exp(lr)),'lo':float(np.exp(lr-1.96*se)),'hi':float(np.exp(lr+1.96*se))})
pd.DataFrame(lrows).to_csv(OUT+'lift_top_calls.csv',index=False); print(top.head(5).to_string()); print(pd.DataFrame(lrows).round(3).to_string())

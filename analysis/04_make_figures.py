import json, numpy as np, pandas as pd, shutil
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
V4='results_v4/'; AN='analysis_out/'; OUT='analysis_out/figures/'
import os; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'axes.grid':True,'grid.alpha':.25,'grid.linewidth':.5})
NAME={'cnn':'1D-CNN','bilstm':'Bi-LSTM (max-pool)','xgboost':'XGBoost','random_forest':'Random Forest','transformer':'Transformer (4 layers)','transformer_1layer':'Transformer (1 layer)','tfidf_logreg':'TF-IDF + LogReg','nn1':'1-NN (edit distance)'}
COL={'cnn':'#0072B2','bilstm':'#E69F00','transformer':'#009E73','transformer_1layer':'#56B4E9','random_forest':'#D55E00','xgboost':'#CC79A7','tfidf_logreg':'#7F7F7F','nn1':'#222222'}
S3=json.load(open(V4+'step3_all_models.json'))
def vals(m,k): return np.array([r[k] for r in S3[m]['per_seed']],float)
def save(f,n): f.savefig(OUT+n,bbox_inches='tight',facecolor='white'); plt.close(f)
ORDER=['xgboost','random_forest','cnn','bilstm','tfidf_logreg','nn1','transformer_1layer','transformer']
# Fig1 pipeline
fig,ax=plt.subplots(figsize=(7.0,3.9)); ax.set_xlim(0,100); ax.set_ylim(0,60); ax.axis('off'); ax.grid(False)
def box(x,y,w,h,t,b,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fc,ec='#333',lw=.8)); ax.text(x+w/2,y+h-2.2,t,ha='center',va='top',fontsize=8.2,fontweight='bold'); ax.text(x+w/2,y+h-6.0,b,ha='center',va='top',fontsize=6.9,linespacing=1.25)
def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=1,color='#333'))
c1,c2,c3,c4='#E8F1FA','#FFF4DC','#E6F4EA','#F7E6F0'; y1=34; h=22; w=21.5
box(1,y1,w,h,'1. Dataset','Oliveira API-call\nsequences\n43,876 rows\n100 calls each\nbinary label',c1)
box(26,y1,w,h,'2. Audit','13,206 unique\nsequences (30.1%)\n33 conflicting or\nmixed-label components\nexcluded (74 sequences)\n13,132 retained',c1)
box(51,y1,w,h,'3. Group-aware split','components of sequences\nwithin 2 edits kept\nin one partition\n9,380 / 1,876 / 1,876\n(84 benign in test)',c1)
box(76,y1,w,h,'4. Features','integer ids (deep models)\nn-gram counts (RF, XGBoost)\nTF-IDF (LogReg)\nedit distance (1-NN)',c1)
y2=3
box(1,y2,w,h,'5. Models x 10 seeds','CNN, Bi-LSTM,\ntwo Transformers, RF,\nXGBoost, TF-IDF + LogReg,\n1-NN; learning rate and\nthreshold set on validation',c2)
box(26,y2,w,h,'6. Evaluation','macro-F1, benign recall,\nbenign average precision\ngroup bootstrap CIs\ngroup-aware CV (5 folds x 2)\nNadeau-Bengio + Holm',c3)
box(51,y2,w,h,'7. Ablations','naive / exact-dedup /\ngroup split comparison\nweighting vs threshold\n2x2 weighting x hardening\nnear-duplicate strata',c3)
box(76,y2,w,h,'8. Sanity checks','padding attack\n100-sequence external\nmalware set\nCNN saliency deletion',c4)
for xa in (22.5,47.5,72.5): arrow(xa+0.8,y1+h/2,xa+3.2,y1+h/2)
ax.plot([87,87],[y1-0.6,y1-3.5],color='#333',lw=1); ax.add_patch(FancyArrowPatch((87,y1-3.5),(12,y1-3.5),arrowstyle='-',lw=1,color='#333')); arrow(12,y1-3.5,12,y2+h+0.6)
for xa in (22.5,47.5,72.5): arrow(xa+0.8,y2+h/2,xa+3.2,y2+h/2)
save(fig,'fig1_pipeline.png')
# Fig2 leakage (3 splits)
L=pd.read_csv(V4+'leakage_controlled_per_run.csv'); ST=json.load(open(V4+'leakage_split_stats.json'))
splits=[('naive (random rows of the raw data)','Naive random rows','#bbbbbb'),('exact-dedup (unique sequences, random)','Exact-duplicate removal','#88b4d8'),('group-aware (components kept together)','Group-aware','#1f5fa6')]
ms=['xgboost','random_forest','cnn','bilstm','tfidf_logreg','transformer']
fig,ax=plt.subplots(figsize=(6.6,3.1)); w=0.26
for j,(sn,lab,c) in enumerate(splits):
    mu=[];sd=[]
    for m in ms:
        v=L[(L.split==sn)&(L.model==m)].macro_f1.values; mu.append(v.mean()); sd.append(v.std(ddof=1) if len(v)>1 else 0)
    ax.bar(np.arange(len(ms))+(j-1)*w,mu,w,yerr=sd,capsize=2,color=c,edgecolor='k',lw=.4,label=lab)
ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAME[m].replace(' (','\n(') for m in ms],fontsize=7.5); ax.set_ylim(0.7,1.0); ax.set_ylabel('Macro-F1 (3 seeds)'); ax.legend(frameon=False,loc='upper right',ncol=3,fontsize=7.5)
save(fig,'fig2_leakage.png')
# Fig3 near-duplicate distances
nd=np.load(AN+'nearest_distance.npz'); d=nd['dmin_test']; y=nd['y_test']; aud=json.load(open(AN+'near_duplicate_audit.json'))
fig,ax=plt.subplots(1,2,figsize=(6.6,2.9)); bins_=np.arange(0,45,2)
ax[0].hist([d[y==1],d[y==0]],bins=bins_,stacked=True,color=['#0072B2','#D55E00'],label=['malware test sequences','benign test sequences'],edgecolor='white',lw=.3)
ax[0].set_xlabel('Edit distance to nearest training sequence'); ax[0].set_ylabel('Test sequences'); ax[0].legend(frameon=False); ax[0].axvline(np.median(d),color='k',ls='--',lw=.8); ax[0].text(np.median(d)+.8,ax[0].get_ylim()[1]*.55,f'median = {np.median(d):.0f}',fontsize=7.5)
xs=np.arange(0,46)
for lab,sel,c in [('all test',np.ones(len(d),bool),'k'),('malware',y==1,'#0072B2'),('benign',y==0,'#D55E00')]: ax[1].plot(xs,[(d[sel]<=t).mean() for t in xs],label=lab,color=c)
ax[1].set_xlabel('Edit-distance threshold t'); ax[1].set_ylabel('Fraction of test within t edits'); ax[1].legend(frameon=False,loc='lower right')
save(fig,'fig3_neardup.png')

# Fig4 comparison
fig,ax=plt.subplots(1,2,figsize=(6.9,3.2))
for k,(key,tit) in enumerate([('macro_f1','Macro-F1 (validation-tuned threshold)'),('benign_average_precision','Benign-positive average precision')]):
    xs=0
    for m in ORDER:
        v=vals(m,key)
        if np.all(np.isnan(v)): xs+=1; continue
        ax[k].bar(xs,v.mean(),yerr=(v.std(ddof=1) if len(v)>1 else 0),capsize=3,color=COL[m],edgecolor='k',lw=.5)
        if len(v)>1 and v.std()>0: ax[k].scatter(np.full(len(v),xs)+np.linspace(-.18,.18,len(v)),v,s=5,color='k',alpha=.55,zorder=3)
        xs+=1
    ax[k].set_xticks(range(len(ORDER))); ax[k].set_xticklabels([NAME[m].replace(' (','\n(') for m in ORDER],rotation=60,ha='right',fontsize=7); ax[k].set_title(tit,fontsize=8.5)
ax[0].set_ylim(0.6,1.0); ax[1].set_ylim(0,1.0); ax[1].axhline(0.0448,color='r',ls='--',lw=.9); ax[1].text(-0.4,0.07,'prevalence baseline = 0.045',color='r',fontsize=7,bbox=dict(fc='white',ec='none',pad=1))
save(fig,'fig4_comparison.png')
# Fig5 CV
cv=pd.read_csv(V4+'step4B_repeated_group_cv_per_run.csv'); piv=cv.pivot_table(index=['repeat','fold'],columns='model',values='macro_f1'); order=piv.mean().sort_values(ascending=False).index.tolist()
fig,ax=plt.subplots(figsize=(6.0,3.0)); bp=ax.boxplot([piv[m] for m in order],patch_artist=True,widths=.55,medianprops=dict(color='k'),showfliers=False)
for p,m in zip(bp['boxes'],order): p.set_facecolor(COL[m]); p.set_alpha(.85)
for i,m in enumerate(order): ax.scatter(np.full(10,i+1)+np.linspace(-.15,.15,10),piv[m],s=6,color='k',zorder=3)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels([NAME[m].replace(' (','\n(') for m in order],rotation=35,ha='right',fontsize=7.5); ax.set_ylabel('Macro-F1 per CV evaluation')
save(fig,'fig5_cv.png')
# Fig6 strata
st=pd.read_csv(AN+'strata_seed42.csv'); bins=['<=5 edits','6-10','11-20','>20']; fig,ax=plt.subplots(1,2,figsize=(6.9,3.0)); wd=.26
for k,(col,tit) in enumerate([('accuracy','Accuracy'),('benign_recall','Benign recall')]):
    for j,m in enumerate(['cnn','bilstm','xgboost']):
        s=st[st.model==m].set_index('distance_bin').loc[bins]; ax[k].bar(np.arange(4)+(j-1)*wd,s[col],wd,color=COL[m],edgecolor='k',lw=.4,label=NAME[m])
    s0=st[st.model=='cnn'].set_index('distance_bin').loc[bins]; ax[k].set_xticks(range(4)); ax[k].set_xticklabels([f"{b.replace('<=','≤')}\nn={int(s0.loc[b,'n'])}\nbenign={int(s0.loc[b,'n_benign'])}" for b in bins],fontsize=7); ax[k].set_title(tit+' by distance to nearest training sequence',fontsize=8.5)
ax[0].set_ylim(.8,1.0); ax[1].set_ylim(0,1.05); h_,l_=ax[0].get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(.5,-.16))
save(fig,'fig6_strata.png')
# Fig7 weighting vs threshold
W=json.load(open(V4+'step5_weighting_ablation.json')); sts=sorted(set(r['weighting_strength'] for r in W))
fig,ax=plt.subplots(1,2,figsize=(6.9,3.0))
for k,(a,b,tit) in enumerate([('benign_recall_at0.5','benign_recall','Benign recall'),('macro_f1_at0.5','macro_f1','Macro-F1')]):
    for key,lab,c,ls in [(a,'default threshold 0.5','#D55E00','-'),(b,'validation-tuned threshold','#0072B2','--')]:
        mu=[np.mean([r[key] for r in W if r['weighting_strength']==s]) for s in sts]; sd=[np.std([r[key] for r in W if r['weighting_strength']==s],ddof=1) for s in sts]
        ax[k].errorbar(sts,mu,yerr=sd,marker='o',capsize=3,color=c,ls=ls,label=lab,ms=4)
    ax[k].set_xlabel('Class-weighting strength'); ax[k].set_title(tit+' (CNN, 10 seeds)',fontsize=8.5)
ax[0].set_ylim(.6,1.0); ax[1].set_ylim(.80,.94)   # v5: y-range covers every error bar; legend moved below the panels
h_,l_=ax[0].get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(.5,-.08))
save(fig,'fig7_weighting.png')
# Fig8 2x2
G=pd.read_csv(V4+'step7_2x2_ablation_per_seed.csv'); conds=[('hardening_0_weighting_0','neither'),('hardening_0_weighting_1','weighting only'),('hardening_1_weighting_0','hardening only'),('hardening_1_weighting_1','both')]; cc=['#999999','#0072B2','#D55E00','#7B3F98']
fig,ax=plt.subplots(1,2,figsize=(6.9,3.0))
for k,(col,tit) in enumerate([('benign_recall_at0.5','Benign recall, default threshold 0.5'),('benign_recall','Benign recall, validation-tuned threshold')]):
    for i,(c,lab) in enumerate(conds):
        v=G[G.condition==c][col].values; ax[k].bar(i,v.mean(),yerr=v.std(ddof=1),capsize=3,color=cc[i],edgecolor='k',lw=.5); ax[k].scatter(np.full(len(v),i)+np.linspace(-.18,.18,len(v)),v,s=6,color='k',alpha=.6,zorder=3)
    ax[k].set_xticks(range(4)); ax[k].set_xticklabels([l for _,l in conds],rotation=20,ha='right'); ax[k].set_title(tit,fontsize=8.5); ax[k].set_ylim(.5,1.0)
save(fig,'fig8_2x2.png')
# Fig9 saliency
sd_=pd.read_csv(AN+'saliency_deletion.csv'); fig,ax=plt.subplots(1,2,figsize=(6.9,3.0),sharey=True); mcol={'saliency':'#D55E00','random':'#0072B2','frequency_matched':'#009E73'}
for a,mt,tt in zip(ax,['token_307','zero_embedding'],['(a) Masked with an unused token id','(b) Masked by zeroing the embedding']):
    for mode,lab in [('saliency','top-saliency positions'),('random','random positions'),('frequency_matched','frequency-matched positions')]:
        s=sd_[(sd_.mask_type==mt)&(sd_.masking==mode)]; a.plot(s.k_masked,s.mean_confidence,marker='o',ms=3.5,color=mcol[mode],label=lab); a.fill_between(s.k_masked,s.ci_low,s.ci_high,color=mcol[mode],alpha=.2)
    a.set_title(tt,fontsize=8.5); a.set_xlabel('Positions masked (k)')
ax[0].set_ylabel('Mean confidence in original prediction'); ax[0].legend(frameon=False,fontsize=7.5,loc='lower left'); ax[0].set_ylim(.8,1.0)
save(fig,'fig9_saliency.png')
# Supp S1 adversarial across seeds (2x2 conditions)
NL=[0,5,10,20,30]; fig,ax=plt.subplots(1,2,figsize=(6.9,3.0))
for k,(pre,tit) in enumerate([('adv_malware_recall_','(a) Malware recall under padding'),('adv_benign_recall_','(b) Benign recall when benign is padded (diagnostic)')]):
    for i,(c,lab) in enumerate(conds):
        g=G[G.condition==c]; mu=np.array([g[f'{pre}{n}'].mean() for n in NL]); sd=np.array([g[f'{pre}{n}'].std(ddof=1) for n in NL])
        ax[k].plot(NL,mu*100,marker='o',ms=3.5,color=cc[i],label=lab); ax[k].fill_between(NL,(mu-sd)*100,(mu+sd)*100,color=cc[i],alpha=.15)
    ax[k].set_xlabel('Injected decoy calls'); ax[k].set_title(tit,fontsize=8.5); ax[k].set_xticks(NL)
ax[0].set_ylim(95,100.2); ax[0].set_ylabel('Recall (%)'); ax[0].legend(frameon=False,fontsize=7.5,loc='lower left')
save(fig,'figS1_adversarial.png')
# Supp S2 cross-dataset (val-tuned threshold)
cd_=pd.read_csv(V4+'step8_crossdataset_wilson_intervals.csv'); cd_=cd_[cd_.threshold_mode=='val-tuned threshold'].reset_index(drop=True)
fig,ax=plt.subplots(figsize=(5.8,2.7))
for i,r in cd_.iterrows():
    ax.errorbar(r.malware_recall*100,i,xerr=[[(r.malware_recall-r.wilson95_low)*100],[(r.wilson95_high-r.malware_recall)*100]],fmt='o',color=COL[r.model],capsize=3); ax.text(100.4,i,f'{r.malware_recall*100:.0f}%',va='center',fontsize=7.5)
ax.set_yticks(range(len(cd_))); ax.set_yticklabels([NAME[m] for m in cd_.model]); ax.invert_yaxis(); ax.set_xlim(min(82,cd_.wilson95_low.min()*100-3),104); ax.set_xlabel('Malware recall on the 100-sequence external set (%), Wilson 95% interval')
save(fig,'figS2_crossdata.png'); print('figs done')

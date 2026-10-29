"""Writes the tables of the paper as csv files from the saved results (means, sample SDs)."""
from common import *
S3=json.load(open(R4+'step3_all_models.json')); T=OUT+'tables/'; os.makedirs(T,exist_ok=True)
def ms(v): v=np.array(v,float); return v.mean(), (v.std(ddof=1) if len(v)>1 else 0.0)
rows=[]
for m,d in S3.items():
    r={'model':m}
    for k in ['macro_f1','macro_f1_at0.5','benign_recall','malware_recall','benign_average_precision','roc_auc','threshold','train_time_sec']:
        v=[x[k] for x in d['per_seed']]; mu,sd=ms(v); r[k+'_mean']=mu; r[k+'_sd']=sd
    r['train_time_total_sec']=float(sum(x['train_time_sec'] for x in d['per_seed'])); rows.append(r)
pd.DataFrame(rows).to_csv(T+'table3_main_comparison_and_table11_compute.csv',index=False)
L=pd.read_csv(R4+'leakage_controlled_per_run.csv'); L.groupby(['split','model']).macro_f1.agg(['mean',lambda s:s.std(ddof=1),'count']).to_csv(T+'table2_leakage_controlled.csv')
pd.DataFrame(json.load(open(R4+'leakage_split_stats.json'))).to_csv(T+'table2_leakage_split_stats.csv',index=False)
W=pd.DataFrame(json.load(open(R4+'step5_weighting_ablation.json'))); W.groupby('weighting_strength')[['benign_recall_at0.5','benign_recall','malware_recall_at0.5','macro_f1_at0.5','macro_f1']].agg(['mean',lambda s:s.std(ddof=1)]).to_csv(T+'table7_weighting_vs_threshold.csv')
G=pd.read_csv(R4+'step7_2x2_ablation_per_seed.csv'); cols=['benign_recall_at0.5','benign_recall','macro_f1','adv_malware_recall_30','adv_benign_recall_30']
G.groupby('condition')[cols].agg(['mean',lambda s:s.std(ddof=1)]).to_csv(T+'table8_2x2.csv')
print('tables written to',T)

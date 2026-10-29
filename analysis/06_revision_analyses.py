"""Analyses added in revision v5 (run after 01 and 02):
  (a) two-way ANOVA for the 2 x 2 weighting-by-hardening ablation (seeds are shared across cells, so a factorial
      model replaces the Welch tests of v4);
  (b) threshold-free metrics (benign AP, ROC-AUC) per class-weighting strength (Table 7);
  (c) macro-F1 on test sequences more than 20 edits from every training sequence, ten seeds for every model
      (RF / XGBoost / TF-IDF are refitted; each refit must reproduce the stored full-test result);
  (d) padding attack on the seed-42 Random Forest, XGBoost, TF-IDF + LogReg and 1-NN (Table S3), and the fitted
      classical models saved to analysis_out/models_classical/;
  (e) connected components of the exhaustive near-duplicate graph at 2, 5 and 10 edits.
"""
from common import *
import collections, pickle
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein

S3 = json.load(open(R4 + 'step3_all_models.json')); TUNE = json.load(open(R4 + 'tuning.json'))
cd = load_clean(); X = cd[COLS].to_numpy(np.int32); y = cd.malware.to_numpy(); sp = cd.split.to_numpy()
Xtr, ytr, Xva, yva, Xte, yte = X[sp == 'train'], y[sp == 'train'], X[sp == 'validation'], y[sp == 'validation'], X[sp == 'test'], y[sp == 'test']
spl = np.load(R4 + 'dedup_splits.npz'); assert (spl['X_test'] == Xte).all() and (spl['X_train'] == Xtr).all()
dmin = np.load(OUT + 'nearest_distance.npz')['dmin_test']; far = dmin > 20

# ---- (a) two-way ANOVA, sum-coded main effects (effect = level 1 minus level 0)
G = pd.read_csv(R4 + 'step7_2x2_ablation_per_seed.csv'); W_ = G.weighting.astype(int).to_numpy(); H_ = G.hardening.astype(int).to_numpy()
Xs = np.column_stack([np.ones(len(G)), 2 * W_ - 1, 2 * H_ - 1, (2 * W_ - 1) * (2 * H_ - 1)]); rows = []
for col in ['benign_recall_at0.5', 'benign_recall', 'macro_f1']:
    yy = G[col].to_numpy(float); b = np.linalg.lstsq(Xs, yy, rcond=None)[0]; r = yy - Xs @ b; dfree = len(yy) - 4
    cov = (r @ r / dfree) * np.linalg.inv(Xs.T @ Xs)
    for name, i in [('weighting', 1), ('hardening', 2), ('interaction', 3)]:
        t = b[i] / np.sqrt(cov[i, i]); rows.append({'metric': col, 'term': name, 'effect': float(2 * b[i]) if i < 3 else float(4 * b[i]), 'F': float(t * t), 'df_resid': dfree, 'p': float(2 * stats.t.sf(abs(t), dfree))})
pd.DataFrame(rows).to_csv(OUT + 'anova_2x2.csv', index=False)

# ---- (b) threshold-free metrics per weighting strength
Wj = pd.DataFrame(json.load(open(R4 + 'step5_weighting_ablation.json')))
Wj.groupby('weighting_strength')[['benign_average_precision', 'roc_auc']].agg(['mean', lambda s: s.std(ddof=1)]).to_csv(OUT + 'weighting_threshold_free.csv')

# ---- classical models: exact copy of fit_classical in code/rerun_v4.py
def docs(A): return [" ".join(str(t) for t in r) for r in A]
def fit_classical(name, hp, seed, Xf, yf):
    if name in ('random_forest', 'xgboost'):
        vec = CountVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", max_features=5000, lowercase=False); F = vec.fit_transform(docs(Xf)).astype(np.float32)
        mdl = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=seed) if name == 'random_forest' else \
              xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=hp['learning_rate'], subsample=0.8, colsample_bytree=0.8, n_jobs=8, random_state=seed, eval_metric='logloss', tree_method='hist')   # 8 threads as in the published runs
    else:
        vec = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", max_features=5000); F = vec.fit_transform(docs(Xf)); mdl = LogisticRegression(C=hp['C'], max_iter=1000, random_state=seed)
    mdl.fit(F, yf); return vec, mdl, (lambda A: mdl.predict_proba(vec.transform(docs(A)))[:, 1])

# ---- (c) far subset, ten seeds per model
St = seq_str(Xtr); Q = seq_str(Xte); arg = np.empty(len(Q), int)
for i in range(0, len(Q), 250): arg[i:i + 250] = cdist(Q[i:i + 250], St, scorer=Levenshtein.distance, dtype=np.int32, workers=-1).argmin(1)
far_rows = []; saved = {}; reproducible = {}
for m in ['xgboost', 'random_forest', 'cnn', 'bilstm', 'tfidf_logreg', 'nn1', 'transformer_1layer', 'transformer']:
    full, sub, ben = [], [], []
    for r in S3[m]['per_seed']:
        if m == 'nn1': p = ytr[arg].astype(float)
        elif m in ('cnn', 'bilstm', 'transformer', 'transformer_1layer'): p = np.load(R4 + f"runs/{m}__lr{TUNE[m]['lr']:g}__seed{r['seed']}__main.npz")['test']
        elif m == 'tfidf_logreg' and 'tfidf_logreg' in saved: p = saved['tfidf_logreg'][2](Xte)
        else:
            f_ = fit_classical(m, TUNE[m], r['seed'], Xtr, ytr); p = f_[2](Xte)
            if r['seed'] == 42:
                # tree ensembles can change between library versions: refits are used only if they reproduce the stored run
                reproducible[m] = abs(macro_f1(yte, (p >= r['threshold']).astype(int)) - r['macro_f1']) < 1e-9
                if reproducible[m]: saved[m] = f_
                else:
                    print(f'WARNING: {m} refit does not reproduce the stored seed-42 run with this library version; using the saved seed-42 probabilities only')
                    p = np.load(R4 + f'{m}_seed42_probs.npy')
        pr = (p >= r['threshold']).astype(int)
        assert abs(macro_f1(yte, pr) - r['macro_f1']) < 1e-9, (m, r['seed'], macro_f1(yte, pr), r['macro_f1'])   # every row reproduces the stored run
        full.append(macro_f1(yte, pr)); sub.append(macro_f1(yte[far], pr[far])); ben.append(((pr == 0) & (yte == 0) & far).sum() / ((yte == 0) & far).sum())
        if reproducible.get(m) is False: break
    far_rows.append({'model': m, 'seeds': len(full), 'n_far': int(far.sum()), 'n_far_benign': int(((yte == 0) & far).sum()), 'macro_f1_full_mean': np.mean(full), 'macro_f1_full_sd': np.std(full, ddof=1) if len(full) > 1 else np.nan,
                     'macro_f1_far_mean': np.mean(sub), 'macro_f1_far_sd': np.std(sub, ddof=1) if len(sub) > 1 else np.nan, 'benign_recall_far_mean': np.mean(ben), 'drop_mean': np.mean(full) - np.mean(sub)})
pd.DataFrame(far_rows).to_csv(OUT + 'far_subset_ten_seeds.csv', index=False)

# ---- (d) padding attack on the seed-42 classical models and 1-NN (same attack sets as rerun_v4.py)
def make_adv(A, ids, n, seed):
    r = np.random.default_rng(seed); Xa = np.asarray(A).copy(); n = min(n, Xa.shape[1] - 20)
    for i in range(len(Xa)):
        pos = r.choice(np.arange(20, Xa.shape[1]), size=n, replace=False); Xa[i, pos] = r.choice(ids, size=n, replace=True)
    return Xa
TOP = [c for c, _ in collections.Counter(Xtr[ytr == 0].flatten().tolist()).most_common(15)]; XM, XB = Xte[yte == 1], Xte[yte == 0]; NL = [0, 5, 10, 20, 30]
ATT = {0: (XM, XB)}
for n in NL[1:]: ATT[n] = (make_adv(XM, TOP, n, 2026 + n), make_adv(XB, TOP, n, 4026 + n))
def nn1(A): return ytr[np.concatenate([cdist(seq_str(A[i:i + 250]), St, scorer=Levenshtein.distance, dtype=np.int32, workers=-1).argmin(1) for i in range(0, len(A), 250)])].astype(float)
adv = []
for m in ['random_forest', 'xgboost', 'tfidf_logreg', 'nn1']:
    if m != 'nn1' and m not in saved: print(f'WARNING: {m} skipped in the padding attack (refit not reproducible here)'); continue
    thr = S3[m]['per_seed'][0]['threshold']; pf = nn1 if m == 'nn1' else saved[m][2]
    for n in NL:
        pm, pb = pf(ATT[n][0]), pf(ATT[n][1]); s = int((pm >= thr).sum()); lo, hi = wilson(s, len(pm))
        adv.append({'model': m, 'n_injected': n, 'malware_recall': s / len(pm), 'wilson_low': lo, 'wilson_high': hi, 'benign_recall_padded': float((pb < thr).mean()), 'threshold': thr})
pd.DataFrame(adv).to_csv(OUT + 'adv_seed42_classical.csv', index=False)
os.makedirs(OUT + 'models_classical', exist_ok=True)
for m, (vec, mdl, _) in saved.items(): pickle.dump({'vectorizer': vec, 'model': mdl, 'threshold': S3[m]['per_seed'][0]['threshold']}, open(OUT + f'models_classical/{m}_seed42.pkl', 'wb'))

# ---- (e) components of the exhaustive near-duplicate graph
S = seq_str(X); pairs = {2: [], 5: [], 10: []}
for i in range(0, len(S), 800):
    D = cdist(S[i:i + 800], S, scorer=Levenshtein.distance, score_cutoff=10, dtype=np.uint8, workers=-1)
    for th in pairs:
        a, b = np.where(D <= th); k = (i + a) < b; pairs[th].append(np.column_stack([i + a[k], b[k]]))
comp = []
for th, pp in pairs.items():
    par = np.arange(len(S))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in np.concatenate(pp):
        ra, rb = f(a), f(b)
        if ra != rb: par[ra] = rb
    roots = np.array([f(i) for i in range(len(S))]); u, c = np.unique(roots, return_counts=True); big = u[c.argmax()]
    comp.append({'max_edits': th, 'pairs': int(sum(len(p) for p in pp)), 'components': len(u), 'largest_component': int(c.max()), 'largest_share': float(c.max() / len(S)), 'benign_in_largest': int((y[roots == big] == 0).sum()),
                 'components_with_both_labels': int(pd.Series(y).groupby(roots).nunique().gt(1).sum())})
pd.DataFrame(comp).to_csv(OUT + 'exhaustive_components.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string()); print(pd.read_csv(OUT + 'weighting_threshold_free.csv').to_string())
print(pd.DataFrame(far_rows).round(4).to_string()); print(pd.DataFrame(adv).round(4).to_string()); print(pd.DataFrame(comp).to_string())

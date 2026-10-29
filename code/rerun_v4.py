#!/usr/bin/env python
"""
rerun_v4.py - re-runs the experiments after the guide's technical feedback.

Changes vs. the v3 notebook (all in one place, all configurable below):
  1. Random Forest / XGBoost use API n-gram COUNTS (1-3 grams), not raw integer ids by position.
  2. Bi-LSTM has max-pooling over time (like the CNN's global max-pool / Transformer's pooling).
  3. Learning rate (deep models, XGBoost, LogReg C) and the decision threshold of EVERY model are chosen on
     the VALIDATION split only; the test split is never used for any choice.
  4. 1-NN (edit distance) baseline is a row everywhere (main comparison and CV).
  5. Class-weighting strength is chosen on validation; weighting is compared with just moving the threshold.
  6. Leakage experiment: naive split -> exact-dedup split -> group split, same pipeline, same sizes.
  7. Deterministic training (tf.config.experimental.enable_op_determinism + seeds).
  8. Adversarial recall is computed for ALL four 2x2 conditions and all 10 seeds.
v5 changes: tier E collapses consecutive repeated calls in the external sequences before taking the first 100 calls,
  matching the Oliveira encoding (the v4 run took the first 100 raw tokens, 61% of which repeated the previous call);
  XGBoost uses a fixed 8 threads (XGB_THREADS) because its results depend on the thread count.

Usage (terminal, in the folder that contains this file):
    python rerun_v4.py check          # 2-minute environment + determinism test + time estimate
    python rerun_v4.py A              # main comparison (tuning + 10 seeds)           ~6 h
    python rerun_v4.py B              # repeated group CV (needs A)                   ~4 h
    python rerun_v4.py C              # weighting sweep + 2x2 + adversarial (needs A) ~1.5 h
    python rerun_v4.py D              # naive / exact-dedup / group leakage test (needs A) ~2 h
    python rerun_v4.py E              # external 100-sequence set (needs A)           ~1 min
    python rerun_v4.py package        # zips results_v4 for you to send back
Everything is cached in results_v4/runs, so you can stop (Ctrl+C) and re-run the same command to resume.
"""
import argparse, os, sys, json, time, hashlib, collections, itertools, warnings, shutil, platform
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("tier", choices=["check", "A", "B", "C", "D", "E", "package"])
ap.add_argument("--smoke", action="store_true", help="tiny/fast settings to test the code path")
ap.add_argument("--results", default="results_v4")
ap.add_argument("--step2", default="revision_step2")
ap.add_argument("--raw", default="dynamic_api_call_sequence_per_malware_100_0_306.csv")
ap.add_argument("--cross", default="sample_analysis_data.txt")
ap.add_argument("--models", default="", help="comma list to restrict tier A, e.g. cnn,bilstm")
ap.add_argument("--no-determinism", action="store_true")
args = ap.parse_args()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
if not args.no_determinism:
    tf.config.experimental.enable_op_determinism()
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc, average_precision_score
import xgboost as xgb
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein

# ----------------------------------------------------------------------------- configuration
SMOKE = args.smoke
RES = args.results + ("_smoke" if SMOKE else "")
os.makedirs(f"{RES}/runs", exist_ok=True); os.makedirs(f"{RES}/models", exist_ok=True); os.makedirs(f"{RES}/cv", exist_ok=True)
CONFIG = {"vocab_size": 308, "seq_len": 100, "embed_dim": 128, "num_heads": 8, "ff_dim": 256, "lstm_hidden": 128,
          "num_transformer_layers": 4, "batch_size": 256, "epochs": 25, "early_stopping_patience": 4}
SEEDS = [42, 7, 123, 2024, 999, 17, 37, 73, 101, 2026]
TUNE_SEEDS = [42, 7]
LR_GRID = [3e-4, 1e-3, 3e-3]
XGB_LR_GRID = [0.05, 0.1, 0.2]
LOGREG_C_GRID = [0.1, 1.0, 10.0]
THR_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
DEEP = ["cnn", "bilstm", "transformer", "transformer_1layer"]
CLASSICAL = ["random_forest", "xgboost", "tfidf_logreg"]
MAIN_MODELS = DEEP + CLASSICAL + ["nn1"]
CV_MODELS = ["cnn", "bilstm", "transformer", "random_forest", "xgboost", "tfidf_logreg", "nn1"]   # 1-layer Transformer skipped in CV (time)
CV_SPLITS, CV_REPEATS = 5, 2
WEIGHT_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
HARDEN_INJECT = 10
ATTACK_N = [5, 10, 20, 30]
LEAK_SEEDS = [42, 7, 123]
LEAK_MODELS = ["cnn", "bilstm", "transformer", "random_forest", "xgboost", "tfidf_logreg", "nn1"]   # remove "transformer" to save ~2 h
XGB_THREADS = 8   # v5: XGBoost (hist) results depend on the thread count; the published runs used 8 threads (n_jobs=-1 on an 8-thread CPU)
if SMOKE:
    CONFIG["epochs"] = 1; SEEDS = [42, 7]; TUNE_SEEDS = [42]; LR_GRID = [1e-3, 3e-3]; XGB_LR_GRID = [0.1, 0.2]; LOGREG_C_GRID = [1.0, 10.0]
    WEIGHT_STRENGTHS = [0.0, 0.75]; ATTACK_N = [10]; LEAK_SEEDS = [42]; CV_REPEATS = 1

def log(*a):
    msg = " ".join(str(x) for x in a); print(msg, flush=True)
    with open(f"{RES}/run_log.txt", "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")

# ----------------------------------------------------------------------------- data
def load_clean():
    cd = pd.read_csv(os.path.join(args.step2, "grouped_clean_dataset.csv"))
    sp = pd.read_csv(os.path.join(args.step2, "group_aware_split_assignments.csv"))
    cd = cd.merge(sp[["source_row_index", "component_id", "split"]], on="source_row_index", how="inner", suffixes=("", "_a"), validate="one_to_one")
    cols = [f"t_{i}" for i in range(100)]
    X = cd[cols].to_numpy("int32"); y = cd["malware"].to_numpy("int32"); g = cd["component_id"].to_numpy(); s = cd["split"].to_numpy()
    return X, y, g, s

def smoke_sub(X, y, n_mal, n_ben, seed=0):
    rng = np.random.default_rng(seed); im = np.where(y == 1)[0]; ib = np.where(y == 0)[0]
    idx = np.concatenate([rng.choice(im, min(n_mal, len(im)), replace=False), rng.choice(ib, min(n_ben, len(ib)), replace=False)])
    return idx

X_ALL, Y_ALL, G_ALL, S_ALL = load_clean()
tr, va, te = S_ALL == "train", S_ALL == "validation", S_ALL == "test"
X_train, y_train, X_val, y_val, X_test, y_test = X_ALL[tr], Y_ALL[tr], X_ALL[va], Y_ALL[va], X_ALL[te], Y_ALL[te]
G_train = G_ALL[tr]
if SMOKE:
    i1 = smoke_sub(X_train, y_train, 500, 60, 1); i2 = smoke_sub(X_val, y_val, 150, 30, 2); i3 = smoke_sub(X_test, y_test, 150, 30, 3)
    X_train, y_train, X_val, y_val, X_test, y_test = X_train[i1], y_train[i1], X_val[i2], y_val[i2], X_test[i3], y_test[i3]
    keep = np.zeros(len(X_ALL), bool); keep[np.where(tr)[0][i1]] = True; keep[np.where(va)[0][i2]] = True; keep[np.where(te)[0][i3]] = True
    X_ALL, Y_ALL, G_ALL, S_ALL = X_ALL[keep], Y_ALL[keep], G_ALL[keep], S_ALL[keep]
np.savez(f"{RES}/dedup_splits.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

# ----------------------------------------------------------------------------- metrics
def best_threshold(y, p):
    """Threshold maximising macro-F1 on the VALIDATION data (ties -> closest to 0.5)."""
    thr = THR_GRID; pred = p[None, :] >= thr[:, None]; ym = (y == 1)[None, :]; yb = (y == 0)[None, :]
    tp1 = (pred & ym).sum(1); fp1 = (pred & yb).sum(1); fn1 = (~pred & ym).sum(1)
    tp0 = (~pred & yb).sum(1); fp0 = (~pred & ym).sum(1); fn0 = (pred & yb).sum(1)
    f = (2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1) + 2 * tp0 / np.maximum(2 * tp0 + fp0 + fn0, 1)) / 2
    cand = np.where(f >= f.max() - 1e-12)[0]; i = cand[np.argmin(np.abs(thr[cand] - 0.5))]
    return float(thr[i]), float(f[i])

def macro_f1(y, pred):
    out = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))

def evaluate(p, y, thr=0.5, hard=False):
    p = np.asarray(p, float); y = np.asarray(y); pred = (p >= thr).astype(int); p05 = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average=None, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {"threshold": float(thr), "accuracy": float((pred == y).mean()), "macro_f1": macro_f1(y, pred),
           "benign_precision": float(prec[0]), "benign_recall": float(rec[0]), "malware_precision": float(prec[1]), "malware_recall": float(rec[1]),
           "confusion_matrix": cm.tolist(), "macro_f1_at0.5": macro_f1(y, p05),
           "benign_recall_at0.5": float(((p05 == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
           "malware_recall_at0.5": float(((p05 == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
           "benign_prevalence_baseline": float((y == 0).mean())}
    if hard:
        for k in ["roc_auc", "benign_average_precision", "malware_average_precision", "fpr_at_tpr0.99", "actual_tpr_near_0.99", "fpr_at_tpr0.95", "actual_tpr_near_0.95"]: out[k] = float("nan")
        return out
    fpr, tpr, _ = roc_curve(y, p); out["roc_auc"] = float(auc(fpr, tpr))
    out["benign_average_precision"] = float(average_precision_score((y == 0).astype(int), 1 - p)); out["malware_average_precision"] = float(average_precision_score(y, p))
    for t in (0.99, 0.95):
        i = int(np.argmax(tpr >= t)) if np.any(tpr >= t) else -1
        out[f"fpr_at_tpr{t}"] = float(fpr[i]); out[f"actual_tpr_near_{t}"] = float(tpr[i])
    return out

def wilson(s, n, z=1.959963984540054):
    if n == 0: return [float("nan")] * 2
    p = s / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]

def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "values": [r[k] for r in rows]} for k in keys}

# ----------------------------------------------------------------------------- models
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6); self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate); self.drop2 = layers.Dropout(rate)
    def call(self, x, training=False, return_attention=False):
        a, sc = self.att(x, x, return_attention_scores=True); a = self.drop1(a, training=training)
        o1 = self.ln1(x + a); f = self.drop2(self.ffn(o1), training=training); o2 = self.ln2(o1 + f)
        return (o2, sc) if return_attention else o2
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kw):
        super().__init__(**kw); self.token_emb = layers.Embedding(vocab_size, embed_dim); self.pos_emb = layers.Embedding(seq_len, embed_dim)
    def call(self, x): return self.token_emb(x) + self.pos_emb(tf.range(start=0, limit=tf.shape(x)[-1], delta=1))
CUSTOM = {"TransformerBlock": TransformerBlock, "PositionalEmbedding": PositionalEmbedding}

def build_deep(name, lr, seed):
    keras.utils.set_random_seed(seed); c = CONFIG
    inp = layers.Input(shape=(c["seq_len"],))
    if name == "cnn":
        x = layers.Embedding(c["vocab_size"], c["embed_dim"])(inp)
        x = layers.Conv1D(128, 5, activation="relu", padding="same")(x); x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation="relu", padding="same")(x); x = layers.GlobalMaxPooling1D()(x)
    elif name == "bilstm":   # CHANGED: return_sequences + max-pooling over time
        x = layers.Embedding(c["vocab_size"], c["embed_dim"])(inp)
        x = layers.Bidirectional(layers.LSTM(c["lstm_hidden"], return_sequences=True))(x); x = layers.GlobalMaxPooling1D()(x)
    elif name in ("transformer", "transformer_1layer"):
        n = 1 if name == "transformer_1layer" else c["num_transformer_layers"]
        x = PositionalEmbedding(c["seq_len"], c["vocab_size"], c["embed_dim"])(inp)
        for _ in range(n): x = TransformerBlock(c["embed_dim"], c["num_heads"], c["ff_dim"])(x)
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x); out = layers.Dense(2, activation="softmax")(x)
    m = keras.Model(inp, out); m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"]); return m

def kp(model, X): return model.predict(X, verbose=0, batch_size=512)[:, 1]

def class_weight_at_strength(y, s):
    y = np.asarray(y); w0 = len(y) / (2 * max(1, int((y == 0).sum()))); w1 = len(y) / (2 * max(1, int((y == 1).sum())))
    return {0: float(1 + s * (w0 - 1)), 1: float(1 + s * (w1 - 1))}

# attack sets (fixed once): padding with the 15 most frequent benign calls from the TRAIN split
def make_adv(X, ids, n, seed):
    r = np.random.default_rng(seed); Xa = np.asarray(X).copy(); n = min(n, Xa.shape[1] - 20)
    for i in range(len(Xa)):
        pos = r.choice(np.arange(20, Xa.shape[1]), size=n, replace=False); Xa[i, pos] = r.choice(ids, size=n, replace=True)
    return Xa
TOP_BENIGN = [c for c, _ in collections.Counter(X_train[y_train == 0].flatten().tolist()).most_common(15)]
XM, XB = X_test[y_test == 1], X_test[y_test == 0]
ATT = {0: (XM, XB)}
for n in ATTACK_N: ATT[n] = (make_adv(XM, TOP_BENIGN, n, 2026 + n), make_adv(XB, TOP_BENIGN, n, 4026 + n))
ADV_KEYS = sorted(ATT)

def runfile(model, lr, seed, tag): return f"{RES}/runs/{model}__lr{lr:g}__seed{seed}__{tag}.npz"

def train_deep(model, lr, seed, Xf, yf, Xv, yv, Xt, tag="main", cw=None, adv=False, save=None):
    """Train one deep model (cached). Returns dict(val, test, time, epochs, adv_m, adv_b)."""
    f = runfile(model, lr, seed, tag)
    if os.path.exists(f):
        d = np.load(f)
        if (not adv or "adv_m" in d.files) and (save is None or os.path.exists(save)):
            return {k: d[k] for k in d.files}
    m = build_deep(model, lr, seed)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=CONFIG["early_stopping_patience"], restore_best_weights=True)
    t0 = time.time()
    h = m.fit(Xf, yf, validation_data=(Xv, yv), batch_size=CONFIG["batch_size"], epochs=CONFIG["epochs"], class_weight=cw, callbacks=[es], verbose=0)
    out = {"val": kp(m, Xv), "test": kp(m, Xt), "time": np.array(time.time() - t0), "epochs": np.array(len(h.history["loss"]))}
    if adv:
        out["adv_m"] = np.stack([kp(m, ATT[n][0]) for n in ADV_KEYS]); out["adv_b"] = np.stack([kp(m, ATT[n][1]) for n in ADV_KEYS])
    if save: m.save(save)
    np.savez(f, **out); keras.backend.clear_session(); return out

def docs(X): return [" ".join(str(t) for t in r) for r in X]

def fit_classical(name, hp, seed, Xf, yf):
    """Returns predict(X)->P(malware). RF/XGB use n-gram COUNTS; LogReg uses TF-IDF n-grams."""
    if name in ("random_forest", "xgboost"):
        vec = CountVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", max_features=5000, lowercase=False)
        F = vec.fit_transform(docs(Xf)).astype(np.float32)
        if name == "random_forest": mdl = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=seed)
        else: mdl = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=hp["learning_rate"], subsample=0.8, colsample_bytree=0.8,
                                       n_jobs=XGB_THREADS, random_state=seed, eval_metric="logloss", tree_method="hist")
    else:
        vec = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", max_features=5000)
        F = vec.fit_transform(docs(Xf)); mdl = LogisticRegression(C=hp["C"], max_iter=1000, random_state=seed)
    mdl.fit(F, yf)
    return lambda X: mdl.predict_proba(vec.transform(docs(X)))[:, 1]

def nn1_predict(Xf, yf, Xq, chunk=250):
    S = ["".join(chr(256 + int(t)) for t in r) for r in Xf]; Q = ["".join(chr(256 + int(t)) for t in r) for r in Xq]; arg = np.empty(len(Q), int); dm = np.empty(len(Q), int)
    for i in range(0, len(Q), chunk):
        D = cdist(Q[i:i + chunk], S, scorer=Levenshtein.distance, dtype=np.int32, workers=-1); arg[i:i + chunk] = D.argmin(1); dm[i:i + chunk] = D.min(1)
    return yf[arg].astype(float), dm

def tuning_load():
    f = f"{RES}/tuning.json"; return json.load(open(f)) if os.path.exists(f) else {}
def tuning_save(t): json.dump(t, open(f"{RES}/tuning.json", "w"), indent=2)

def get_hp(name):
    t = tuning_load()
    if name not in t: raise SystemExit(f"Run tier A first (no tuning found for {name}).")
    return t[name]

# ----------------------------------------------------------------------------- generic: one model on a given (fit, val, test) triple
def run_model(name, seed, Xf, yf, Xv, yv, Xt, yt, tag, hp):
    """Fit `name` with chosen hp, tune the threshold on validation, evaluate on test. Returns metrics dict."""
    if name == "nn1":
        pt, _ = nn1_predict(Xf, yf, Xt); r = evaluate(pt, yt, 0.5, hard=True); r.update({"seed": seed, "train_time_sec": 0.0}); return r
    t0 = time.time()
    if name in DEEP:
        o = train_deep(name, hp["lr"], seed, Xf, yf, Xv, yv, Xt, tag=tag); pv, pt, tt = o["val"], o["test"], float(o["time"])
    else:
        pr = fit_classical(name, hp, seed, Xf, yf); pv, pt = pr(Xv), pr(Xt); tt = time.time() - t0
    thr, vf = best_threshold(yv, pv); r = evaluate(pt, yt, thr); r.update({"seed": seed, "train_time_sec": tt, "val_macro_f1_tuned": vf})
    r["val_macro_f1_at0.5"] = macro_f1(yv, (pv >= 0.5).astype(int)); return r

# ============================================================================= tiers
def tier_check():
    log("python", sys.version.split()[0], "| tensorflow", tf.__version__, "| keras", keras.__version__, "| platform", platform.platform())
    log("GPU:", tf.config.list_physical_devices("GPU"), "| determinism:", not args.no_determinism)
    sub = np.arange(min(1500, len(X_train)))
    pa = pb = None
    for k in range(2):
        m = build_deep("cnn", 1e-3, 42); m.fit(X_train[sub], y_train[sub], batch_size=256, epochs=2, verbose=0, shuffle=True)
        p = kp(m, X_test[:200]); pa, pb = (p, pb) if k == 0 else (pa, p); keras.backend.clear_session()
    log("determinism test (same seed twice): max |diff| =", float(np.abs(pa - pb).max()), "-> OK" if float(np.abs(pa - pb).max()) == 0 else "-> NOT IDENTICAL (see README: re-run with --no-determinism and tell me)")
    est = 0
    for name, runs in [("cnn", 14 + 40 + 20 + 10 * 10 / 10 * 0), ("bilstm", 14), ("transformer", 14), ("transformer_1layer", 14)]:
        m = build_deep(name, 1e-3, 42); t0 = time.time(); m.fit(X_train, y_train, batch_size=256, epochs=1, verbose=0); et = time.time() - t0; keras.backend.clear_session()
        est += et * 12 * runs; log(f"  one epoch of {name}: {et:.0f}s -> {runs} runs x ~12 epochs ~ {et*12*runs/3600:.1f} h")
    log(f"rough total for tier A (+C for the CNN): ~{est/3600:.1f} h; tier B ~ 0.7x tier A's three main deep models; tier D ~ 0.35x. Real times vary.")

def tier_A():
    only = [m for m in args.models.split(",") if m] or MAIN_MODELS
    tuning = tuning_load(); allres = json.load(open(f"{RES}/step3_all_models.json")) if os.path.exists(f"{RES}/step3_all_models.json") else {}
    for name in only:
        t0 = time.time(); log(f"=== TIER A: {name} ===")
        if name == "nn1":
            rows = [run_model("nn1", s, X_train, y_train, X_val, y_val, X_test, y_test, "main", {}) for s in SEEDS[:1]]
            rows = [dict(rows[0], seed=s) for s in SEEDS]   # deterministic: identical for every seed
        elif name in DEEP:
            if name not in tuning:
                tab = []
                for lr in LR_GRID:
                    for s in TUNE_SEEDS:
                        o = train_deep(name, lr, s, X_train, y_train, X_val, y_val, X_test, "main", adv=(name == "cnn"),
                                       save=(f"{RES}/models/{name}_lr{lr:g}_seed42.keras" if s == 42 else None))
                        thr, vf = best_threshold(y_val, o["val"]); tab.append({"lr": lr, "seed": s, "val_macro_f1_tuned_thr": vf, "val_macro_f1_at0.5": macro_f1(y_val, (o["val"] >= .5).astype(int))})
                        log(f"  tune {name} lr={lr:g} seed={s}: val macro-F1 (tuned thr) = {vf:.4f}")
                df = pd.DataFrame(tab); best = float(df.groupby("lr")["val_macro_f1_tuned_thr"].mean().idxmax())
                tuning[name] = {"lr": best, "table": tab}; tuning_save(tuning); log(f"  -> chosen lr for {name}: {best:g}")
            hp = tuning[name]; rows = []
            tm = f"{RES}/models/{name}_lr{hp['lr']:g}_seed42.keras"
            if os.path.exists(tm): shutil.copy(tm, f"{RES}/models/{name}_seed42.keras")
            for f_ in os.listdir(f"{RES}/models"):
                if f_.startswith(f"{name}_lr") and f_.endswith("_seed42.keras"): os.remove(f"{RES}/models/{f_}")
            for s in SEEDS:
                sv = f"{RES}/models/{name}_seed42.keras" if s == 42 else None
                o = train_deep(name, hp["lr"], s, X_train, y_train, X_val, y_val, X_test, "main", adv=(name == "cnn"), save=sv)
                thr, vf = best_threshold(y_val, o["val"]); r = evaluate(o["test"], y_test, thr)
                r.update({"seed": s, "train_time_sec": float(o["time"]), "epochs_run": int(o["epochs"]), "lr": hp["lr"], "val_macro_f1_tuned": vf, "val_macro_f1_at0.5": macro_f1(y_val, (o["val"] >= .5).astype(int))})
                if s == 42: np.save(f"{RES}/{name}_seed42_probs.npy", o["test"]); np.save(f"{RES}/{name}_seed42_val_probs.npy", o["val"])
                rows.append(r); log(f"  {name} seed={s}: macro-F1={r['macro_f1']:.4f} (thr {thr:.2f}; @0.5 {r['macro_f1_at0.5']:.4f}) benign recall={r['benign_recall']:.3f}")
        else:   # classical
            if name not in tuning:
                grid = [{"learning_rate": v} for v in XGB_LR_GRID] if name == "xgboost" else [{"C": v} for v in LOGREG_C_GRID] if name == "tfidf_logreg" else [{}]
                tab = []
                for hp in grid:
                    pr = fit_classical(name, hp, 42, X_train, y_train); pv = pr(X_val); thr, vf = best_threshold(y_val, pv); tab.append({**hp, "val_macro_f1_tuned_thr": vf}); log(f"  tune {name} {hp}: val macro-F1 = {vf:.4f}")
                bi = int(np.argmax([t["val_macro_f1_tuned_thr"] for t in tab])); tuning[name] = {**grid[bi], "table": tab}; tuning_save(tuning)
            hp = tuning[name]; rows = []
            seeds_here = [SEEDS[0]] if name == "tfidf_logreg" else SEEDS
            for s in seeds_here:
                rows.append(run_model(name, s, X_train, y_train, X_val, y_val, X_test, y_test, "main", hp)); log(f"  {name} seed={s}: macro-F1={rows[-1]['macro_f1']:.4f} benign recall={rows[-1]['benign_recall']:.3f}")
            if name == "tfidf_logreg": rows = [dict(rows[0], seed=s) for s in SEEDS]   # deterministic
            for s in SEEDS[:1]: np.save(f"{RES}/{name}_seed42_probs.npy", fit_classical(name, hp, 42, X_train, y_train)(X_test))
        allres[name] = {"per_seed": rows, "summary": summarize(rows)}; json.dump(allres, open(f"{RES}/step3_all_models.json", "w"), indent=2)
        log(f"  {name}: macro-F1 {allres[name]['summary']['macro_f1']['mean']:.4f} +- {allres[name]['summary']['macro_f1']['std']:.4f} ({time.time()-t0:.0f}s)")
    log("TIER A done.")

def tier_B():
    out_rows = []; hp = {m: get_hp(m) for m in CV_MODELS if m != "nn1"}
    for rep in range(CV_REPEATS):
        sp = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=1000 + rep)
        for fold, (otr, ote) in enumerate(sp.split(X_ALL, Y_ALL, groups=G_ALL), start=1):
            Xo, yo, go = X_ALL[otr], Y_ALL[otr], G_ALL[otr]; Xt, yt = X_ALL[ote], Y_ALL[ote]
            assert not (set(go) & set(G_ALL[ote]))
            li, vi = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=2000 + rep * 10 + fold).split(Xo, yo, groups=go))
            Xf, yf, Xv, yv = Xo[li], yo[li], Xo[vi], yo[vi]; seed = SEEDS[(rep * CV_SPLITS + fold - 1) % len(SEEDS)]
            for name in CV_MODELS:
                cf = f"{RES}/cv/{name}__r{rep+1}f{fold}.json"
                if os.path.exists(cf): out_rows.append(json.load(open(cf))); continue
                r = run_model(name, seed, Xf, yf, Xv, yv, Xt, yt, f"cv_r{rep+1}f{fold}", hp.get(name, {}))
                r.update({"model": name, "repeat": rep + 1, "fold": fold}); json.dump(r, open(cf, "w")); out_rows.append(r)
                log(f"  CV r{rep+1} f{fold} {name}: macro-F1={r['macro_f1']:.4f} benign recall={r['benign_recall']:.3f}")
    df = pd.DataFrame(out_rows); df.drop(columns=["confusion_matrix"], errors="ignore").to_csv(f"{RES}/step4B_repeated_group_cv_per_run.csv", index=False)
    s = df.groupby("model")[["macro_f1", "benign_recall", "malware_recall", "benign_average_precision", "roc_auc"]].agg(["mean", "std"]); s.columns = ["_".join(c) for c in s.columns]
    s.to_csv(f"{RES}/step4B_repeated_group_cv_summary.csv"); log(s.round(4).to_string()); log("TIER B done.")

def adv_metrics(o, thr):
    m = {}
    for i, n in enumerate(ADV_KEYS):
        m[f"adv_malware_recall_{n}"] = float((o["adv_m"][i] >= thr).mean()); m[f"adv_benign_recall_{n}"] = float((o["adv_b"][i] < thr).mean())
        m[f"adv_malware_recall_{n}_at0.5"] = float((o["adv_m"][i] >= 0.5).mean()); m[f"adv_benign_recall_{n}_at0.5"] = float((o["adv_b"][i] < 0.5).mean())
    return m

def tier_C():
    hp = get_hp("cnn"); lr = hp["lr"]; sweep = []
    for s in WEIGHT_STRENGTHS:
        for seed in SEEDS:
            tag = "main" if s == 0 else f"sweep_s{s:g}"
            o = train_deep("cnn", lr, seed, X_train, y_train, X_val, y_val, X_test, tag, cw=(class_weight_at_strength(y_train, s) if s > 0 else None), adv=True)
            thr, vf = best_threshold(y_val, o["val"]); r = evaluate(o["test"], y_test, thr)
            r.update({"weighting_strength": s, "seed": seed, "val_macro_f1_tuned": vf, "val_macro_f1_at0.5": macro_f1(y_val, (o["val"] >= .5).astype(int)), **adv_metrics(o, thr)})
            sweep.append(r); log(f"  sweep s={s} seed={seed}: test macro-F1@0.5={r['macro_f1_at0.5']:.4f} tuned={r['macro_f1']:.4f} benign recall@0.5={r['benign_recall_at0.5']:.3f}")
    json.dump(sweep, open(f"{RES}/step5_weighting_ablation.json", "w"), indent=2)
    sdf = pd.DataFrame(sweep); gb = sdf.groupby("weighting_strength")
    summ = pd.DataFrame({"val_macro_f1_at0.5_mean": gb["val_macro_f1_at0.5"].mean(), "val_macro_f1_tuned_mean": gb["val_macro_f1_tuned"].mean(),
                         "test_macro_f1_at0.5_mean": gb["macro_f1_at0.5"].mean(), "test_macro_f1_tuned_mean": gb["macro_f1"].mean(),
                         "test_benign_recall_at0.5_mean": gb["benign_recall_at0.5"].mean(), "test_benign_recall_tuned_mean": gb["benign_recall"].mean(),
                         "test_malware_recall_at0.5_mean": gb["malware_recall_at0.5"].mean(), "test_malware_recall_tuned_mean": gb["malware_recall"].mean()}).reset_index()
    best_any = float(summ.loc[summ["val_macro_f1_at0.5_mean"].idxmax(), "weighting_strength"])        # best strength on VALIDATION incl. "none"
    sstar = float(summ[summ.weighting_strength > 0].sort_values("val_macro_f1_at0.5_mean", ascending=False).iloc[0]["weighting_strength"])   # best POSITIVE strength on validation, used for the 2x2
    summ.to_csv(f"{RES}/step5_weighting_summary.csv", index=False)
    json.dump({"best_strength_on_validation_including_none": best_any, "chosen_positive_strength_on_validation": sstar}, open(f"{RES}/step5_chosen_strength.json", "w"))
    log(f"  best strength on validation (incl. none) = {best_any}; best positive strength used for the 2x2 = {sstar}")
    log(summ.round(4).to_string()); log(f"  weighting strength chosen on VALIDATION: {sstar}")
    cmp = []
    for lab, s in [("unweighted", 0.0), (f"weighted s={sstar:g}", sstar)]:
        g = sdf[sdf.weighting_strength == s]
        for thr_mode, mf, br, mr in [("0.5", "macro_f1_at0.5", "benign_recall_at0.5", "malware_recall_at0.5"), ("val-tuned", "macro_f1", "benign_recall", "malware_recall")]:
            cmp.append({"model": lab, "threshold": thr_mode, "macro_f1_mean": g[mf].mean(), "macro_f1_sd": g[mf].std(ddof=1), "benign_recall_mean": g[br].mean(), "benign_recall_sd": g[br].std(ddof=1), "malware_recall_mean": g[mr].mean()})
    pd.DataFrame(cmp).to_csv(f"{RES}/step5_weighting_vs_threshold.csv", index=False); log(pd.DataFrame(cmp).round(4).to_string())
    rows = []; cond = {"hardening_0_weighting_0": (False, 0.0), "hardening_0_weighting_1": (False, sstar), "hardening_1_weighting_0": (True, 0.0), "hardening_1_weighting_1": (True, sstar)}
    for cname, (harden, s) in cond.items():
        for seed in SEEDS:
            if harden:
                Xa = make_adv(X_train[y_train == 1], TOP_BENIGN, HARDEN_INJECT, seed); Xf = np.concatenate([X_train, Xa]); yf = np.concatenate([y_train, np.ones(len(Xa), "int32")])
            else: Xf, yf = X_train, y_train
            tag = ("main" if (not harden and s == 0) else f"sweep_s{s:g}" if not harden else f"h1_s{s:g}")
            sv = f"{RES}/models/hardened_cnn_seed42.keras" if (harden and s > 0 and seed == 42) else None
            o = train_deep("cnn", lr, seed, Xf, yf, X_val, y_val, X_test, tag, cw=(class_weight_at_strength(yf, s) if s > 0 else None), adv=True, save=sv)
            thr, vf = best_threshold(y_val, o["val"]); r = evaluate(o["test"], y_test, thr)
            r.update({"condition": cname, "hardening": harden, "weighting": s > 0, "weighting_strength": s, "seed": seed, **adv_metrics(o, thr)}); rows.append(r)
            log(f"  {cname} seed={seed}: macro-F1@0.5={r['macro_f1_at0.5']:.4f} benign recall@0.5={r['benign_recall_at0.5']:.3f} adv recall@{ADV_KEYS[-1]}={r[f'adv_malware_recall_{ADV_KEYS[-1]}_at0.5']:.3f}")
    df = pd.DataFrame(rows); df.drop(columns=["confusion_matrix"]).to_csv(f"{RES}/step7_2x2_ablation_per_seed.csv", index=False)
    df.groupby("condition").mean(numeric_only=True).to_csv(f"{RES}/step7_2x2_ablation_summary.csv"); log("TIER C done.")

def tier_D():
    if not os.path.exists(args.raw) and not SMOKE: raise SystemExit(f"Raw dataset {args.raw} not found (put the Oliveira CSV next to this script, or pass --raw PATH).")
    cols = [f"t_{i}" for i in range(100)]
    if os.path.exists(args.raw): raw = pd.read_csv(args.raw); Xr, yr = raw[cols].to_numpy("int32"), raw["malware"].to_numpy("int32")
    else: r = np.random.default_rng(0); ii = r.integers(0, len(X_ALL), 3000); Xr, yr = X_ALL[ii], Y_ALL[ii]   # smoke only
    n_tr, n_va, n_te = len(X_train), len(X_val), len(X_test); nb_all = int((np.concatenate([y_train, y_val, y_test]) == 0).sum()); nm_all = len(X_train) + len(X_val) + len(X_test) - nb_all
    rng = np.random.default_rng(42)
    def draw(X, y):     # sample the same class counts as the clean dataset, then stratified random split with the same sizes
        ib = rng.choice(np.where(y == 0)[0], min(nb_all, int((y == 0).sum())), replace=False); im = rng.choice(np.where(y == 1)[0], min(nm_all, int((y == 1).sum())), replace=False)
        idx = np.concatenate([ib, im]); Xs, ys = X[idx], y[idx]
        a, b, ya, yb = train_test_split(Xs, ys, test_size=n_va + n_te, stratify=ys, random_state=42); v, t, yv_, yt_ = train_test_split(b, yb, test_size=n_te, stratify=yb, random_state=42)
        return (a, ya, v, yv_, t, yt_)
    splits = {"naive (random rows of the raw data)": draw(Xr, yr), "exact-dedup (unique sequences, random)": draw(X_ALL, Y_ALL),
              "group-aware (components kept together)": (X_train, y_train, X_val, y_val, X_test, y_test)}
    hp = {m: get_hp(m) for m in LEAK_MODELS if m != "nn1"}; rows = []; stats = []
    def key(r): return hashlib.md5(np.asarray(r, "int32").tobytes()).hexdigest()
    for sname, (a, ya, v, yv_, t, yt_) in splits.items():
        trh = {key(r) for r in a}; exact = float(np.mean([key(r) in trh for r in t]))
        S = ["".join(chr(256 + int(x)) for x in r) for r in a]; Q = ["".join(chr(256 + int(x)) for x in r) for r in t]; near = 0
        for i in range(0, len(Q), 250): near += int((cdist(Q[i:i + 250], S, scorer=Levenshtein.distance, score_cutoff=2, dtype=np.uint8, workers=-1).min(1) <= 2).sum())
        stats.append({"split": sname, "n_train": len(a), "n_val": len(v), "n_test": len(t), "test_benign": int((yt_ == 0).sum()), "test_exact_duplicate_of_train": exact, "test_within_2_edits_of_train": near / len(t)})
        log(f"  split '{sname}': test sequences identical to a training sequence = {exact*100:.1f}%, within 2 edits = {near/len(t)*100:.1f}%")
        for name in LEAK_MODELS:
            for seed in (LEAK_SEEDS if name not in ("nn1", "tfidf_logreg") else LEAK_SEEDS[:1]):
                tag = "main" if sname.startswith("group") else "leak_" + sname.split()[0]
                r = run_model(name, seed, a, ya, v, yv_, t, yt_, tag, hp.get(name, {})); r.update({"split": sname, "model": name}); rows.append(r)
                log(f"    {name} seed={seed}: macro-F1={r['macro_f1']:.4f} benign recall={r['benign_recall']:.3f}")
    df = pd.DataFrame(rows); df.drop(columns=["confusion_matrix"]).to_csv(f"{RES}/leakage_controlled_per_run.csv", index=False)
    sm = df.groupby(["split", "model"])[["macro_f1", "benign_recall", "malware_recall", "benign_average_precision"]].agg(["mean", "std"]); sm.columns = ["_".join(c) for c in sm.columns]
    sm.to_csv(f"{RES}/leakage_controlled_summary.csv"); json.dump(stats, open(f"{RES}/leakage_split_stats.json", "w"), indent=2); log(sm.round(4).to_string()); log("TIER D done.")

def tier_E():
    if not os.path.exists(args.cross): log(f"  {args.cross} not found - skipping tier E (put sample_analysis_data.txt next to this script)."); return
    calls = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oliveira_calls.json"))); idx = {n.lower(): i for i, n in enumerate(calls)}; PAD = 307
    seqs = []; n_short = 0
    for line in open(args.cross).read().splitlines():
        toks = []
        for t in line.split():   # v5: collapse consecutive repeated calls first, as in the Oliveira encoding ("first 100 non-repeated consecutive calls")
            if not toks or t.lower() != toks[-1]: toks.append(t.lower())
        ids = [idx.get(t, PAD) for t in toks[:100]]; n_short += len(ids) < 100; ids += [PAD] * (100 - len(ids)); seqs.append(ids)
    Xc = np.array(seqs, "int32"); n = len(Xc); log(f"  external sample: {n} sequences ({n_short} shorter than 100 calls after collapsing repeats, padded with id {PAD})")
    rows = []
    for name in ["bilstm", "transformer", "cnn", "random_forest", "xgboost", "tfidf_logreg"]:
        hp = get_hp(name)
        if name in DEEP:
            f = f"{RES}/models/{name}_seed42.keras"
            if not os.path.exists(f): log(f"  missing {f}; run tier A first"); continue
            p = kp(keras.models.load_model(f, custom_objects=CUSTOM), Xc); pv = np.load(f"{RES}/{name}_seed42_val_probs.npy")
        else:
            pr = fit_classical(name, hp, 42, X_train, y_train); p = pr(Xc); pv = pr(X_val)
        thr, _ = best_threshold(y_val, pv)
        for mode, th in [("val-tuned threshold", thr), ("threshold 0.5", 0.5)]:
            s = int((p >= th).sum()); lo, hi = wilson(s, n); rows.append({"model": name, "threshold_mode": mode, "threshold": th, "malware_recall": s / n, "wilson95_low": lo, "wilson95_high": hi, "n_external_malware": n})
    pd.DataFrame(rows).to_csv(f"{RES}/step8_crossdataset_wilson_intervals.csv", index=False); log(pd.DataFrame(rows).round(3).to_string()); log("TIER E done.")

def tier_package():
    base = f"{args.results}_package"; shutil.make_archive(base, "zip", RES); log(f"Created {base}.zip  ({os.path.getsize(base+'.zip')/1e6:.0f} MB) - send me this file.")

{"check": tier_check, "A": tier_A, "B": tier_B, "C": tier_C, "D": tier_D, "E": tier_E, "package": tier_package}[args.tier]()

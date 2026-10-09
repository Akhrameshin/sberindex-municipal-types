"""Поиск по (метод, K) с покластерной устойчивостью по Хеннигу на последнем месяце: 20 подвыборок по 80% МО."""
import sys, logging; sys.path.insert(0, "src"); sys.path.insert(0, "external/KEFRiN")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
import kefrin; logging.disable(logging.CRITICAL)
from config import CFG, SEED
from features import build
from pipeline import *
from evalkit import subsamples
d = build(); Xt, _ = month_features(d); X = Xt[-1]; N = len(X); rng = np.random.default_rng(SEED); SUB = subsamples(N, reps=20, frac=0.8)
def run(method, x, K, gam=None):
    a = graph(x)
    if method == "k-means": return KMeans(K, n_init=5, random_state=0).fit_predict(x)
    if method == "KEFRiNc": return kefrin.KEFRiNc(x, (a.toarray() > 0).astype(float), rho=1.0, xi=1.0, n_clusters=K, preprocessing_y="z_score", preprocessing_p="none")
    if method == "спектральный": return SpectralClustering(K, affinity="precomputed", random_state=0, assign_labels="cluster_qr").fit_predict(a)
    return snap(to_ig(a), gam)
def hennig(method, K, gam=None, R=20):
    base = run(method, X, K, gam); cl = sorted(set(base)); J = {c: [] for c in cl}
    for keep in SUB[:R]:                       # один и тот же набор подвыборок для всех методов и K (иначе разница включает шум выборки)
        lab = run(method, X[keep], K, gam)
        for c in cl:
            A = set(keep[base[keep] == c]); J[c].append(max(len(A & set(keep[lab == s])) / len(A | set(keep[lab == s])) for s in set(lab)))
    m = np.array([np.mean(v) for v in J.values()]); return len(cl), m.mean(), m.min(), int((m > .75).sum())
rows = []
for K in (3, 4, 5, 6, 7, 8):
    for mth in ("k-means", "KEFRiNc", "спектральный"):
        k, mean, mn, good = hennig(mth, K); rows.append(dict(метод=mth, K=k, J_среднее=mean, J_мин=mn, устойчивых_из=f"{good}/{k}")); print(rows[-1], flush=True)
g = calibrate(to_ig(graph(X)), 5); 
for tgt in (3, 4, 5, 6, 7):
    gam = calibrate(to_ig(graph(X)), tgt); k, mean, mn, good = hennig("Leiden", None, gam); rows.append(dict(метод="Leiden (срез)", K=k, J_среднее=mean, J_мин=mn, устойчивых_из=f"{good}/{k}")); print(rows[-1], flush=True)
pd.DataFrame(rows).round(3).to_csv("data/processed/hennig_static.csv", index=False)

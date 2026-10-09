"""Подбор K для «общий k-means + Витерби»: Хеннинг (20 подвыборок 80%), глобальный ARI, SW, смены, число кластеров с J>0.75."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from config import CFG, SEED
from features import build
from pipeline import *
from methods2 import pooled_kmeans_viterbi
from icvi import silhouette
from evalkit import subsamples, jaccard_per_cluster
d = build(); Xt, _ = month_features(d); N = Xt[0].shape[0]; LAM = CFG["headline"]["lam"]
SUB = subsamples(N, reps=20, frac=0.8)      # один набор подвыборок на все K
rows = []
for K in (3, 4, 5, 6, 7, 8):
    base, km = pooled_kmeans_viterbi(Xt, K, LAM, seed=SEED); b = base[-1]; cl = sorted(set(b)); J = {c: [] for c in cl}; ar = []
    for r, keep in enumerate(SUB):
        lab = pooled_kmeans_viterbi([x[keep] for x in Xt], K, LAM, seed=r)[0][-1]
        ar.append(ARI(b[keep], lab))
        for c, v in jaccard_per_cluster(b, lab, keep).items(): J[c].append(v)
    m = np.array([np.nanmean(v) for v in J.values()])
    ch = np.mean([(base[t] != base[t + 1]).mean() for t in range(len(base) - 1)]); sw = np.mean([silhouette(Xt[t], base[t]) for t in (0, 10, 21)])
    rows.append(dict(K=K, J_среднее=m.mean(), J_мин=m.min(), устойчивых=f"{(m > .75).sum()}/{K}", ARI_подвыб=np.mean(ar), SW=sw, смен_мес=ch, смен_за_период=(base[0] != base[-1]).mean(), размеры=np.bincount(b).tolist()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(3).to_csv("data/processed/pooled_select.csv", index=False)

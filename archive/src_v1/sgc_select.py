"""Сколько шагов графового сглаживания брать: X̃ = Â^s X (SGC) + общий k-means + Витерби.

Сравнение s=0,1,2,3 на обоих уровнях K при λ из конфига. Хеннинг — на одном и том же наборе 8 подвыборок.
SW считается на ИСХОДНЫХ (несглаженных) признаках, AVI/ANUI/Q — на графе из исходных признаков, чтобы
сглаживание не оценивало само себя. s=0 — признаки без участия сети: это наш отрицательный контроль,
отвечающий на вопрос, добавляет ли сеть что-нибудь к атрибутам."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, scipy.sparse as sp
from config import CFG, SEED
from features import build
from pipeline import *
from methods2 import pooled_kmeans_viterbi
from icvi import graph_indices, silhouette
from evalkit import subsamples, jaccard_per_cluster
d = build(); Xt, _ = month_features(d); N = Xt[0].shape[0]; LAM = CFG["headline"]["lam"]
SUB = subsamples(N, reps=20, frac=0.8)     # один набор подвыборок на все s и K — тот же, что в stability.py и lam_select.py

def smooth(x, s):
    if s == 0: return x
    n = x.shape[0]; A = graph(x); A = A.maximum(A.T) + sp.eye(n); dg = np.asarray(A.sum(1)).ravel() ** -.5; Ah = sp.diags(dg) @ A @ sp.diags(dg)
    for _ in range(s): x = Ah @ x
    return x
def run(xs, K, s, seed=0): return pooled_kmeans_viterbi([smooth(x, s) for x in xs], K, LAM, seed=seed)[0]
Ag = [graph(x) for x in Xt]; rows = []
for s in (0, 1, 2, 3):
    Xs = [smooth(x, s) for x in Xt]
    for K in CFG["headline"]["Ks"]:
        base = pooled_kmeans_viterbi(Xs, K, LAM, seed=SEED)[0]; b = base[-1]; cl = sorted(set(b)); J = {c: [] for c in cl}
        for r, keep in enumerate(SUB):
            lab = run([x[keep] for x in Xt], K, s, r)[-1] if s == 0 else \
                pooled_kmeans_viterbi([smooth(x[keep], s) for x in Xt], K, LAM, seed=r)[0][-1]
            for c, v in jaccard_per_cluster(b, lab, keep).items(): J[c].append(v)
        m = np.array([np.nanmean(v) for v in J.values()]); gi = [graph_indices(Ag[t], base[t]) for t in (0, 10, 21)]
        rows.append(dict(s=s, K=K, J_среднее=m.mean(), J_мин=m.min(), SW=np.mean([silhouette(Xt[t], base[t]) for t in (0, 10, 21)]),
                         AVI=np.mean([g["AVI"] for g in gi]), ANUI=np.mean([g["ANUI"] for g in gi]), Q=np.mean([g["Q"] for g in gi]), смен_за_период=(base[0] != base[-1]).mean()))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).round(3).to_csv("data/processed/sgc_select.csv", index=False)

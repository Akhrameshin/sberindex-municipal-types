"""Выбор числа типов для заголовочного метода GS-TKM: K = 3…10, все индексы конкурса и устойчивость.
Для каждого K: SW, CH, S_Dbw, AVI, AVU, MQ, Q (среднее по трём месяцам), z относительно перестановки меток для AVI и AVU
(сырые AVI и AVU при разных K несопоставимы: у случайного разбиения AVI падает как 1/K), churn, Хеннинг на тех же 20 подвыборках, ARI подвыборок."""
import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from config import CFG, SEED
from features import build
from pipeline import month_features, graph
from headline import fit
from icvi import all_indices, graph_indices, DIRECTION
from evalkit import subsamples, jaccard_per_cluster

d = build(); Xt, _ = month_features(d); N = Xt[0].shape[0]; T = len(Xt); A = [graph(x) for x in Xt]
SUB = subsamples(N, reps=20, frac=0.8); rng = np.random.default_rng(SEED); rows = []
for K in range(3, 11):
    t0 = time.time(); L, _ = fit(Xt, K); b = L[-1]
    ev = [all_indices(Xt[t], A[t], L[t]) for t in (0, T // 2, T - 1)]
    z = {}
    for k in ("AVI", "AVU"):
        null = [graph_indices(A[T - 1], rng.permutation(b))[k] for _ in range(100)]
        z[f"z_{k}"] = DIRECTION[k] * (ev[-1][k] - np.mean(null)) / (np.std(null) + 1e-12)
    J = {c: [] for c in sorted(set(b))}; ar = []
    for r, keep in enumerate(SUB):
        lab = fit([x[keep] for x in Xt], K, seed=r)[0][-1]; ar.append(ARI(b[keep], lab))
        for c, v in jaccard_per_cluster(b, lab, keep).items(): J[c].append(v)
    m = np.array([np.nanmean(v) for v in J.values()])
    rows.append(dict(K=K, **{k: np.mean([e[k] for e in ev]) for k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q")}, **z,
                     churn=np.mean([(L[t] != L[t + 1]).mean() for t in range(T - 1)]), J_среднее=m.mean(), J_худший=m.min(),
                     устойчивых=f"{int((m > .75).sum())}/{len(m)}", ARI_подвыб=np.mean(ar), размеры=np.bincount(b, minlength=K).tolist()))
    print(f"K={K} ({time.time() - t0:.0f}с)", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).round(4).to_csv("data/processed/k_sweep.csv", index=False)

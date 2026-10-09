"""Устойчивость заголовочного метода (оба уровня K): (1) Хеннинг — покластерный Жаккар, 20 подвыборок по 80% МО;
(2) глобальный ARI на тех же подвыборках; (3) шум 0.15σ в признаках, 10 повторов: ARI и число сохранившихся кластеров."""
import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from config import CFG, SEED
from features import build
from pipeline import month_features
from headline import fit
from evalkit import subsamples, jaccard_per_cluster
d = build(); Xt, _ = month_features(d); N = Xt[0].shape[0]
SUB = subsamples(N, reps=20, frac=0.8)      # один набор подвыборок на все K
out = []
for K in ([int(a) for a in sys.argv[1:]] or CFG["headline"]["Ks"]):
    base = fit(Xt, K)[0][-1]; cl = sorted(set(base)); J = {c: [] for c in cl}; ar = []; t0 = time.time()
    for r, keep in enumerate(SUB):
        lab = fit([x[keep] for x in Xt], K, seed=r)[0][-1]; ar.append(ARI(base[keep], lab))
        for c, v in jaccard_per_cluster(base, lab, keep).items(): J[c].append(v)
    nz = []; nk = []
    for r in range(10):
        rr = np.random.default_rng(1000 + r); lab = fit([x + rr.normal(0, CFG["compare"]["noise_sigma"], x.shape) for x in Xt], K, seed=r)[0][-1]
        nz.append(ARI(base, lab)); nk.append(int((np.bincount(lab, minlength=K) > 0.01 * N).sum()))
    h = pd.DataFrame({"K": K, "кластер": cl, "n": [int((base == c).sum()) for c in cl], "Жаккар_среднее": [np.nanmean(J[c]) for c in cl], "Жаккар_мин": [np.nanmin(J[c]) for c in cl],
                      "доля_подвыборок>0.75": [np.nanmean(np.array(J[c]) > .75) for c in cl]})
    h["вердикт"] = pd.cut(h["Жаккар_среднее"], [0, .6, .75, .85, 1.01], labels=["нестабилен", "слабо", "устойчив", "высокоустойчив"])
    h.to_csv(f"data/processed/hennig_K{K}.csv", index=False); print(f"\nK={K}  ({time.time()-t0:.0f}с)"); print(h.round(3).to_string(index=False))
    out.append(dict(K=K, ARI_подвыборка80=np.mean(ar), ARI_подвыборка80_мин=np.min(ar), ARI_шум=np.mean(nz), ARI_шум_мин=np.min(nz), K_после_шума_мин=min(nk), K_после_шума_ср=np.mean(nk)))
pd.DataFrame(out).round(3).to_csv("data/processed/stability.csv", index=False); print(); print(pd.DataFrame(out).round(3).to_string(index=False))

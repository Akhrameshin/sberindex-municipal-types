"""Чувствительность заголовочного метода к штрафу Витерби λ — цена устойчивости.

λ управляет единственным гиперпараметром динамического слоя: штрафом за смену типа между месяцами
(в долях медианного расстояния до центра). λ=0 — независимые назначения по общим центроидам.
Таблица отвечает на прямой вопрос жюри: не куплена ли устойчивость запретом на изменения?
Колонки: churn — средняя доля МО, сменивших тип за месяц; пребывание — 1/churn, обрезано периодом наблюдения;
J_по_кластерам — средний по кластерам покластерный Жаккар (Хеннинг, те же 20 подвыборок по 80%, что в stability.py);
J_худший_кластер — тот же показатель у худшего кластера. Внимание: в stability.py колонка «Жаккар_мин» означает другое —
минимум ПО ПОДВЫБОРКАМ для данного кластера, а здесь минимум ПО КЛАСТЕРАМ. SW — силуэт; ARI_к_базе — согласие с λ из конфига."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from config import CFG, SEED
from features import build
from pipeline import month_features
from headline import fit
from icvi import silhouette
from evalkit import subsamples, jaccard_per_cluster

LAMS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
d = build(); Xt, months = month_features(d); N = Xt[0].shape[0]; T = len(Xt)
base_lam = CFG["headline"]["lam"]
SUB = subsamples(N, reps=20, frac=0.8)      # один набор подвыборок на все K и λ — тот же, что в stability.py
rows = []
for K in CFG["headline"]["Ks"]:
    base = fit(Xt, K, lam=base_lam)[0]
    for lam in LAMS:
        L = fit(Xt, K, lam=lam)[0]; last = L[-1]; cl = sorted(set(last))
        churn = np.mean([(L[t] != L[t + 1]).mean() for t in range(T - 1)])
        J = {c: [] for c in cl}
        for r, keep in enumerate(SUB):
            lab = fit([x[keep] for x in Xt], K, lam=lam, seed=r)[0][-1]
            for c, v in jaccard_per_cluster(last, lab, keep).items(): J[c].append(v)
        j = np.array([np.nanmean(v) for v in J.values()])
        rows.append(dict(K=K, λ=lam, churn=churn,
                         пребывание_мес=min(1 / churn, T) if churn > 0 else T,
                         пребывание_цензура=churn == 0 or 1 / churn > T,
                         J_по_кластерам=j.mean(), J_худший_кластер=j.min(), устойчивых=f"{int((j > .75).sum())}/{K}",
                         SW=np.mean([silhouette(Xt[t], L[t]) for t in (0, T // 2, T - 1)]),
                         смен_за_период=(L[0] != L[-1]).mean(), ARI_к_базе=ARI(base[-1], last)))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows); r.to_csv("data/processed/lam_select.csv", index=False)
pd.set_option("display.width", 220); print()
print(r.round(3).to_string(index=False))
print(f"\nВ конфиге λ={base_lam}. Цензура пребывания: при churn ниже 1/{T} оценка не отличима от «не менялся за период».")

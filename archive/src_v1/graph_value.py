"""Приносит ли сеть что-нибудь сверх атрибутов? Прямой тест на реальных данных, до результатов не подгоняется.

Гипотеза H2 (фиксируется до запуска): графовое сглаживание типов помогает предсказывать прирост 2024 к 2023 вне выборки ТОЛЬКО тогда, когда граф
построен из сигнала, которого нет в признаках разбиения. В синтетике (synthetic.py, режимы knn_graph и comp_graph) это так: граф из тех же признаков
не помогает, граф из дополняющего сигнала помогает.

Схема та же, что в outoftime.py: типы замораживаются по расходам ТОЛЬКО за 2023 год (режим spend, без Росстата и динамических дескрипторов);
цели — прирост log расходов и прирост доли маркетплейсов 2024 к 2023; метрика — OOS MAE, RepeatedKFold 5×10, одни и те же разбиения для всех вариантов.
Варианты графа для сглаживания (s=1): косинус профиля (те же признаки), DTW, лаговая корреляция, корреляция приростов, дорожная гравитация; контроль — без графа (s=0).
Для каждого варианта — изменение MAE относительно s=0 (парное, по разбиениям; интервал ±2 s.e. по 10 повторам) и согласие типов 2023 и 2024 годов (ARI).
Знак «−» у изменения MAE — граф помог."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import adjusted_rand_score as ARI
from config import CFG, SEED
from features import build
from pipeline import month_features
from headline import fit
from edge_rules import make_gfun, road_matrix

d = build(); ids = d["ids"]; N = len(ids); full = np.arange(N)
def year(sl):
    dd = dict(d); dd["tensor"] = d["tensor"][sl]; dd["months"] = d["months"][sl]; return dd
Xt23, _ = month_features(year(slice(0, 12)), lens="spend"); Xt24, _ = month_features(year(slice(12, 24)), lens="spend")
D = road_matrix(ids); T = d["tensor"]; sh = d["shares"]
y = {"прирост log расходов": T[12:, :, 0].mean(0) - T[:12, :, 0].mean(0), "прирост доли маркетплейсов": sh[12:, :, 3].mean(0) - sh[:12, :, 3].mean(0)}
# Подтверждающие цели: на них варианты графа НЕ отбирались. Первые две цели — поисковые (по ним мы впервые увидели, какой граф помогает).
CONFIRM = {"прирост доли общепита": 1, "прирост доли продовольствия": 2, "прирост доли здоровья": 0, "прирост доли транспорта": 4}
for nm_, ci in CONFIRM.items(): y[nm_] = sh[12:, :, ci].mean(0) - sh[:12, :, ci].mean(0)
ROLE = {k: ("подтверждающая" if k in CONFIRM else "поисковая") for k in y}
level = T[:12, :, 0].mean(0); reg = pd.Series(d["meta"].region_name.fillna("?").values, index=ids)
dm = lambda s: pd.get_dummies(s, drop_first=True).values.astype(float)
Rg = dm(reg); Lv = ((level - level.mean()) / level.std())[:, None]
folds = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(np.zeros((N, 1))))


def oos(X, t):
    out = []
    for tr, te in folds:
        A = np.c_[np.ones(len(tr)), X[tr]]; b = np.linalg.lstsq(A, t[tr], rcond=None)[0]; out.append(np.abs(t[te] - np.c_[np.ones(len(te)), X[te]] @ b).mean())
    return np.array(out)


GRAPHS = ["без графа (s=0)", "cos_feat", "dtw", "lag_corr", "corr_growth", "road_gravity"]
rows = []
for K in CFG["headline"]["Ks"]:
    types = {}
    for g in GRAPHS:
        if g.startswith("без"): L23 = fit(Xt23, K, s=0)[0][-1]; L24 = fit(Xt24, K, s=0)[0][-1]
        else: L23 = fit(Xt23, K, gfun=make_gfun(g, Xt23, full, D))[0][-1]; L24 = fit(Xt24, K, gfun=make_gfun(g, Xt24, full, D))[0][-1]
        types[g] = (L23, L24)
    for tn, t in y.items():
        base = {m: oos(X, t) for m, X in (("типы", dm(pd.Series(types[GRAPHS[0]][0]))), ("регионы + типы", np.c_[Rg, dm(pd.Series(types[GRAPHS[0]][0]))]), ("уровень + типы", np.c_[Lv, dm(pd.Series(types[GRAPHS[0]][0]))]))}
        for g in GRAPHS:
            Ty = dm(pd.Series(types[g][0]))
            for m, X in (("типы", Ty), ("регионы + типы", np.c_[Rg, Ty]), ("уровень + типы", np.c_[Lv, Ty])):
                a = oos(X, t); diff = (a - base[m]) / base[m].mean()           # парно по одним и тем же разбиениям
                per_rep = diff.reshape(10, 5).mean(1); se = per_rep.std(ddof=1) / np.sqrt(10)
                rows.append({"K": K, "цель": tn, "роль": ROLE[tn], "граф": g, "модель": m, "MAE": a.mean(), "изменение_MAE_против_s0": per_rep.mean(), "±2se": 2 * se,
                             "ARI_типов_2023_и_2024": ARI(*types[g]), "ARI_с_типами_без_графа": ARI(types[g][0], types[GRAPHS[0]][0])})
    print(f"K={K} готово", flush=True)
r = pd.DataFrame(rows); r.round(5).to_csv("data/processed/graph_value.csv", index=False)
pd.set_option("display.width", 250)
print(r[r.модель != "типы"].round(4).to_string(index=False))
